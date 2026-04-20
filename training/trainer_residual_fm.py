"""Residual flow-matching trainer (Track C).

Same structure as `trainer_fm.FMTrainer`, but the FM target is the residual
between the GT expression and a frozen stage-2 transformer's teacher-forced
prediction — normalized to unit std per dim using precomputed residual stats.

  r_raw  = expression - stage2(audio, emotion, prev_expr, target=expression)
  r_norm = (r_raw - residual_mean) / residual_std
  FM learns velocity field that maps N(0, I) -> r_norm.

The deterministic stage-2 head owns audio-correlated lip/jaw motion; the FM
only models audio-orthogonal variation (blinks, idle motion, personal tics).

Pre-requisite: residual_mean/residual_std in train.h5/stats — populate with
`scripts/compute_residual_stats.py`. The entry point loads them and passes
them into this trainer.
"""

from pathlib import Path

import torch
from tqdm import tqdm

from training.trainer_fm import FMTrainer
from utils.checkpoint import checkpoint_path, save_checkpoint
from utils.ddp import is_main_process
from utils.logging_utils import log_metrics


class ResidualFMTrainer(FMTrainer):
    """FM on residuals. Subclass of FMTrainer — reuses `_sample_t` and save/stop API."""

    def __init__(
        self,
        config,
        generator,
        optimizer,
        train_loader,
        val_loader,
        device,
        frozen_stage2,
        residual_std: torch.Tensor,
        residual_mean: torch.Tensor | None = None,
        wandb_run=None,
        dim_weights=None,
    ):
        super().__init__(
            config=config,
            generator=generator,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            wandb_run=wandb_run,
            dim_weights=dim_weights,
        )
        # Ref-clip eval requires summing stage2_pred + residual_sample — wiring
        # planned for a follow-up. V1 reports val loss only.
        self.ref_clip = None

        self.frozen_stage2 = frozen_stage2
        self.frozen_stage2.eval()
        for p in self.frozen_stage2.parameters():
            p.requires_grad_(False)

        # Shape [D]; broadcast over [B, T, D] at use time.
        self.residual_std = residual_std
        self.residual_mean = (
            residual_mean if residual_mean is not None
            else torch.zeros_like(residual_std)
        )

    @torch.no_grad()
    def _compute_residual(self, audio, emotion, prev_expr, expression):
        """Run frozen stage 2 (teacher-forced) and return normalized residual.

        All quantities are in expression-normalized space (z-scored by
        expr_mean/std from the dataset). The residual is further normalized
        by residual_std so its target distribution has std ~1 per dim.
        """
        pred = self.frozen_stage2(audio, emotion, prev_expr, target_expression=expression)
        r_raw = expression - pred
        return (r_raw - self.residual_mean) / self.residual_std

    def _extract_residual_window(self, r_full, audio, start: int):
        """Slice a `window_size`-frame sub-window of the residual sequence.

        Differs from FMTrainer._extract_window: prev conditioning comes from
        the residual itself (not full expression), so the FM sees residual
        context, not audio-correlated context. For start=0, prev is zeros
        (no prior residual in this clip).
        """
        W = self.config.fm.window_size
        P = self.config.data.prev_frames
        C = self.config.data.context_past
        F = self.config.data.context_future

        r_1 = r_full[:, start:start + W]
        audio_win = audio[:, start:start + C + W + F]

        B, _, D = r_full.shape
        if start >= P:
            prev_res = r_full[:, start - P:start]
        elif start == 0:
            prev_res = torch.zeros(B, P, D, device=r_full.device, dtype=r_full.dtype)
        else:
            pad = torch.zeros(B, P - start, D, device=r_full.device, dtype=r_full.dtype)
            prev_res = torch.cat([pad, r_full[:, :start]], dim=1)
        return r_1, audio_win, prev_res

    def _fm_loss_on_residual(self, r_win, audio_win, emotion, prev_res):
        B = r_win.shape[0]
        x_0 = torch.randn_like(r_win)
        t = self._sample_t(B)
        t_bhw = t[:, None, None]
        x_t = (1.0 - t_bhw) * x_0 + t_bhw * r_win
        u_target = r_win - x_0
        u_pred = self.generator(x_t, t, audio_win, emotion, prev_res)
        sq = (u_pred - u_target) ** 2
        if self.dim_weights is not None:
            sq = sq * self.dim_weights
        return sq.mean()

    def train_epoch(self, epoch: int) -> float:
        self.generator.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())

        T = self.config.data.seq_len
        W = self.config.fm.window_size

        for batch in pbar:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            r_full = self._compute_residual(audio, emotion, prev_expr, expression)

            start = int(torch.randint(0, max(T - W + 1, 1), (1,)).item())
            r_win, audio_win, prev_res = self._extract_residual_window(r_full, audio, start)

            loss = self._fm_loss_on_residual(r_win, audio_win, emotion, prev_res)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.config.fm.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}", start=start)
                if self.global_step % 50 == 0:
                    log_metrics(
                        {
                            "residual_fm/train_loss": loss.item(),
                            "residual_fm/lr": self.optimizer.param_groups[0]["lr"],
                            "residual_fm/window_start": start,
                        },
                        step=self.global_step,
                        run=self.wandb_run,
                    )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.generator.eval()
        total_loss = 0.0
        num_batches = 0

        T = self.config.data.seq_len
        W = self.config.fm.window_size

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            r_full = self._compute_residual(audio, emotion, prev_expr, expression)
            start = 0
            r_win, audio_win, prev_res = self._extract_residual_window(r_full, audio, start)
            loss = self._fm_loss_on_residual(r_win, audio_win, emotion, prev_res)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if is_main_process():
            log_metrics(
                {"residual_fm/val_loss": avg_loss, "residual_fm/epoch": epoch},
                step=self.global_step,
                run=self.wandb_run,
            )
            print(f"  Val Residual FM Loss: {avg_loss:.4f}")

        return avg_loss

    def save(self, epoch: int, val_loss: float):
        if not is_main_process():
            return

        from dataclasses import asdict

        path = checkpoint_path(self.config.paths.checkpoint_dir, epoch, prefix="residual_fm")
        save_checkpoint(
            path, epoch, val_loss, asdict(self.config),
            global_step=self.global_step,
            generator=(self.generator, self.optimizer),
        )
        print(f"  Checkpoint saved: {path}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            best_path = str(Path(self.config.paths.checkpoint_dir) / "residual_fm_best.pt")
            save_checkpoint(
                best_path, epoch, val_loss, asdict(self.config),
                global_step=self.global_step,
                generator=(self.generator, self.optimizer),
            )
            print(f"  New best model saved: {best_path}")
        else:
            self.epochs_without_improvement += 1
