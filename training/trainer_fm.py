"""Flow-matching training loop.

Parallel track to `trainer_stage2` and `trainer_stage3`. Same dataset, same
audio encoder, same reference-clip evaluation — the only differences are the
target (noised windows + velocity regression) and inference (Euler integration
over t ∈ [0, 1]).

Best-checkpoint selection uses best-of-K raw-space L1 on the reference clip:
the FM generator is stochastic, so reporting a single sample's L1 against a
deterministic baseline would be unfair. K samples are drawn, min raw-L1 is
taken. Per-dim std ratios are reported from the best-L1 sample.
"""

from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.ref_clip import (
    compute_ref_metrics,
    fm_slide_inference,
    prepare_ref_clip,
)
from utils.checkpoint import checkpoint_path, save_checkpoint
from utils.ddp import is_main_process
from utils.logging_utils import log_metrics


class FMTrainer:
    """Rectified-flow training for the FM generator."""

    def __init__(
        self,
        config,
        generator,
        optimizer,
        train_loader,
        val_loader,
        device,
        wandb_run=None,
        dim_weights=None,
    ):
        self.config = config
        self.generator = generator
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run
        self.dim_weights = dim_weights
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Reuse the same ref-clip infra as Stage 2/3 for apples-to-apples A/B.
        self.ref_clip = None
        if is_main_process() and config.stage2.ref_clip_audio_path:
            self.ref_clip = prepare_ref_clip(
                config, self.train_loader.dataset.stats, device
            )

    def _sample_t(self, batch_size: int) -> torch.Tensor:
        if self.config.fm.t_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(batch_size, device=self.device))
        return torch.rand(batch_size, device=self.device)

    def _extract_window(self, expression, audio, prev_expr, start: int):
        """Slice a training batch to a single `window_size`-frame sub-window.

        The 60-frame training clip gives us several valid sub-windows. One is
        chosen per mini-batch (same `start` across the batch — different `start`
        per sample complicates tensor slicing without adding much signal).
        """
        W = self.config.fm.window_size
        P = self.config.data.prev_frames
        C = self.config.data.context_past
        F = self.config.data.context_future

        x_1 = expression[:, start:start + W]
        audio_win = audio[:, start:start + C + W + F]
        if start == 0:
            prev_win = prev_expr
        else:
            full_expr = torch.cat([prev_expr, expression], dim=1)
            prev_win = full_expr[:, start:start + P]
        return x_1, audio_win, prev_win

    def _fm_loss(self, x_1, audio, emotion, prev):
        """Rectified-flow loss: MSE(u_pred, x_1 - x_0) on x_t = (1-t)x_0 + t x_1.

        Loss is optionally per-dim-std weighted to match the transformer track's
        dim-weighted L1 (so dims with larger natural range contribute more, in
        line with the perceptual-importance weighting used elsewhere).
        """
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = self._sample_t(B)  # [B]
        t_bhw = t[:, None, None]
        x_t = (1.0 - t_bhw) * x_0 + t_bhw * x_1
        u_target = x_1 - x_0
        u_pred = self.generator(x_t, t, audio, emotion, prev)
        sq = (u_pred - u_target) ** 2  # [B, W, D]
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

            # Pick one random sub-window offset for this batch.
            start = int(torch.randint(0, max(T - W + 1, 1), (1,)).item())
            x_1, audio_win, prev_win = self._extract_window(expression, audio, prev_expr, start)

            loss = self._fm_loss(x_1, audio_win, emotion, prev_win)

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
                            "fm/train_loss": loss.item(),
                            "fm/lr": self.optimizer.param_groups[0]["lr"],
                            "fm/window_start": start,
                        },
                        step=self.global_step,
                        run=self.wandb_run,
                    )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Val: FM loss on held-out sub-windows + best-of-K ref-clip eval."""
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

            # Deterministic validation offset (not a random choice — fair across epochs).
            start = 0
            x_1, audio_win, prev_win = self._extract_window(expression, audio, prev_expr, start)
            loss = self._fm_loss(x_1, audio_win, emotion, prev_win)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if is_main_process():
            log_metrics(
                {"fm/val_loss": avg_loss, "fm/epoch": epoch},
                step=self.global_step,
                run=self.wandb_run,
            )
            print(f"  Val FM Loss: {avg_loss:.4f}")

        if self.ref_clip is not None:
            K = max(self.config.fm.eval_n_samples, 1)
            best_l1 = float("inf")
            best_m = None
            for _ in range(K):
                pred_norm = fm_slide_inference(
                    self.generator,
                    self.ref_clip,
                    window_size=self.config.fm.window_size,
                    nfe=self.config.fm.nfe_inference,
                    context_past=self.config.data.context_past,
                    context_future=self.config.data.context_future,
                    prev_frames=self.config.data.prev_frames,
                    expr_dim=self.config.data.flame_expr_dim,
                    device=self.device,
                )
                m = compute_ref_metrics(pred_norm, self.ref_clip)
                if m["ref_raw_l1"] < best_l1:
                    best_l1 = m["ref_raw_l1"]
                    best_m = m

            if is_main_process():
                log_metrics(
                    {
                        "fm/val_ref_raw_l1_bestK": best_m["ref_raw_l1"],
                        "fm/val_ref_raw_l1_lag_bestK": best_m["ref_raw_l1_lag"],
                        "fm/val_ref_std_ratio_full_bestK": best_m["ref_std_ratio_full"],
                        "fm/val_ref_std_ratio_mouth_bestK": best_m["ref_std_ratio_mouth"],
                        "fm/val_ref_std_ratio_problem_bestK": best_m["ref_std_ratio_problem"],
                        "fm/eval_n_samples": K,
                    },
                    step=self.global_step,
                    run=self.wandb_run,
                )
                print(
                    f"  Val Ref Raw L1 best-of-{K} "
                    f"({Path(self.config.stage2.ref_clip_audio_path).stem}): "
                    f"{best_m['ref_raw_l1']:.4f} "
                    f"(lag-tol {best_m['ref_raw_l1_lag']:.4f}) | "
                    f"std ratio full={best_m['ref_std_ratio_full']:.3f} "
                    f"mouth={best_m['ref_std_ratio_mouth']:.3f} "
                    f"problem={best_m['ref_std_ratio_problem']:.3f}"
                )
            return best_m["ref_raw_l1"]

        return avg_loss

    def save(self, epoch: int, val_loss: float):
        if not is_main_process():
            return

        from dataclasses import asdict

        path = checkpoint_path(self.config.paths.checkpoint_dir, epoch, prefix="fm")
        save_checkpoint(
            path, epoch, val_loss, asdict(self.config),
            global_step=self.global_step,
            generator=(self.generator, self.optimizer),
        )
        print(f"  Checkpoint saved: {path}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            best_path = str(Path(self.config.paths.checkpoint_dir) / "fm_best.pt")
            save_checkpoint(
                best_path, epoch, val_loss, asdict(self.config),
                global_step=self.global_step,
                generator=(self.generator, self.optimizer),
            )
            print(f"  New best model saved: {best_path}")
        else:
            self.epochs_without_improvement += 1

    def should_stop(self, patience: int) -> bool:
        return patience > 0 and self.epochs_without_improvement >= patience
