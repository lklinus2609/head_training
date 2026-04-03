"""Stage 3 training loop: adversarial co-training of generator and discriminator."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    gradient_penalty,
    l1_reconstruction_loss,
)
from training.schedulers import get_lambda_adv
from utils.checkpoint import checkpoint_path, save_checkpoint
from utils.ddp import is_main_process
from utils.logging_utils import log_metrics


class Stage3Trainer:
    """Handles the Stage 3 adversarial co-training loop."""

    def __init__(
        self,
        config,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        train_loader,
        val_loader,
        device,
        wandb_run=None,
    ):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.gen_scaler = torch.cuda.amp.GradScaler()
        self.disc_scaler = torch.cuda.amp.GradScaler()

    def _make_disc_windows(
        self, expression, velocity, acceleration, audio, window_size: int
    ):
        """Extract K-frame windows from full sequences for discriminator input.

        Args:
            expression: [B, T, 100]
            velocity: [B, T, 100]
            acceleration: [B, T, 100]
            audio: [B, T_audio, C_audio] (needs alignment)
            window_size: K frames per window.

        Returns:
            windows: [B * num_windows, K, input_dim]
        """
        B, T, D = expression.shape

        # Trim audio to match expression length
        audio_trimmed = audio[:, :T, :]

        # Concatenate features
        full = torch.cat([expression, velocity, acceleration, audio_trimmed], dim=-1)  # [B, T, D_total]

        # Extract overlapping windows
        windows = []
        for start in range(0, T - window_size + 1, window_size):
            windows.append(full[:, start : start + window_size])

        if not windows:
            # Fallback: take whatever fits
            windows.append(full[:, :window_size])

        windows = torch.cat(windows, dim=0)  # [B * num_windows, K, D_total]
        return windows

    def _short_horizon_forward(self, expression, audio, emotion, prev_expr):
        """Batched short-horizon prediction: predict H frames at a time using GT context.

        Eliminates the teacher-forcing mismatch by using the same prediction mode
        as autoregressive inference (target_expression=None).

        Args:
            expression: Ground truth expression [B, T, D].
            audio: Audio features with context [B, C+T+F, audio_dim].
            emotion: Emotion labels [B].
            prev_expr: Previous expression frames [B, P, D].

        Returns:
            (predictions, targets): each [B, N*H, D] where N = T // H.
        """
        B, T, D = expression.shape
        C = self.config.data.context_past
        F = self.config.data.context_future
        P = self.config.data.prev_frames
        H = self.config.stage3.gen_horizon

        # Concatenate prev context and expression for easy window extraction
        # full_expr[:, t:t+P] gives the P GT frames before position t
        full_expr = torch.cat([prev_expr, expression], dim=1)  # [B, P+T, D]

        # Collect windows for all positions with stride H
        positions = list(range(0, T - H + 1, H))
        N = len(positions)
        audio_len = C + H + F

        # Extract prev_expression and audio windows for each position
        prev_list = []
        audio_list = []
        target_list = []
        for t in positions:
            prev_list.append(full_expr[:, t:t + P])           # [B, P, D]
            audio_list.append(audio[:, t:t + audio_len])       # [B, C+H+F, audio_dim]
            target_list.append(expression[:, t:t + H])         # [B, H, D]

        # Stack and reshape into a single large batch [B*N, ...]
        prev_windows = torch.stack(prev_list, dim=1).reshape(B * N, P, D)
        audio_windows = torch.stack(audio_list, dim=1).reshape(B * N, audio_len, -1)
        emotion_expanded = emotion.unsqueeze(1).expand(-1, N).reshape(B * N)
        targets = torch.stack(target_list, dim=1).reshape(B * N, H, D)

        # Forward pass: predict H frames per window (no teacher forcing)
        with torch.cuda.amp.autocast():
            preds = self.generator(
                audio_windows, emotion_expanded, prev_windows,
                target_expression=None, max_len=H,
            )  # [B*N, H, D]

        # Reshape back to [B, N*H, D]
        preds = preds.reshape(B, N * H, D)
        targets = targets.reshape(B, N * H, D)
        return preds, targets

    def _compute_disc_windows_from_generated(
        self, pred_expression, batch, window_size: int
    ):
        """Build discriminator windows using generated expressions."""
        from data.flame_utils import compute_velocity, compute_acceleration
        import numpy as np

        # Compute velocity and acceleration from generated expressions
        pred_np = pred_expression.detach().cpu().numpy()
        B, T, D = pred_np.shape
        vel = torch.zeros_like(pred_expression)
        acc = torch.zeros_like(pred_expression)

        # Simple finite differences in torch
        vel[:, 1:] = pred_expression[:, 1:] - pred_expression[:, :-1]
        vel[:, 0] = vel[:, 1]
        if T >= 3:
            acc[:, 1:-1] = pred_expression[:, 2:] - 2 * pred_expression[:, 1:-1] + pred_expression[:, :-2]
            acc[:, 0] = acc[:, 1]
            acc[:, -1] = acc[:, -2]

        audio = batch["audio"].to(self.device)
        return self._make_disc_windows(pred_expression, vel, acc, audio, window_size)

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch with alternating D/G updates.

        Returns:
            Dict of average metrics for the epoch.
        """
        self.generator.train()
        self.discriminator.train()

        totals = {"d_loss": 0, "g_loss": 0, "recon": 0, "adv": 0, "gp": 0}
        num_batches = 0
        K = self.config.data.disc_window

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())

        for batch in pbar:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            velocity = batch["velocity"].to(self.device)
            acceleration = batch["acceleration"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            # === Discriminator step ===
            for _ in range(self.config.stage3.disc_steps_per_gen):
                # Real windows from data
                real_windows = self._make_disc_windows(expression, velocity, acceleration, audio, K)

                # Generated expressions
                with torch.no_grad(), torch.cuda.amp.autocast():
                    pred_expr = self.generator(audio, emotion, prev_expr, target_expression=expression)
                fake_windows = self._compute_disc_windows_from_generated(pred_expr, batch, K)

                with torch.cuda.amp.autocast():
                    real_score = self.discriminator(real_windows)
                    fake_score = self.discriminator(fake_windows.detach())
                    d_loss = discriminator_loss(real_score, fake_score)

                # GP in float32 (numerically sensitive, already uses math kernel)
                gp = gradient_penalty(self.discriminator, real_windows.float(), fake_windows.detach().float())
                total_d_loss = d_loss + self.config.discriminator.gp_weight * gp

                self.disc_optimizer.zero_grad()
                self.disc_scaler.scale(total_d_loss).backward()
                self.disc_scaler.unscale_(self.disc_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.config.stage3.grad_clip
                )
                self.disc_scaler.step(self.disc_optimizer)
                self.disc_scaler.update()

            # === Generator step ===
            # Pass 1: Teacher forcing for adversarial loss (needs full sequences for disc windows)
            with torch.cuda.amp.autocast():
                pred_expr_tf = self.generator(audio, emotion, prev_expr, target_expression=expression)
                fake_windows = self._compute_disc_windows_from_generated(pred_expr_tf, batch, K)
                fake_score = self.discriminator(fake_windows)
                adv_loss = generator_adversarial_loss(fake_score)

            # Pass 2: Short-horizon for reconstruction loss (matches inference)
            pred_sh, target_sh = self._short_horizon_forward(expression, audio, emotion, prev_expr)
            recon_loss = l1_reconstruction_loss(pred_sh, target_sh)

            # Combined generator loss
            lambda_adv = get_lambda_adv(
                self.global_step,
                self.config.stage3.lambda_warmup_steps,
                self.config.stage3.lambda_adv_start,
                self.config.stage3.lambda_adv_end,
            )
            g_loss = recon_loss + lambda_adv * adv_loss

            self.gen_optimizer.zero_grad()
            self.gen_scaler.scale(g_loss).backward()
            self.gen_scaler.unscale_(self.gen_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.config.stage3.grad_clip
            )
            self.gen_scaler.step(self.gen_optimizer)
            self.gen_scaler.update()

            # Track metrics
            totals["d_loss"] += d_loss.item()
            totals["g_loss"] += g_loss.item()
            totals["recon"] += recon_loss.item()
            totals["adv"] += adv_loss.item()
            totals["gp"] += gp.item()
            num_batches += 1
            self.global_step += 1

            if is_main_process():
                pbar.set_postfix(
                    d=f"{d_loss.item():.3f}",
                    g=f"{g_loss.item():.3f}",
                    r=f"{recon_loss.item():.3f}",
                    lam=f"{lambda_adv:.4f}",
                )

                if self.global_step % 10 == 0:
                    d_real_acc = (real_score > 0.0).float().mean().item()
                    d_fake_acc = (fake_score < 0.0).float().mean().item()
                    log_metrics(
                        {
                            "stage3/d_loss": d_loss.item(),
                            "stage3/g_loss": g_loss.item(),
                            "stage3/recon_loss": recon_loss.item(),
                            "stage3/adv_loss": adv_loss.item(),
                            "stage3/gp": gp.item(),
                            "stage3/lambda_adv": lambda_adv,
                            "stage3/d_real_acc": d_real_acc,
                            "stage3/d_fake_acc": d_fake_acc,
                        },
                        step=self.global_step,
                        run=self.wandb_run,
                    )

                    # Warn about training collapse
                    if d_real_acc > 0.99 and d_fake_acc > 0.99:
                        print(
                            f"  WARNING: Discriminator accuracy ~100% at step {self.global_step}. "
                            "Generator may be collapsing."
                        )

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run validation."""
        self.generator.eval()
        total_recon = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            pred = self.generator(audio, emotion, prev_expr, target_expression=expression)
            recon = l1_reconstruction_loss(pred, expression)
            mse = F.mse_loss(pred, expression)

            total_recon += recon.item()
            total_mse += mse.item()
            num_batches += 1

        avg_recon = total_recon / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)

        if is_main_process():
            log_metrics(
                {
                    "stage3/val_recon": avg_recon,
                    "stage3/val_mse": avg_mse,
                    "stage3/epoch": epoch,
                },
                step=self.global_step,
                run=self.wandb_run,
            )
            print(f"  Val L1: {avg_recon:.4f}, Val MSE: {avg_mse:.4f}")

        return avg_recon

    def save(self, epoch: int, val_loss: float):
        """Save checkpoint (rank 0 only)."""
        if not is_main_process():
            return

        from dataclasses import asdict

        path = checkpoint_path(self.config.paths.checkpoint_dir, epoch, prefix="stage3")
        save_checkpoint(
            path, epoch, val_loss, asdict(self.config),
            generator=(self.generator, self.gen_optimizer),
            discriminator=(self.discriminator, self.disc_optimizer),
        )
        print(f"  Checkpoint saved: {path}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = str(
                __import__("pathlib").Path(self.config.paths.checkpoint_dir) / "stage3_best.pt"
            )
            save_checkpoint(
                best_path, epoch, val_loss, asdict(self.config),
                generator=(self.generator, self.gen_optimizer),
                discriminator=(self.discriminator, self.disc_optimizer),
            )
            print(f"  New best model saved: {best_path}")
