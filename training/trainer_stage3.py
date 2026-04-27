"""Stage 3 training loop: adversarial co-training of generator and discriminator."""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.ref_clip import (
    ar_slide_inference,
    compute_ref_metrics,
    prepare_ref_clip,
)
from training.losses import (
    acceleration_loss,
    covariance_loss,
    discriminator_loss,
    generator_adversarial_loss,
    gradient_penalty,
    l1_reconstruction_loss,
    spectral_loss,
    velocity_loss,
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
        dim_weights=None,
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
        self.dim_weights = dim_weights
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.gen_scaler = torch.cuda.amp.GradScaler()
        self.disc_scaler = torch.cuda.amp.GradScaler()

        # Reference-clip evaluation for apples-to-apples A/B comparison across
        # Stage 2, Stage 3, and the FM track. Uses the same clip as Stage 2.
        self.ref_clip = None
        if is_main_process() and config.stage2.ref_clip_audio_path:
            self.ref_clip = prepare_ref_clip(
                config, self.train_loader.dataset.stats, device
            )

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

    def _short_horizon_forward(self, expression, audio, emotion, prev_expr, p_drift: float = 0.0):
        """Batched short-horizon prediction: predict H frames at a time using GT context.

        Eliminates the teacher-forcing mismatch by using the same prediction mode
        as autoregressive inference (target_expression=None).

        When p_drift > 0, for each window beyond the first, each sample in the
        batch independently decides (with probability p_drift) to replace its
        GT prev-frames with the generator's own detached output from the prior
        window. This practises self-drift recovery at training time, closing
        the train/inference gap that drives motion-range damping.

        Returns:
            (predictions, targets): each [B, N*H, D] where N = T // H.
        """
        B, T, D = expression.shape
        C = self.config.data.context_past
        F = self.config.data.context_future
        P = self.config.data.prev_frames
        H = self.config.stage3.gen_horizon

        full_expr = torch.cat([prev_expr, expression], dim=1)  # [B, P+T, D]
        positions = list(range(0, T - H + 1, H))
        N = len(positions)
        audio_len = C + H + F

        if p_drift <= 0.0:
            prev_list, audio_list, target_list = [], [], []
            for t in positions:
                prev_list.append(full_expr[:, t:t + P])
                audio_list.append(audio[:, t:t + audio_len])
                target_list.append(expression[:, t:t + H])

            prev_windows = torch.stack(prev_list, dim=1).reshape(B * N, P, D)
            audio_windows = torch.stack(audio_list, dim=1).reshape(B * N, audio_len, -1)
            emotion_expanded = emotion.unsqueeze(1).expand(-1, N).reshape(B * N)
            targets = torch.stack(target_list, dim=1).reshape(B * N, H, D)

            with torch.cuda.amp.autocast():
                preds = self.generator(
                    audio_windows, emotion_expanded, prev_windows,
                    target_expression=None, max_len=H,
                )  # [B*N, H, D]

            preds = preds.reshape(B, N * H, D)
            targets = targets.reshape(B, N * H, D)
            return preds, targets

        # Self-drift path: sequential across windows; detached generator-own
        # prev mixed in per-sample. Gradients flow only through each window.
        running_prev = full_expr[:, 0:P]
        preds_all = []
        targets_all = []
        for j, t in enumerate(positions):
            gt_prev = full_expr[:, t:t + P]
            if j == 0:
                prev_win = gt_prev
            else:
                drift_mask = torch.rand(B, device=expression.device) < p_drift
                prev_win = torch.where(
                    drift_mask[:, None, None], running_prev.detach(), gt_prev
                )
            audio_win = audio[:, t:t + audio_len]
            with torch.cuda.amp.autocast():
                pred = self.generator(
                    audio_win, emotion, prev_win,
                    target_expression=None, max_len=H,
                )  # [B, H, D]
            preds_all.append(pred)
            targets_all.append(expression[:, t:t + H])
            running_prev = pred[:, -P:]

        preds = torch.cat(preds_all, dim=1)
        targets = torch.cat(targets_all, dim=1)
        return preds, targets

    def _effective_p_drift(self, epoch: int) -> float:
        start = self.config.stage3.p_drift_start
        end = self.config.stage3.p_drift_end
        w = self.config.stage3.p_drift_warmup_epochs
        if w <= 0 or epoch >= w:
            return end
        return start + (end - start) * (epoch / w)

    def _compute_disc_windows_from_generated(
        self, pred_expression, batch, window_size: int
    ):
        """Build discriminator windows using generated expressions."""
        B, T, D = pred_expression.shape
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

        totals = {
            "d_loss": 0, "g_loss": 0, "recon": 0, "adv": 0, "gp": 0,
            "vel": 0, "accel": 0, "spec": 0, "cov": 0,
        }
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

            # === Generator forward (once, with gradients) ===
            p_drift = self._effective_p_drift(epoch)
            pred_sh, target_sh = self._short_horizon_forward(
                expression, audio, emotion, prev_expr, p_drift=p_drift,
            )
            pred_sh_reshaped = pred_sh.reshape(expression.shape[0], -1, expression.shape[-1])

            # === Discriminator step ===
            for _ in range(self.config.stage3.disc_steps_per_gen):
                real_windows = self._make_disc_windows(expression, velocity, acceleration, audio, K)
                fake_windows = self._compute_disc_windows_from_generated(pred_sh_reshaped.detach(), batch, K)

                with torch.cuda.amp.autocast():
                    real_score = self.discriminator(real_windows)
                    fake_score = self.discriminator(fake_windows)
                    d_loss = discriminator_loss(real_score, fake_score)

                gp = gradient_penalty(self.discriminator, real_windows.float(), fake_windows.float())
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
            recon_loss = l1_reconstruction_loss(pred_sh, target_sh, self.dim_weights)

            # Auxiliary motion-range losses carried forward from Stage 2.
            # Zero λ = original stage3 behaviour (pure L1 anchor). Positive λ
            # keeps damping counter-pressure alive while adversarial pushes
            # off the L1 median.
            lambda_vel = self.config.stage3.lambda_vel
            if lambda_vel > 0.0:
                vel_loss = velocity_loss(pred_sh, target_sh, self.dim_weights)
            else:
                vel_loss = torch.zeros((), device=pred_sh.device)
            lambda_accel = self.config.stage3.lambda_accel
            if lambda_accel > 0.0:
                accel_loss = acceleration_loss(pred_sh, target_sh, self.dim_weights)
            else:
                accel_loss = torch.zeros((), device=pred_sh.device)
            lambda_spec = self.config.stage3.lambda_spec
            if lambda_spec > 0.0:
                spec_loss = spectral_loss(pred_sh, target_sh, self.dim_weights)
            else:
                spec_loss = torch.zeros((), device=pred_sh.device)
            lambda_cov = self.config.stage3.lambda_cov
            if lambda_cov > 0.0:
                cov_loss = covariance_loss(pred_sh, target_sh)
            else:
                cov_loss = torch.zeros((), device=pred_sh.device)

            fake_windows_g = self._compute_disc_windows_from_generated(pred_sh_reshaped, batch, K)
            with torch.cuda.amp.autocast():
                fake_score = self.discriminator(fake_windows_g)
                adv_loss = generator_adversarial_loss(fake_score)

            # Combined generator loss
            lambda_adv = get_lambda_adv(
                self.global_step,
                self.config.stage3.lambda_warmup_steps,
                self.config.stage3.lambda_adv_start,
                self.config.stage3.lambda_adv_end,
            )
            g_loss = (
                recon_loss
                + lambda_vel * vel_loss
                + lambda_accel * accel_loss
                + lambda_spec * spec_loss
                + lambda_cov * cov_loss
                + lambda_adv * adv_loss
            )

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
            totals["vel"] += vel_loss.item()
            totals["accel"] += accel_loss.item()
            totals["spec"] += spec_loss.item()
            totals["cov"] += cov_loss.item()
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
                            "stage3/vel_loss": vel_loss.item(),
                            "stage3/accel_loss": accel_loss.item(),
                            "stage3/spec_loss": spec_loss.item(),
                            "stage3/cov_loss": cov_loss.item(),
                            "stage3/lambda_adv": lambda_adv,
                            "stage3/lambda_vel": lambda_vel,
                            "stage3/lambda_accel": lambda_accel,
                            "stage3/lambda_spec": lambda_spec,
                            "stage3/lambda_cov": lambda_cov,
                            "stage3/d_real_acc": d_real_acc,
                            "stage3/d_fake_acc": d_fake_acc,
                            "stage3/p_drift_eff": p_drift,
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
        """Run validation using short-horizon forward (matches training)."""
        self.generator.eval()
        total_recon = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            pred, target = self._short_horizon_forward(expression, audio, emotion, prev_expr)
            recon = l1_reconstruction_loss(pred, target, self.dim_weights)
            mse = F.mse_loss(pred, target)

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

        if self.ref_clip is not None:
            pred_norm = ar_slide_inference(
                self.generator,
                self.ref_clip,
                horizon=self.config.stage3.gen_horizon,
                context_past=self.config.data.context_past,
                context_future=self.config.data.context_future,
                prev_frames=self.config.data.prev_frames,
                expr_dim=self.config.data.flame_expr_dim,
                device=self.device,
            )
            m = compute_ref_metrics(pred_norm, self.ref_clip)
            if is_main_process():
                log_metrics(
                    {
                        "stage3/val_ref_raw_l1": m["ref_raw_l1"],
                        "stage3/val_ref_raw_l1_lag": m["ref_raw_l1_lag"],
                        "stage3/val_ref_std_ratio_full": m["ref_std_ratio_full"],
                        "stage3/val_ref_std_ratio_mouth": m["ref_std_ratio_mouth"],
                        "stage3/val_ref_std_ratio_problem": m["ref_std_ratio_problem"],
                    },
                    step=self.global_step,
                    run=self.wandb_run,
                )
                print(
                    f"  Val Ref Raw L1 "
                    f"({Path(self.config.stage2.ref_clip_audio_path).stem}): "
                    f"{m['ref_raw_l1']:.4f} "
                    f"(lag-tol {m['ref_raw_l1_lag']:.4f}) | "
                    f"std ratio full={m['ref_std_ratio_full']:.3f} "
                    f"mouth={m['ref_std_ratio_mouth']:.3f} "
                    f"problem={m['ref_std_ratio_problem']:.3f}"
                )
            metric = self.config.stage3.selection_metric
            if metric == "ref_raw_l1_lag":
                return m["ref_raw_l1_lag"]
            if metric == "composite":
                # L1_lag stays the anchor; std-ratio penalty rewards motion
                # range climbing back toward 1.0. 0.5 weight balances the
                # typical L1 scale (~0.3) against ratio distances (~0.1–0.2).
                return m["ref_raw_l1_lag"] + 0.5 * abs(m["ref_std_ratio_problem"] - 1.0)
            return m["ref_raw_l1"]

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
