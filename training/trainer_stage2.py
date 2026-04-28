"""Stage 2 training loop: generator pretraining with L1 reconstruction loss."""

from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.ref_clip import (
    aggregate_ref_metrics,
    ar_slide_inference,
    compute_ref_metrics,
    prepare_ref_clips,
)
from training.losses import (
    acceleration_loss,
    covariance_loss,
    l1_reconstruction_loss,
    spectral_loss,
    variance_matching_loss,
    velocity_loss,
)
from utils.checkpoint import checkpoint_path, save_checkpoint
from utils.ddp import is_main_process, reduce_mean
from utils.logging_utils import log_metrics


class Stage2Trainer:
    """Handles the Stage 2 generator pretraining loop."""

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

        self.use_bf16 = getattr(config.stage2, "use_bf16", False) and device.type == "cuda"
        if self.use_bf16 and is_main_process():
            print("Using bfloat16 autocast for forward + loss (no GradScaler needed)")

        # Reference-clip evaluation for best-checkpoint selection.
        # Multi-clip mode (config.stage2.ref_clip_audio_paths non-empty)
        # averages metrics across clips and uses lag-tolerant L1 for
        # selection. Legacy single-clip mode (only ref_clip_audio_path set)
        # keeps using raw L1 for selection so old runs stay comparable.
        # Loaded once on rank 0 here; _run_ref_inference reuses the tensors.
        self.ref_clips: list[dict] | None = None
        self.multi_clip_selection: bool = False
        if is_main_process() and (
            config.stage2.ref_clip_audio_paths or config.stage2.ref_clip_audio_path
        ):
            self.ref_clips = prepare_ref_clips(
                config, self.train_loader.dataset.stats, device
            )
            self.multi_clip_selection = bool(config.stage2.ref_clip_audio_paths)

    def _short_horizon_forward(self, expression, audio, emotion, prev_expr, p_drift: float = 0.0):
        """Batched short-horizon prediction: predict H frames at a time using GT context.

        Eliminates the teacher-forcing mismatch by using the same prediction mode
        as autoregressive inference (target_expression=None).

        When p_drift > 0, for each window beyond the first, each sample in the
        batch independently decides (with probability p_drift) to replace its
        GT prev-frames with the generator's own detached output from the prior
        window. This practises self-drift recovery at training time, closing
        the train/inference gap that drives motion-range damping.

        Args:
            expression: Ground truth expression [B, T, D].
            audio: Audio features with context [B, C+T+F, audio_dim].
            emotion: Emotion labels [B].
            prev_expr: Previous expression frames [B, P, D].
            p_drift: Probability of using the generator's own prev-frames
                instead of GT for windows beyond the first. Default 0 → fast
                batched path.

        Returns:
            (predictions, targets): each [B, N*H, D] where N = T // H.
        """
        B, T, D = expression.shape
        C = self.config.data.context_past
        F = self.config.data.context_future
        P = self.config.data.prev_frames
        H = self.config.stage2.gen_horizon

        full_expr = torch.cat([prev_expr, expression], dim=1)  # [B, P+T, D]
        positions = list(range(0, T - H + 1, H))
        N = len(positions)
        audio_len = C + H + F

        if p_drift <= 0.0:
            # Fast batched path: stack all windows and run one forward.
            prev_list, audio_list, target_list = [], [], []
            for t in positions:
                prev_list.append(full_expr[:, t:t + P])
                audio_list.append(audio[:, t:t + audio_len])
                target_list.append(expression[:, t:t + H])

            prev_windows = torch.stack(prev_list, dim=1).reshape(B * N, P, D)
            if self.generator.training and self.config.stage2.prev_frames_noise_std > 0:
                prev_windows = prev_windows + torch.randn_like(prev_windows) * self.config.stage2.prev_frames_noise_std
            audio_windows = torch.stack(audio_list, dim=1).reshape(B * N, audio_len, -1)
            emotion_expanded = emotion.unsqueeze(1).expand(-1, N).reshape(B * N)
            targets = torch.stack(target_list, dim=1).reshape(B * N, H, D)

            preds = self.generator(
                audio_windows, emotion_expanded, prev_windows,
                target_expression=None, max_len=H,
            )  # [B*N, H, D]

            preds = preds.reshape(B, N * H, D)
            targets = targets.reshape(B, N * H, D)
            return preds, targets

        # Self-drift path: sequential across windows. Detach the drifted prev
        # so we don't BPTT across the whole clip — gradients only flow from
        # each window's own L1 back through its own forward.
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
            if self.generator.training and self.config.stage2.prev_frames_noise_std > 0:
                noise = torch.randn_like(prev_win) * self.config.stage2.prev_frames_noise_std
                if j > 0:
                    noise = torch.where(drift_mask[:, None, None], torch.zeros_like(noise), noise)
                prev_win = prev_win + noise
            audio_win = audio[:, t:t + audio_len]
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
        start = self.config.stage2.p_drift_start
        end = self.config.stage2.p_drift_end
        w = self.config.stage2.p_drift_warmup_epochs
        if w <= 0 or epoch >= w:
            return end
        return start + (end - start) * (epoch / w)

    @torch.no_grad()
    def _run_ref_inference(self) -> tuple[dict, list[tuple[str, dict]]]:
        """Sliding-mode AR inference across all configured ref clips.

        Returns:
            (aggregated_metrics, per_clip): aggregated_metrics is the
            mean of each scalar metric across clips. per_clip is a list of
            (label, metrics) for diagnostic logging.
        """
        per_clip: list[tuple[str, dict]] = []
        for clip in self.ref_clips:
            pred_norm = ar_slide_inference(
                self.generator,
                clip,
                horizon=self.config.stage2.gen_horizon,
                context_past=self.config.data.context_past,
                context_future=self.config.data.context_future,
                prev_frames=self.config.data.prev_frames,
                expr_dim=self.config.data.flame_expr_dim,
                device=self.device,
            )
            metrics = compute_ref_metrics(pred_norm, clip)
            per_clip.append((clip["label"], metrics))

        aggregated = aggregate_ref_metrics([m for _, m in per_clip])
        return aggregated, per_clip

    def _effective_lambda_var(self, epoch: int) -> float:
        lv = self.config.stage2.lambda_var
        w = self.config.stage2.lambda_var_warmup_epochs
        d = self.config.stage2.lambda_var_decay_epochs
        if lv == 0.0 or epoch < w:
            return lv
        if d > 0 and epoch < w + d:
            return lv * (1.0 - (epoch - w) / d)
        return 0.0

    def _effective_lambda_var_full(self, epoch: int) -> float:
        """Late-stage full-rollout variance matching weight.

        Zero until `lambda_var_full_start`, then `lambda_var_full` until the
        last `lambda_var_full_decay_epochs` of training, which linearly fade
        the weight back to zero. Motivation (commit e549ef7): avoid a strong
        late-epoch regularization gradient that can kick the optimizer out
        of a good basin.
        """
        lvf = self.config.stage2.lambda_var_full
        if lvf <= 0.0:
            return 0.0
        start = self.config.stage2.lambda_var_full_start
        if epoch < start:
            return 0.0
        decay = self.config.stage2.lambda_var_full_decay_epochs
        total = self.config.stage2.epochs
        decay_start = total - decay if decay > 0 else total
        if epoch < decay_start:
            return lvf
        if decay > 0 and epoch < total:
            return lvf * (1.0 - (epoch - decay_start) / decay)
        return 0.0

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.generator.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())

        for batch in pbar:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            # Short-horizon forward (matches autoregressive inference)
            p_drift = self._effective_p_drift(epoch)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.use_bf16 else nullcontext()
            )
            with autocast_ctx:
                pred, target = self._short_horizon_forward(
                    expression, audio, emotion, prev_expr, p_drift=p_drift,
                )
                l1 = l1_reconstruction_loss(pred, target, self.dim_weights)
                lambda_var = self._effective_lambda_var(epoch)
                if lambda_var > 0.0:
                    var_loss = variance_matching_loss(pred, target)
                    loss = l1 + lambda_var * var_loss
                else:
                    var_loss = torch.zeros((), device=pred.device)
                    loss = l1

                # Velocity / acceleration L1. Fully batched — no sequential cost.
                # Velocity -> directly penalises motion-range damping.
                # Acceleration -> directly penalises frame-to-frame jitter.
                lambda_vel = self.config.stage2.lambda_vel
                if lambda_vel > 0.0:
                    vel_loss = velocity_loss(pred, target, self.dim_weights)
                    loss = loss + lambda_vel * vel_loss
                else:
                    vel_loss = torch.zeros((), device=pred.device)

                lambda_accel = self.config.stage2.lambda_accel
                if lambda_accel > 0.0:
                    accel_loss = acceleration_loss(pred, target, self.dim_weights)
                    loss = loss + lambda_accel * accel_loss
                else:
                    accel_loss = torch.zeros((), device=pred.device)

                # Spectral magnitude L1: pressures high-frequency motion content
                # that time-domain L1 under-weights. Attacks the "averaging out"
                # failure mode directly without changing the inference path.
                lambda_spec = self.config.stage2.lambda_spec
                if lambda_spec > 0.0:
                    spec_loss = spectral_loss(pred, target, self.dim_weights)
                    loss = loss + lambda_spec * spec_loss
                else:
                    spec_loss = torch.zeros((), device=pred.device)

                # Cross-dim covariance L1: enforces coordinated multi-dim
                # motion (jaw-lip, etc.) that the diagonal variance match
                # leaves unconstrained.
                lambda_cov = self.config.stage2.lambda_cov
                if lambda_cov > 0.0:
                    cov_loss = covariance_loss(pred, target)
                    loss = loss + lambda_cov * cov_loss
                else:
                    cov_loss = torch.zeros((), device=pred.device)

                # Optional late-stage full-rollout variance matching. Sequential
                # (p_drift=1.0 path) so only enable when velocity/accel losses
                # aren't enough to keep per-dim spread calibrated.
                lambda_var_full = self._effective_lambda_var_full(epoch)
                if lambda_var_full > 0.0:
                    full_pred, full_target = self._short_horizon_forward(
                        expression, audio, emotion, prev_expr, p_drift=1.0,
                    )
                    var_full_loss = variance_matching_loss(full_pred, full_target)
                    loss = loss + lambda_var_full * var_full_loss
                else:
                    var_full_loss = torch.zeros((), device=pred.device)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.config.stage2.grad_clip
            )
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if is_main_process():
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l1=f"{l1.item():.4f}",
                    var=f"{var_loss.item():.4f}",
                )
                if self.global_step % 50 == 0:
                    log_metrics(
                        {
                            "stage2/train_loss": loss.item(),
                            "stage2/train_l1": l1.item(),
                            "stage2/train_var": var_loss.item(),
                            "stage2/train_vel": vel_loss.item(),
                            "stage2/train_accel": accel_loss.item(),
                            "stage2/train_spec": spec_loss.item(),
                            "stage2/train_cov": cov_loss.item(),
                            "stage2/train_var_full": var_full_loss.item(),
                            "stage2/lambda_var_eff": lambda_var,
                            "stage2/lambda_vel_eff": lambda_vel,
                            "stage2/lambda_accel_eff": lambda_accel,
                            "stage2/lambda_spec_eff": lambda_spec,
                            "stage2/lambda_cov_eff": lambda_cov,
                            "stage2/lambda_var_full_eff": lambda_var_full,
                            "stage2/p_drift_eff": p_drift,
                            "stage2/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.global_step,
                        run=self.wandb_run,
                    )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run validation; returns the best-checkpoint selection metric.

        When `stage2.ref_clip_audio_path` is set, returns raw-space L1 from
        sliding inference on that clip (aligned with inference quality).
        Otherwise falls back to average weighted val L1 — prior behavior.
        """
        self.generator.eval()
        total_loss = 0.0
        total_var = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            pred, target = self._short_horizon_forward(expression, audio, emotion, prev_expr)
            loss = l1_reconstruction_loss(pred, target, self.dim_weights)
            var_loss = variance_matching_loss(pred, target)
            mse = torch.nn.functional.mse_loss(pred, target)

            total_loss += loss.item()
            total_var += var_loss.item()
            total_mse += mse.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_var = total_var / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)

        if is_main_process():
            log_metrics(
                {
                    "stage2/val_loss": avg_loss,
                    "stage2/val_var": avg_var,
                    "stage2/val_mse": avg_mse,
                    "stage2/epoch": epoch,
                },
                step=self.global_step,
                run=self.wandb_run,
            )
            print(f"  Val L1: {avg_loss:.4f}, Val Var: {avg_var:.4f}, Val MSE: {avg_mse:.4f}")

        # Reference-clip eval — selection metric when configured.
        if self.ref_clips is not None:
            m, per_clip = self._run_ref_inference()
            if is_main_process():
                # Aggregated metrics (averaged across clips, or single-clip values).
                log_metrics(
                    {
                        "stage2/val_ref_raw_l1": m["ref_raw_l1"],
                        "stage2/val_ref_raw_l1_lag": m["ref_raw_l1_lag"],
                        "stage2/val_ref_std_ratio_full": m["ref_std_ratio_full"],
                        "stage2/val_ref_std_ratio_mouth": m["ref_std_ratio_mouth"],
                        "stage2/val_ref_std_ratio_problem": m["ref_std_ratio_problem"],
                    },
                    step=self.global_step,
                    run=self.wandb_run,
                )
                # Per-clip diagnostic metrics so we can see which clips
                # the model handles well vs. poorly.
                if len(per_clip) > 1:
                    per_clip_log = {}
                    for label, pm in per_clip:
                        per_clip_log[f"stage2/ref/{label}/raw_l1"] = pm["ref_raw_l1"]
                        per_clip_log[f"stage2/ref/{label}/raw_l1_lag"] = pm["ref_raw_l1_lag"]
                        per_clip_log[f"stage2/ref/{label}/std_ratio_full"] = pm["ref_std_ratio_full"]
                    log_metrics(per_clip_log, step=self.global_step, run=self.wandb_run)

                if len(per_clip) == 1:
                    label = per_clip[0][0]
                    print(
                        f"  Val Ref ({label}): "
                        f"L1={m['ref_raw_l1']:.4f} "
                        f"(lag-tol {m['ref_raw_l1_lag']:.4f}) | "
                        f"std ratio full={m['ref_std_ratio_full']:.3f} "
                        f"mouth={m['ref_std_ratio_mouth']:.3f} "
                        f"problem={m['ref_std_ratio_problem']:.3f}"
                    )
                else:
                    print(
                        f"  Val Ref (avg over {len(per_clip)} clips): "
                        f"L1={m['ref_raw_l1']:.4f} "
                        f"(lag-tol {m['ref_raw_l1_lag']:.4f}) | "
                        f"std ratio full={m['ref_std_ratio_full']:.3f} "
                        f"mouth={m['ref_std_ratio_mouth']:.3f} "
                        f"problem={m['ref_std_ratio_problem']:.3f}"
                    )
            # Multi-clip mode → lag-tolerant selection (more robust to
            # per-clip timing offsets). Legacy single-clip mode → raw L1
            # so existing run histories stay directly comparable.
            return m["ref_raw_l1_lag"] if self.multi_clip_selection else m["ref_raw_l1"]

        return avg_loss

    def save(self, epoch: int, val_loss: float):
        """Save checkpoint (rank 0 only)."""
        if not is_main_process():
            return

        from dataclasses import asdict

        path = checkpoint_path(self.config.paths.checkpoint_dir, epoch, prefix="stage2")
        save_checkpoint(
            path, epoch, val_loss, asdict(self.config),
            global_step=self.global_step,
            generator=(self.generator, self.optimizer),
        )
        print(f"  Checkpoint saved: {path}")

        # Save best model separately
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            best_path = str(
                __import__("pathlib").Path(self.config.paths.checkpoint_dir) / "stage2_best.pt"
            )
            save_checkpoint(
                best_path, epoch, val_loss, asdict(self.config),
                global_step=self.global_step,
                generator=(self.generator, self.optimizer),
            )
            print(f"  New best model saved: {best_path}")
        else:
            self.epochs_without_improvement += 1

    def should_stop(self, patience: int) -> bool:
        """Check if training should stop due to no improvement."""
        return patience > 0 and self.epochs_without_improvement >= patience
