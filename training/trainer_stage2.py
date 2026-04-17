"""Stage 2 training loop: generator pretraining with L1 reconstruction loss."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from data.audio_features import extract_mel
from training.losses import l1_reconstruction_loss, variance_matching_loss
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

        # Reference-clip raw-L1 evaluation for best-checkpoint selection.
        # Loaded once on rank 0 here; _run_ref_inference reuses the tensors each val.
        self.ref_clip = None
        if is_main_process() and config.stage2.ref_clip_audio_path:
            self.ref_clip = self._prepare_ref_clip(config.stage2.ref_clip_audio_path)

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
        H = self.config.stage2.gen_horizon

        # Concatenate prev context and expression for easy window extraction
        full_expr = torch.cat([prev_expr, expression], dim=1)  # [B, P+T, D]

        # Collect windows for all positions with stride H
        positions = list(range(0, T - H + 1, H))
        N = len(positions)
        audio_len = C + H + F

        prev_list = []
        audio_list = []
        target_list = []
        for t in positions:
            prev_list.append(full_expr[:, t:t + P])
            audio_list.append(audio[:, t:t + audio_len])
            target_list.append(expression[:, t:t + H])

        # Stack and reshape into a single large batch [B*N, ...]
        prev_windows = torch.stack(prev_list, dim=1).reshape(B * N, P, D)
        audio_windows = torch.stack(audio_list, dim=1).reshape(B * N, audio_len, -1)
        emotion_expanded = emotion.unsqueeze(1).expand(-1, N).reshape(B * N)
        targets = torch.stack(target_list, dim=1).reshape(B * N, H, D)

        # Forward pass: predict H frames per window (no teacher forcing)
        preds = self.generator(
            audio_windows, emotion_expanded, prev_windows,
            target_expression=None, max_len=H,
        )  # [B*N, H, D]

        # Reshape back to [B, N*H, D]
        preds = preds.reshape(B, N * H, D)
        targets = targets.reshape(B, N * H, D)
        return preds, targets

    def _prepare_ref_clip(self, audio_path: str) -> dict | None:
        """Pre-load the reference clip for per-val-epoch raw-L1 evaluation.

        Mirrors the setup portion of viewer/generate_sequence.py:generate_from_model.
        Returns None if the audio file or matching BEAT2 npz is missing (logs a warning).
        Only called on rank 0.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"  [ref_clip] audio not found: {audio_path} — skipping ref eval")
            return None

        if self.config.data.audio_feature != "mel":
            print(f"  [ref_clip] only mel audio_feature supported; got "
                  f"{self.config.data.audio_feature} — skipping ref eval")
            return None

        waveform, sr = sf.read(str(audio_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)

        audio_feats = extract_mel(
            waveform, sr, self.config.data.fps,
            n_mels=self.config.data.mel_n_mels,
            win_ms=self.config.data.mel_win_ms,
            hop_ms=self.config.data.mel_hop_ms,
        )

        C = self.config.data.context_past
        F = self.config.data.context_future
        audio_padded = np.pad(audio_feats, ((C, F), (0, 0)), mode="edge")
        audio_tensor = torch.from_numpy(audio_padded).float().unsqueeze(0).to(self.device)

        # Emotion from filename: {speaker}_{name}_{emotion}_{seq}_{take}
        parts = audio_path.stem.split("_")
        try:
            emotion = int(parts[2])
        except (IndexError, ValueError):
            emotion = 0
        emotion_tensor = torch.tensor([emotion], device=self.device)

        # Find matching BEAT2 npz for ground truth (raw FLAME space, 30 fps)
        beat2_dir = Path(self.config.paths.beat2_raw_dir)
        npz_candidates = list(beat2_dir.rglob(f"{audio_path.stem}.npz"))
        if not npz_candidates:
            print(f"  [ref_clip] no matching npz for {audio_path.stem} under "
                  f"{beat2_dir} — skipping ref eval")
            return None
        npz_data = np.load(str(npz_candidates[0]), allow_pickle=True)
        if "expressions" not in npz_data:
            print(f"  [ref_clip] npz has no 'expressions' field — skipping ref eval")
            return None
        gt_raw = npz_data["expressions"][:, :self.config.data.flame_expr_dim].astype(np.float32)

        # Pull normalization stats from the train dataset (same ones dim_weights uses)
        ds_stats = self.train_loader.dataset.stats
        if "expr_mean" not in ds_stats or "expr_std" not in ds_stats:
            print(f"  [ref_clip] train dataset missing expr_mean/expr_std — skipping ref eval")
            return None
        expr_mean = torch.from_numpy(ds_stats["expr_mean"]).float().to(self.device)
        expr_std = torch.from_numpy(ds_stats["expr_std"]).float().to(self.device)
        expr_std = torch.where(expr_std < 1e-8, torch.ones_like(expr_std), expr_std)

        T = audio_feats.shape[0]
        print(f"  [ref_clip] loaded {audio_path.name}: T={T} frames, emotion={emotion}, "
              f"gt_frames={gt_raw.shape[0]}")

        return {
            "audio": audio_tensor,
            "emotion": emotion_tensor,
            "gt_raw": gt_raw,
            "expr_mean": expr_mean,
            "expr_std": expr_std,
            "T": T,
        }

    @torch.no_grad()
    def _run_ref_inference(self) -> float:
        """Sliding-mode inference on the reference clip; returns raw-space L1."""
        C = self.config.data.context_past
        F = self.config.data.context_future
        P = self.config.data.prev_frames
        H = self.config.stage2.gen_horizon
        D = self.config.data.flame_expr_dim
        T = self.ref_clip["T"]
        audio = self.ref_clip["audio"]
        emotion = self.ref_clip["emotion"]

        gen = self.generator.module if hasattr(self.generator, "module") else self.generator

        prev_expr = torch.zeros(1, P, D, device=self.device)
        chunks = []
        for t in range(0, T, H):
            chunk_len = min(H, T - t)
            audio_chunk = audio[:, t:t + C + chunk_len + F]
            pred = gen(audio_chunk, emotion, prev_expr,
                       target_expression=None, max_len=chunk_len)
            chunks.append(pred)
            prev_expr = torch.cat([prev_expr, pred], dim=1)[:, -P:]

        pred_norm = torch.cat(chunks, dim=1)[0]  # [T, D]
        pred_raw = (pred_norm * self.ref_clip["expr_std"] + self.ref_clip["expr_mean"]).cpu().numpy()

        gt_raw = self.ref_clip["gt_raw"]
        n = min(pred_raw.shape[0], gt_raw.shape[0])
        return float(np.abs(pred_raw[:n] - gt_raw[:n]).mean())

    def _effective_lambda_var(self, epoch: int) -> float:
        lv = self.config.stage2.lambda_var
        w = self.config.stage2.lambda_var_warmup_epochs
        d = self.config.stage2.lambda_var_decay_epochs
        if lv == 0.0 or epoch < w:
            return lv
        if d > 0 and epoch < w + d:
            return lv * (1.0 - (epoch - w) / d)
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
            pred, target = self._short_horizon_forward(expression, audio, emotion, prev_expr)
            l1 = l1_reconstruction_loss(pred, target, self.dim_weights)
            lambda_var = self._effective_lambda_var(epoch)
            if lambda_var > 0.0:
                var_loss = variance_matching_loss(pred, target)
                loss = l1 + lambda_var * var_loss
            else:
                var_loss = torch.zeros((), device=pred.device)
                loss = l1

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
                            "stage2/lambda_var_eff": lambda_var,
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

        # Raw-L1 on reference clip — selection metric when configured.
        if self.ref_clip is not None:
            ref_l1 = self._run_ref_inference()
            if is_main_process():
                log_metrics(
                    {"stage2/val_ref_raw_l1": ref_l1},
                    step=self.global_step,
                    run=self.wandb_run,
                )
                print(f"  Val Ref Raw L1 "
                      f"({Path(self.config.stage2.ref_clip_audio_path).stem}): {ref_l1:.4f}")
            return ref_l1

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
