"""Stage 2 training loop: generator pretraining with L1 reconstruction loss."""

import torch
from tqdm import tqdm

from training.losses import l1_reconstruction_loss
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
            loss = l1_reconstruction_loss(pred, target, self.dim_weights)

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
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if self.global_step % 50 == 0:
                    log_metrics(
                        {"stage2/train_loss": loss.item(), "stage2/lr": self.optimizer.param_groups[0]["lr"]},
                        step=self.global_step,
                        run=self.wandb_run,
                    )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run validation using short-horizon forward (matches training)."""
        self.generator.eval()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            pred, target = self._short_horizon_forward(expression, audio, emotion, prev_expr)
            loss = l1_reconstruction_loss(pred, target, self.dim_weights)
            mse = torch.nn.functional.mse_loss(pred, target)

            total_loss += loss.item()
            total_mse += mse.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)

        if is_main_process():
            log_metrics(
                {
                    "stage2/val_loss": avg_loss,
                    "stage2/val_mse": avg_mse,
                    "stage2/epoch": epoch,
                },
                step=self.global_step,
                run=self.wandb_run,
            )
            print(f"  Val L1: {avg_loss:.4f}, Val MSE: {avg_mse:.4f}")

        return avg_loss

    def save(self, epoch: int, val_loss: float):
        """Save checkpoint (rank 0 only)."""
        if not is_main_process():
            return

        from dataclasses import asdict

        path = checkpoint_path(self.config.paths.checkpoint_dir, epoch, prefix="stage2")
        save_checkpoint(
            path, epoch, val_loss, asdict(self.config),
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
                generator=(self.generator, self.optimizer),
            )
            print(f"  New best model saved: {best_path}")
        else:
            self.epochs_without_improvement += 1

    def should_stop(self, patience: int) -> bool:
        """Check if training should stop due to no improvement."""
        return patience > 0 and self.epochs_without_improvement >= patience
