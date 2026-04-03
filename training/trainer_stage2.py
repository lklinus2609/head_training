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

            # Forward pass with teacher forcing
            pred = self.generator(audio, emotion, prev_expr, target_expression=expression)
            loss = l1_reconstruction_loss(pred, expression, self.dim_weights)

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
        """Run validation and return average loss."""
        self.generator.eval()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            expression = batch["expression"].to(self.device)
            emotion = batch["emotion"].to(self.device)
            prev_expr = batch["prev_expression"].to(self.device)

            pred = self.generator(audio, emotion, prev_expr, target_expression=expression)
            loss = l1_reconstruction_loss(pred, expression, self.dim_weights)
            mse = torch.nn.functional.mse_loss(pred, expression)

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
            best_path = str(
                __import__("pathlib").Path(self.config.paths.checkpoint_dir) / "stage2_best.pt"
            )
            save_checkpoint(
                best_path, epoch, val_loss, asdict(self.config),
                generator=(self.generator, self.optimizer),
            )
            print(f"  New best model saved: {best_path}")
