"""Stage 2 entry point: generator pretraining with L1 reconstruction loss.

Launch with:
    torchrun --nproc_per_node=3 training/train_stage2.py --config configs/stage2_pretrain.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs_schema import load_config
from data.dataset import FacialMotionDataset
from models.generator import Generator
from training.trainer_stage2 import Stage2Trainer
from utils.checkpoint import create_run_dir, find_latest_checkpoint, load_checkpoint
from utils.ddp import cleanup_ddp, get_rank, is_main_process, setup_ddp
from utils.logging_utils import init_wandb
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Generator Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from, or 'auto' to find latest")
    return parser.parse_args()


def get_audio_dim(config) -> int:
    """Determine audio feature dimensionality from config."""
    if config.data.audio_feature == "mel":
        return config.data.mel_n_mels  # 80
    else:
        return config.data.wav2vec_dim  # 768


def main():
    args = parse_args()
    config = load_config(args.config)

    # DDP setup
    local_rank = setup_ddp()
    rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # Seed
    seed_everything(config.stage2.seed + rank)

    if is_main_process():
        print(f"=== Stage 2: Generator Pretraining ===")
        print(f"Config: {args.config}")
        print(f"Device: {device}, World size: {torch.distributed.get_world_size()}")

    # Dataset
    train_h5 = str(Path(config.paths.processed_dir) / "train.h5")
    val_h5 = str(Path(config.paths.processed_dir) / "val.h5")

    train_dataset = FacialMotionDataset(
        train_h5,
        seq_len=config.data.seq_len,
        context_past=config.data.context_past,
        context_future=config.data.context_future,
        prev_frames=config.data.prev_frames,
    )
    val_dataset = FacialMotionDataset(
        val_h5,
        seq_len=config.data.seq_len,
        context_past=config.data.context_past,
        context_future=config.data.context_future,
        prev_frames=config.data.prev_frames,
    )

    # DataLoaders with DDP sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    per_gpu_batch = config.stage2.batch_size // torch.distributed.get_world_size()

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Model
    audio_dim = get_audio_dim(config)
    generator = Generator(
        audio_dim=audio_dim,
        expr_dim=config.data.flame_expr_dim,
        d_model=config.generator.d_model,
        n_layers=config.generator.n_layers,
        n_heads=config.generator.n_heads,
        d_ff=config.generator.d_ff,
        dropout=config.generator.dropout,
        n_emotions=config.generator.n_emotions,
        emotion_embed_dim=config.generator.emotion_embed_dim,
        prev_frames=config.data.prev_frames,
        audio_conv_channels=config.generator.audio_conv_channels,
        audio_conv_kernel_sizes=config.generator.audio_conv_kernel_sizes,
    ).to(device)

    generator = DDP(generator, device_ids=[local_rank], find_unused_parameters=True)

    if is_main_process():
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"Generator params: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.stage2.lr,
        betas=(config.stage2.beta1, config.stage2.beta2),
        weight_decay=config.stage2.weight_decay,
    )

    # wandb
    wandb_run = init_wandb(config, "stage2_pretrain")

    # Checkpoint directory: timestamped run folder
    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config.paths.checkpoint_dir, prefix="stage2")
    if resume_path:
        # Resume into the same run folder
        from pathlib import Path as _P
        config.paths.checkpoint_dir = str(_P(resume_path).parent)
        if is_main_process():
            print(f"Resuming from {resume_path}")
        ckpt = load_checkpoint(resume_path, device, generator=(generator, optimizer))
        start_epoch = ckpt["epoch"] + 1
    else:
        # Fresh run — create new timestamped folder
        config.paths.checkpoint_dir = create_run_dir(config.paths.checkpoint_dir, "stage2")
    if is_main_process():
        print(f"Checkpoint dir: {config.paths.checkpoint_dir}")

    # Compute variance-weighted dimension weights from training stats
    dim_weights = None
    if "expr_std" in train_dataset.stats:
        import numpy as np
        expr_std = torch.from_numpy(train_dataset.stats["expr_std"]).float().to(device)
        dim_weights = expr_std / expr_std.mean()  # normalize to average 1.0
        if is_main_process():
            print(f"Dim weights range: [{dim_weights.min():.3f}, {dim_weights.max():.3f}]")

    # Training loop
    trainer = Stage2Trainer(
        config=config,
        generator=generator,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        wandb_run=wandb_run,
        dim_weights=dim_weights,
    )

    for epoch in range(start_epoch, config.stage2.epochs):
        train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n--- Epoch {epoch}/{config.stage2.epochs - 1} ---")

        train_loss = trainer.train_epoch(epoch)

        if is_main_process():
            print(f"  Train L1: {train_loss:.4f}")

        # Validation
        val_loss = float("inf")
        if epoch % config.stage2.val_every == 0:
            val_loss = trainer.validate(epoch)

        # Checkpointing
        if epoch % config.stage2.save_every == 0:
            trainer.save(epoch, val_loss)

        # Early stopping
        if trainer.should_stop(config.stage2.patience):
            if is_main_process():
                print(f"\nEarly stopping: no improvement for {config.stage2.patience} epochs")
            break

    # Save final checkpoint
    trainer.save(epoch, val_loss)

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
