"""Residual flow-matching training entry point (Track C).

Loads a frozen stage-2 transformer and precomputed residual stats, builds a
fresh FM generator, trains it on `(x_gt - stage2_pred)` normalized by
residual_std. See `training/trainer_residual_fm.py` for the training loop.

Launch with:
    torchrun --nproc_per_node=3 training/train_residual_fm.py \\
        --config configs/residual_fm.yaml

Resume:
    torchrun ... --resume auto
    torchrun ... --resume $WORK/checkpoints/d4head/residual_fm_YYYYMMDD_HHMM/residual_fm_epoch_0010.pt

Pre-requisites:
  1. Stage 2 checkpoint at config.residual_fm.frozen_stage2_checkpoint.
  2. Residual stats in train.h5/stats — run scripts/compute_residual_stats.py.
"""

import argparse
import sys
from pathlib import Path

import h5py
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs_schema import load_config
from data.dataset import FacialMotionDataset
from models.generator import Generator
from models.generator_fm import GeneratorFM
from training.trainer_residual_fm import ResidualFMTrainer
from utils.checkpoint import create_run_dir, find_latest_checkpoint, load_checkpoint
from utils.ddp import cleanup_ddp, get_rank, is_main_process, setup_ddp
from utils.logging_utils import init_wandb
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Residual flow-matching training (Track C)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from, or 'auto' to find latest")
    return parser.parse_args()


def get_audio_dim(config) -> int:
    if config.data.audio_feature == "mel":
        return config.data.mel_n_mels
    return config.data.wav2vec_dim


def _load_residual_stats(train_h5_path: str, device: torch.device):
    with h5py.File(train_h5_path, "r") as hf:
        if "stats" not in hf or "residual_mean" not in hf["stats"] or "residual_std" not in hf["stats"]:
            raise RuntimeError(
                f"Residual stats missing from {train_h5_path}/stats. "
                "Run `python scripts/compute_residual_stats.py --config <your_config>` first."
            )
        r_mean = torch.from_numpy(hf["stats"]["residual_mean"][:]).float().to(device)
        r_std = torch.from_numpy(hf["stats"]["residual_std"][:]).float().to(device)
    r_std = torch.where(r_std < 1e-8, torch.ones_like(r_std), r_std)
    return r_mean, r_std


def _build_frozen_stage2(config, ckpt_path: str, device: torch.device) -> Generator:
    gen = Generator(
        audio_dim=get_audio_dim(config),
        expr_dim=config.data.flame_expr_dim,
        d_model=config.generator.d_model,
        n_layers=config.generator.n_layers,
        n_heads=config.generator.n_heads,
        d_ff=config.generator.d_ff,
        dropout=0.0,
        n_emotions=config.generator.n_emotions,
        emotion_embed_dim=config.generator.emotion_embed_dim,
        prev_frames=config.data.prev_frames,
        audio_conv_channels=config.generator.audio_conv_channels,
        audio_conv_kernel_sizes=config.generator.audio_conv_kernel_sizes,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator_model"])
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen


def main():
    args = parse_args()
    config = load_config(args.config)

    local_rank = setup_ddp()
    rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")

    seed_everything(config.fm.seed + rank)

    if is_main_process():
        print(f"=== Residual Flow-Matching Training (Track C) ===")
        print(f"Config: {args.config}")
        print(f"Device: {device}, World size: {torch.distributed.get_world_size()}")

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

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    per_gpu_batch = config.fm.batch_size // torch.distributed.get_world_size()

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

    audio_dim = get_audio_dim(config)
    generator = GeneratorFM(
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
        time_embed_dim=config.fm.time_embed_dim,
        audio_conv_channels=config.generator.audio_conv_channels,
        audio_conv_kernel_sizes=config.generator.audio_conv_kernel_sizes,
    ).to(device)
    generator = DDP(generator, device_ids=[local_rank])

    if not config.residual_fm.frozen_stage2_checkpoint:
        raise ValueError("residual_fm.frozen_stage2_checkpoint must be set in the config.")
    frozen_stage2 = _build_frozen_stage2(
        config, config.residual_fm.frozen_stage2_checkpoint, device
    )
    if is_main_process():
        frozen_trainable = sum(p.numel() for p in frozen_stage2.parameters() if p.requires_grad)
        assert frozen_trainable == 0, "frozen stage 2 has trainable params — disentanglement broken"
        print(f"Loaded frozen stage 2 from {config.residual_fm.frozen_stage2_checkpoint}")
        print(f"Frozen stage 2 params (requires_grad=False): "
              f"{sum(p.numel() for p in frozen_stage2.parameters()):,}")

    r_mean, r_std = _load_residual_stats(train_h5, device)
    if is_main_process():
        print(f"Residual stats: mean range [{r_mean.min():.4f}, {r_mean.max():.4f}], "
              f"std range [{r_std.min():.4f}, {r_std.max():.4f}]")
        fm_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(f"Trainable FM params: {fm_trainable:,}")

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.fm.lr,
        betas=(config.fm.beta1, config.fm.beta2),
        weight_decay=config.fm.weight_decay,
    )

    wandb_run = init_wandb(config, "residual_fm_training")

    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config.paths.checkpoint_dir, prefix="residual_fm")
    if resume_path:
        config.paths.checkpoint_dir = str(Path(resume_path).parent)
        if is_main_process():
            print(f"Resuming from {resume_path}")
        ckpt = load_checkpoint(resume_path, device, generator=(generator, optimizer))
        start_epoch = ckpt["epoch"] + 1
    else:
        config.paths.checkpoint_dir = create_run_dir(config.paths.checkpoint_dir, "residual_fm")
    if is_main_process():
        print(f"Checkpoint dir: {config.paths.checkpoint_dir}")

    # Residuals are pre-normalized to unit std per dim, so uniform MSE is the
    # right loss. Do not pass expr_std-based dim_weights.
    dim_weights = None

    trainer = ResidualFMTrainer(
        config=config,
        generator=generator,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        frozen_stage2=frozen_stage2,
        residual_std=r_std,
        residual_mean=r_mean,
        wandb_run=wandb_run,
        dim_weights=dim_weights,
    )

    if resume_path:
        trainer.global_step = ckpt.get("global_step") or 0

    scheduler = None
    if config.fm.use_cosine_lr:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.fm.epochs,
            eta_min=config.fm.lr * config.fm.cosine_lr_min_ratio,
        )
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, config.fm.epochs):
        train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n--- Epoch {epoch}/{config.fm.epochs - 1} ---")

        train_loss = trainer.train_epoch(epoch)

        if is_main_process():
            print(f"  Train Residual FM Loss: {train_loss:.4f}")

        val_loss = float("inf")
        if epoch % config.fm.val_every == 0:
            val_loss = trainer.validate(epoch)

        if epoch % config.fm.save_every == 0:
            trainer.save(epoch, val_loss)

        if scheduler is not None:
            scheduler.step()

        if trainer.should_stop(config.fm.patience):
            if is_main_process():
                print(f"\nEarly stopping: no improvement for {config.fm.patience} epochs")
            break

    trainer.save(epoch, val_loss)

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
