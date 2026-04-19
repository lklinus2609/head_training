"""Flow-matching training entry point (parallel track to train_stage2/3).

Launch with:
    torchrun --nproc_per_node=3 training/train_fm.py --config configs/fm.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs_schema import load_config
from data.dataset import FacialMotionDataset
from models.generator_fm import GeneratorFM
from training.trainer_fm import FMTrainer
from utils.checkpoint import create_run_dir, find_latest_checkpoint, load_checkpoint
from utils.ddp import cleanup_ddp, get_rank, is_main_process, setup_ddp
from utils.logging_utils import init_wandb
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Flow-matching training (Track B)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from, or 'auto' to find latest")
    return parser.parse_args()


def get_audio_dim(config) -> int:
    if config.data.audio_feature == "mel":
        return config.data.mel_n_mels
    return config.data.wav2vec_dim


def main():
    args = parse_args()
    config = load_config(args.config)

    local_rank = setup_ddp()
    rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")

    seed_everything(config.fm.seed + rank)

    if is_main_process():
        print(f"=== Flow-Matching Training (Track B) ===")
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

    if is_main_process():
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"GeneratorFM params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.fm.lr,
        betas=(config.fm.beta1, config.fm.beta2),
        weight_decay=config.fm.weight_decay,
    )

    wandb_run = init_wandb(config, "fm_training")

    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config.paths.checkpoint_dir, prefix="fm")
    if resume_path:
        config.paths.checkpoint_dir = str(Path(resume_path).parent)
        if is_main_process():
            print(f"Resuming from {resume_path}")
        ckpt = load_checkpoint(resume_path, device, generator=(generator, optimizer))
        start_epoch = ckpt["epoch"] + 1
    else:
        config.paths.checkpoint_dir = create_run_dir(config.paths.checkpoint_dir, "fm")
    if is_main_process():
        print(f"Checkpoint dir: {config.paths.checkpoint_dir}")

    dim_weights = None
    if "expr_std" in train_dataset.stats:
        expr_std = torch.from_numpy(train_dataset.stats["expr_std"]).float().to(device)
        dim_weights = expr_std / expr_std.mean()
        if is_main_process():
            print(f"Dim weights range: [{dim_weights.min():.3f}, {dim_weights.max():.3f}]")

    trainer = FMTrainer(
        config=config,
        generator=generator,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
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
            print(f"  Train FM Loss: {train_loss:.4f}")

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
