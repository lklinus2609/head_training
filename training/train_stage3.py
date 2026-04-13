"""Stage 3 entry point: adversarial co-training of generator and discriminator.

Launch with:
    torchrun --nproc_per_node=3 training/train_stage3.py --config configs/stage3_adversarial.yaml
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
from models.discriminator import TemporalDiscriminator
from models.generator import Generator
from training.trainer_stage3 import Stage3Trainer
from utils.checkpoint import create_run_dir, find_latest_checkpoint, load_checkpoint
from utils.ddp import cleanup_ddp, get_rank, is_main_process, setup_ddp
from utils.logging_utils import init_wandb
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Adversarial Co-Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--pretrained_gen", type=str, default=None,
                        help="Path to pretrained Stage 2 generator checkpoint. "
                             "Overrides config.stage3.pretrained_gen_path.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume Stage 3 from, or 'auto'")
    return parser.parse_args()


def get_audio_dim(config) -> int:
    if config.data.audio_feature == "mel":
        return config.data.mel_n_mels
    else:
        return config.data.wav2vec_dim


def main():
    args = parse_args()
    config = load_config(args.config)

    local_rank = setup_ddp()
    rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")

    seed_everything(config.stage3.seed + rank)

    audio_dim = get_audio_dim(config)

    if is_main_process():
        print(f"=== Stage 3: Adversarial Co-Training ===")
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    per_gpu_batch = config.stage3.batch_size // torch.distributed.get_world_size()

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

    # Generator
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

    # Load pretrained generator weights from Stage 2
    pretrained_path = args.pretrained_gen or config.stage3.pretrained_gen_path
    if not pretrained_path:
        # Auto-find most recent stage2 run's best checkpoint
        from pathlib import Path as _P
        ckpt_base = _P(config.paths.checkpoint_dir).parent if "stage3" in config.paths.checkpoint_dir else _P(config.paths.checkpoint_dir)
        stage2_runs = sorted(ckpt_base.glob("stage2_*/stage2_best.pt"))
        if stage2_runs:
            pretrained_path = str(stage2_runs[-1])
            if is_main_process():
                print(f"Auto-detected Stage 2 checkpoint: {pretrained_path}")
    if pretrained_path:
        if is_main_process():
            print(f"Loading pretrained generator from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_model"])

    # Discriminator
    # Input dim: expression(100) + velocity(100) + acceleration(100) + audio(C_audio)
    disc_input_dim = 3 * config.data.flame_expr_dim + audio_dim
    discriminator = TemporalDiscriminator(
        input_dim=disc_input_dim,
        d_model=config.discriminator.d_model,
        n_layers=config.discriminator.n_layers,
        n_heads=config.discriminator.n_heads,
        d_ff=config.discriminator.d_ff,
        dropout=config.discriminator.dropout,
        fc_hidden=config.discriminator.fc_hidden,
    ).to(device)

    # Wrap in DDP
    generator = DDP(generator, device_ids=[local_rank])
    discriminator = DDP(discriminator, device_ids=[local_rank])

    if is_main_process():
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(f"Generator params: {gen_params:,}, Discriminator params: {disc_params:,}")

    # Optimizers (separate for G and D with different LR and betas)
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.stage3.gen_lr,
        betas=(config.stage3.gen_beta1, config.stage3.gen_beta2),
        weight_decay=config.stage3.weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.stage3.disc_lr,
        betas=(config.stage3.disc_beta1, config.stage3.disc_beta2),
        weight_decay=config.stage3.weight_decay,
    )

    # wandb
    wandb_run = init_wandb(config, "stage3_adversarial")

    # Checkpoint directory: timestamped run folder
    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config.paths.checkpoint_dir, prefix="stage3")
    if resume_path:
        # Resume into the same run folder
        from pathlib import Path as _P
        config.paths.checkpoint_dir = str(_P(resume_path).parent)
        if is_main_process():
            print(f"Resuming Stage 3 from {resume_path}")
        ckpt = load_checkpoint(
            resume_path, device,
            generator=(generator, gen_optimizer),
            discriminator=(discriminator, disc_optimizer),
        )
        start_epoch = ckpt["epoch"] + 1
    else:
        # Fresh run — create new timestamped folder
        config.paths.checkpoint_dir = create_run_dir(config.paths.checkpoint_dir, "stage3")
    if is_main_process():
        print(f"Checkpoint dir: {config.paths.checkpoint_dir}")

    # Compute variance-weighted dimension weights from training stats
    dim_weights = None
    if "expr_std" in train_dataset.stats:
        expr_std = torch.from_numpy(train_dataset.stats["expr_std"]).float().to(device)
        dim_weights = expr_std / expr_std.mean()  # normalize to average 1.0
        if is_main_process():
            print(f"Dim weights range: [{dim_weights.min():.3f}, {dim_weights.max():.3f}]")

    # Training loop
    trainer = Stage3Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        wandb_run=wandb_run,
        dim_weights=dim_weights,
    )

    for epoch in range(start_epoch, config.stage3.epochs):
        train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n--- Epoch {epoch}/{config.stage3.epochs - 1} ---")

        metrics = trainer.train_epoch(epoch)

        if is_main_process():
            print(f"  D: {metrics['d_loss']:.4f}, G: {metrics['g_loss']:.4f}, "
                  f"Recon: {metrics['recon']:.4f}, Adv: {metrics['adv']:.4f}")

        # Validation
        val_loss = float("inf")
        if epoch % config.stage3.val_every == 0:
            val_loss = trainer.validate(epoch)

        # Checkpointing
        if epoch % config.stage3.save_every == 0:
            trainer.save(epoch, val_loss)

    trainer.save(config.stage3.epochs - 1, val_loss)

    if wandb_run:
        wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
