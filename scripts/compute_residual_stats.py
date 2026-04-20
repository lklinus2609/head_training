"""Compute residual mean/std from a frozen stage-2 transformer on the train set.

Residual (in expression-normalized space) is `r = expression - stage2_pred`,
where `stage2_pred` comes from a teacher-forced forward pass. We compute per-dim
mean and std across the full training split and write them to `train.h5/stats/`
as `residual_mean` and `residual_std`. The residual-FM trainer uses these to
re-normalize the residual to unit std per dim — without this, lip dims (which
stage 2 predicts tightly) have near-zero residual and the FM collapses into
predicting -x_0 instead of the actual signal.

Run once before launching residual-FM training. Safe to re-run with
`--overwrite` if you change the frozen checkpoint.

Usage:
    python scripts/compute_residual_stats.py \\
        --config configs/residual_fm.yaml \\
        [--checkpoint $WORK/checkpoints/d4head/stage2_best.pt] \\
        [--batch_size 64] [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs_schema import load_config
from data.dataset import FacialMotionDataset
from models.generator import Generator


def _get_audio_dim(config) -> int:
    if config.data.audio_feature == "mel":
        return config.data.mel_n_mels
    return config.data.wav2vec_dim


def _build_frozen_stage2(config, ckpt_path: str, device: torch.device) -> Generator:
    gen = Generator(
        audio_dim=_get_audio_dim(config),
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


def _chan_update(count: int, mean: np.ndarray, M2: np.ndarray, batch: np.ndarray):
    """Chan's parallel algorithm for batched mean/variance updates.

    batch: [n, D]. Returns updated (count, mean, M2).
    """
    n_new = batch.shape[0]
    if n_new == 0:
        return count, mean, M2
    batch_mean = batch.mean(axis=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

    n_total = count + n_new
    delta = batch_mean - mean
    new_mean = mean + delta * (n_new / n_total)
    new_M2 = M2 + batch_m2 + (delta ** 2) * count * n_new / n_total
    return n_total, new_mean, new_M2


def main():
    parser = argparse.ArgumentParser(description="Compute residual stats for residual-FM training.")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (e.g. configs/residual_fm.yaml).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Stage 2 checkpoint path. Defaults to "
                             "config.residual_fm.frozen_stage2_checkpoint.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing residual_* stats in train.h5.")
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt_path = args.checkpoint or config.residual_fm.frozen_stage2_checkpoint
    if not ckpt_path:
        raise ValueError(
            "No stage 2 checkpoint provided. Pass --checkpoint or set "
            "residual_fm.frozen_stage2_checkpoint in the config."
        )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_h5 = str(Path(config.paths.processed_dir) / "train.h5")
    if not Path(train_h5).exists():
        raise FileNotFoundError(f"Train HDF5 not found: {train_h5}")

    print(f"Config:           {args.config}")
    print(f"Stage 2 ckpt:     {ckpt_path}")
    print(f"Train HDF5:       {train_h5}")
    print(f"Device:           {device}")

    dataset = FacialMotionDataset(
        train_h5,
        seq_len=config.data.seq_len,
        context_past=config.data.context_past,
        context_future=config.data.context_future,
        prev_frames=config.data.prev_frames,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    generator = _build_frozen_stage2(config, ckpt_path, device)
    print(f"Loaded frozen stage 2: {sum(p.numel() for p in generator.parameters()):,} params")

    D = config.data.flame_expr_dim
    count = 0
    mean = np.zeros(D, dtype=np.float64)
    M2 = np.zeros(D, dtype=np.float64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing residual stats"):
            audio = batch["audio"].to(device, non_blocking=True)
            expression = batch["expression"].to(device, non_blocking=True)
            emotion = batch["emotion"].to(device, non_blocking=True)
            prev = batch["prev_expression"].to(device, non_blocking=True)

            pred = generator(audio, emotion, prev, target_expression=expression)
            r = (expression - pred).cpu().numpy().astype(np.float64)  # [B, T, D]
            r = r.reshape(-1, D)
            count, mean, M2 = _chan_update(count, mean, M2, r)

    var = M2 / max(count - 1, 1)
    std = np.sqrt(var)
    std = np.where(std < 1e-8, 1.0, std)

    print(f"\nResidual stats: count={count:,}")
    print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}] (should be near 0)")
    print(f"  std  range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"  per-dim std (first 10): {np.array2string(std[:10], precision=3)}")

    with h5py.File(train_h5, "r+") as hf:
        if "stats" not in hf:
            raise RuntimeError(f"{train_h5}/stats group missing. Was preprocessing run?")
        stats = hf["stats"]
        for key, val in [("residual_mean", mean), ("residual_std", std)]:
            if key in stats:
                if not args.overwrite:
                    raise RuntimeError(
                        f"'{key}' already exists in {train_h5}/stats. "
                        "Pass --overwrite to replace."
                    )
                del stats[key]
            stats.create_dataset(key, data=val.astype(np.float32))
    print(f"\nWrote residual_mean + residual_std to {train_h5}/stats/")


if __name__ == "__main__":
    main()
