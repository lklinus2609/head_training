"""Strip a training checkpoint down to what inference needs.

Works on either a Stage 2 or Stage 3 checkpoint. Drops optimizer states,
the discriminator (stage 3 only), and RNG snapshots. Keeps the generator
weights, epoch/val_loss metadata, and the full config dict so the model
architecture can be rebuilt at load time (see viewer/generate_sequence.py).

Usage:
    python -m scripts.export_inference_checkpoint \
        --src $WORK/checkpoints/d4head/stage3_20260410_1500/checkpoint_epoch_0042.pt \
        --dst $WORK/checkpoints/d4head/stage3_best_inference.pt
"""

import argparse
from pathlib import Path

import torch


DROP_KEYS = (
    "generator_optimizer",
    "discriminator_model",
    "discriminator_optimizer",
    "rng_states",
)


def strip_checkpoint(src: str, dst: str) -> dict:
    """Load `src`, drop training-only keys, save to `dst`. Returns the stripped dict."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if "generator_model" not in ckpt:
        raise ValueError(
            f"{src}: no 'generator_model' key found — is this a valid training checkpoint?"
        )

    source_stage = "stage3" if "discriminator_model" in ckpt else "stage2"

    stripped = {
        "generator_model": ckpt["generator_model"],
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "config": ckpt.get("config", {}),
        "source_stage": source_stage,
    }

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, dst)

    src_bytes = Path(src).stat().st_size
    dst_bytes = Path(dst).stat().st_size
    pct = 100.0 * (1.0 - dst_bytes / src_bytes) if src_bytes else 0.0
    print(f"Source ({source_stage}):  {src}  ({src_bytes / 1e6:.2f} MB)")
    print(f"Stripped:               {dst}  ({dst_bytes / 1e6:.2f} MB)")
    print(f"Size reduction:         {pct:.1f}%")

    dropped = [k for k in DROP_KEYS if k in ckpt]
    if dropped:
        print(f"Dropped keys:           {', '.join(dropped)}")

    return stripped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--dst", required=True, help="Output path for inference checkpoint (.pt)")
    args = parser.parse_args()

    strip_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    main()
