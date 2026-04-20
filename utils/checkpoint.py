import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def save_checkpoint(
    path: str,
    epoch: int,
    val_loss: float,
    config: dict,
    global_step: int | None = None,
    **model_optimizer_pairs,
):
    """Save a training checkpoint.

    Args:
        path: File path to save to.
        epoch: Current epoch number.
        val_loss: Current validation loss.
        config: Training config as dict.
        global_step: Trainer global step for wandb x-axis continuity across
            resumes. None is tolerated for legacy callers.
        **model_optimizer_pairs: Keyword args where keys are names and values
            are (model, optimizer) tuples. E.g. generator=(gen_model, gen_opt).
    """
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
        "global_step": global_step,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }

    for name, (model, optimizer) in model_optimizer_pairs.items():
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        state[f"{name}_model"] = model_state
        if optimizer is not None:
            state[f"{name}_optimizer"] = optimizer.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    **model_optimizer_pairs,
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        device: Device to map tensors to.
        **model_optimizer_pairs: Same format as save_checkpoint.

    Returns:
        The full checkpoint dict (for epoch, val_loss, etc.).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    for name, (model, optimizer) in model_optimizer_pairs.items():
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt[f"{name}_model"])
        if optimizer is not None and f"{name}_optimizer" in ckpt:
            optimizer.load_state_dict(ckpt[f"{name}_optimizer"])

    # Restore RNG states (best-effort, skip on type errors from device mapping)
    if "rng_states" in ckpt:
        rng = ckpt["rng_states"]
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.random.set_rng_state(rng["torch"].cpu() if isinstance(rng["torch"], torch.Tensor) else rng["torch"])
            cuda_states = rng["cuda"]
            if cuda_states and len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states])
        except Exception:
            pass  # RNG restore is nice-to-have, not critical

    return ckpt


def save_inference_checkpoint(path: str, model, config: dict, epoch: int | None = None, val_loss: float | None = None):
    """Save a lean, inference-only checkpoint.

    Drops optimizer state, discriminator, and RNG snapshots that full training
    checkpoints carry. Matches the schema produced by
    scripts/export_inference_checkpoint.py so inference code can load either.
    """
    model_to_save = model.module if isinstance(model, DDP) else model
    state = {
        "generator_model": model_to_save.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
        "source_stage": None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def find_latest_checkpoint(checkpoint_dir: str, prefix: str = "checkpoint") -> str | None:
    """Find the most recent checkpoint in the most recent run folder.

    Looks for timestamped subdirectories like prefix_20260413_1545/,
    picks the newest one, then finds the highest epoch checkpoint inside it.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    # Find the most recent run folder by sorting timestamped names
    run_dirs = sorted(ckpt_dir.glob(f"{prefix}_*"), key=lambda p: p.name, reverse=True)
    run_dirs = [d for d in run_dirs if d.is_dir()]

    if not run_dirs:
        return None

    # Search within the most recent run folder only
    latest_dir = run_dirs[0]
    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pt")
    best_epoch = -1
    best_path = None

    for f in latest_dir.glob(f"{prefix}_epoch_*.pt"):
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = str(f)

    return best_path


def find_latest_best_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the best checkpoint in the most recent run folder, across any prefix.

    Scans `checkpoint_dir` for subdirectories named `<prefix>_YYYYMMDD_HHMM`
    (stage2, stage3, fm, residual_fm, ...). Picks the newest by timestamp,
    then returns:
      1. `<prefix>_best.pt` inside it, if present;
      2. otherwise the highest-epoch `<prefix>_epoch_XXXX.pt` inside it;
      3. otherwise None.

    Used by inference scripts to default `--checkpoint` to the latest trained
    model without requiring the user to know the exact path.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    ts_re = re.compile(r"^(.+)_(\d{8}_\d{4})$")
    run_dirs = []
    for d in ckpt_dir.iterdir():
        if not d.is_dir():
            continue
        m = ts_re.match(d.name)
        if m:
            run_dirs.append((d, m.group(1), m.group(2)))
    if not run_dirs:
        return None

    run_dirs.sort(key=lambda t: t[2], reverse=True)
    latest_dir, prefix, _ = run_dirs[0]

    best_path = latest_dir / f"{prefix}_best.pt"
    if best_path.exists():
        return str(best_path)

    ep_re = re.compile(rf"{re.escape(prefix)}_epoch_(\d+)\.pt")
    best_epoch = -1
    best_file = None
    for f in latest_dir.glob(f"{prefix}_epoch_*.pt"):
        m = ep_re.match(f.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_file = str(f)
    return best_file


def create_run_dir(checkpoint_dir: str, prefix: str) -> str:
    """Create a timestamped run directory for checkpoints.

    Returns:
        Path like: checkpoint_dir/stage2_20260408_1430/
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(checkpoint_dir) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def checkpoint_path(checkpoint_dir: str, epoch: int, prefix: str = "checkpoint") -> str:
    """Generate a checkpoint file path for a given epoch."""
    return str(Path(checkpoint_dir) / f"{prefix}_epoch_{epoch:04d}.pt")
