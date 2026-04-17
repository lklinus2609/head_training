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
