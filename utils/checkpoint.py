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
    **model_optimizer_pairs,
):
    """Save a training checkpoint.

    Args:
        path: File path to save to.
        epoch: Current epoch number.
        val_loss: Current validation loss.
        config: Training config as dict.
        **model_optimizer_pairs: Keyword args where keys are names and values
            are (model, optimizer) tuples. E.g. generator=(gen_model, gen_opt).
    """
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
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

    # Restore RNG states
    if "rng_states" in ckpt:
        rng = ckpt["rng_states"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])

    return ckpt


def find_latest_checkpoint(checkpoint_dir: str, prefix: str = "checkpoint") -> str | None:
    """Find the most recent checkpoint by epoch number.

    Expects filenames like: checkpoint_epoch_042.pt
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pt")
    best_epoch = -1
    best_path = None

    for f in ckpt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = str(f)

    return best_path


def checkpoint_path(checkpoint_dir: str, epoch: int, prefix: str = "checkpoint") -> str:
    """Generate a checkpoint file path for a given epoch."""
    return str(Path(checkpoint_dir) / f"{prefix}_epoch_{epoch:04d}.pt")
