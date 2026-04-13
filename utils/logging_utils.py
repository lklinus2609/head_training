from dataclasses import asdict
from datetime import datetime

import wandb

from utils.ddp import is_main_process


def init_wandb(config, stage_name: str, run_name: str | None = None):
    """Initialize wandb on rank 0 only.

    Args:
        config: TrainConfig dataclass.
        stage_name: e.g. "stage2_pretrain" or "stage3_adversarial".
        run_name: Optional custom run name.

    Returns:
        wandb.Run or None (if not rank 0 or wandb disabled).
    """
    if not is_main_process():
        return None

    if run_name is None:
        run_name = f"{stage_name}_{datetime.now():%Y%m%d_%H%M}"

    run = wandb.init(
        project=f"{config.wandb_project}-{stage_name}",
        entity=config.wandb_entity or None,
        config=asdict(config),
        name=run_name,
        dir=config.paths.wandb_dir,
        resume="allow",
    )
    return run


def log_metrics(metrics: dict, step: int, run=None):
    """Log metrics to wandb (rank 0 only)."""
    if run is None:
        return
    run.log(metrics, step=step)


def log_video(frames, caption: str, step: int, run=None, fps: int = 30):
    """Log a video to wandb (rank 0 only).

    Args:
        frames: numpy array of shape [T, H, W, C] (uint8).
        caption: Description for the video.
        step: Global step.
        run: wandb run object.
        fps: Frames per second for playback.
    """
    if run is None:
        return
    video = wandb.Video(frames, caption=caption, fps=fps)
    run.log({caption: video}, step=step)
