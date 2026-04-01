"""Training schedulers: adversarial weight warmup and learning rate schedules."""

import math


def get_lambda_adv(step: int, warmup_steps: int, start: float, end: float) -> float:
    """Linear warmup of adversarial loss weight.

    Args:
        step: Current training step.
        warmup_steps: Number of steps over which to linearly increase lambda.
        start: Initial lambda value.
        end: Final lambda value.

    Returns:
        Current lambda_adv value.
    """
    if step >= warmup_steps:
        return end
    return start + (end - start) * step / warmup_steps


def cosine_lr_lambda(step: int, total_steps: int, warmup_steps: int = 0, min_ratio: float = 0.1):
    """Cosine learning rate decay with optional linear warmup.

    Args:
        step: Current step.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        min_ratio: Minimum LR as a fraction of initial LR.

    Returns:
        LR multiplier.
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
