"""Loss functions for generator pretraining and adversarial co-training."""

import torch
import torch.nn.functional as F
from torch import autograd


def l1_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 reconstruction loss between predicted and target FLAME expressions.

    Args:
        pred: Predicted expressions [B, T, D].
        target: Ground truth expressions [B, T, D].
        dim_weights: Per-dimension weights [D], typically raw std normalized
            to mean 1.0. If None, uses uniform weighting.
    """
    if dim_weights is None:
        return F.l1_loss(pred, target)
    per_dim_loss = torch.abs(pred - target)  # [B, T, D]
    weighted = per_dim_loss * dim_weights  # [D] broadcasts over [B, T, D]
    return weighted.mean()


def variance_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-dim within-sample std matching.

    Penalizes the mismatch between predicted and target per-dim motion range,
    averaged across the batch. Complements L1: L1 minimizes at the conditional
    mean (collapsing variance on multimodal targets); this term keeps spread
    calibrated per dim.

    Args:
        pred: Predicted expressions [B, T, D].
        target: Ground truth expressions [B, T, D].
    """
    pred_std = pred.std(dim=1, unbiased=False).mean(dim=0)   # [D]
    gt_std = target.std(dim=1, unbiased=False).mean(dim=0)   # [D]
    return ((pred_std - gt_std) ** 2).mean()


def velocity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 on first-order frame-to-frame differences.

    Pressures the prediction's velocity to track GT velocity. Directly
    attacks motion-range damping: if GT has motion, pred must produce
    matching motion to minimize this loss — L1-on-position alone does not
    enforce this because a slowly-drifting damped prediction can still
    have low per-frame L1.

    Args:
        pred: Predicted expressions [B, T, D].
        target: Ground truth expressions [B, T, D].
        dim_weights: Per-dim weights [D], typically the same inverse-std
            weights applied to L1 reconstruction.
    """
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    return l1_reconstruction_loss(pred_vel, target_vel, dim_weights)


def acceleration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 on second-order frame-to-frame differences.

    Pressures predicted acceleration to match GT. Directly attacks
    jitter: a noisy prediction has large acceleration magnitudes even if
    it tracks the GT mean, so this term penalises frame-to-frame
    discontinuities that velocity alone does not catch.

    Args:
        pred: Predicted expressions [B, T, D].
        target: Ground truth expressions [B, T, D].
        dim_weights: Per-dim weights [D].
    """
    if pred.shape[1] < 3:
        return torch.zeros((), device=pred.device)
    pred_accel = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    target_accel = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    return l1_reconstruction_loss(pred_accel, target_accel, dim_weights)


def discriminator_loss(
    real_score: torch.Tensor,
    fake_score: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for the discriminator.

    Args:
        real_score: Discriminator output for real data [B, 1].
        fake_score: Discriminator output for generated data [B, 1].
    """
    real_loss = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
    fake_loss = F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
    return (real_loss + fake_loss) / 2


def gradient_penalty(
    discriminator: torch.nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
) -> torch.Tensor:
    """Gradient penalty using interpolated samples (WGAN-GP style).

    Args:
        discriminator: The discriminator network.
        real_data: Real discriminator input windows [B, K, D].
        fake_data: Fake discriminator input windows [B, K, D].

    Returns:
        Gradient penalty scalar.
    """
    B = real_data.size(0)
    alpha = torch.rand(B, 1, 1, device=real_data.device)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).detach()
    interpolated.requires_grad_(True)

    # Force math-based attention kernel because SDPA efficient/flash kernels
    # do not support second-order gradients needed by gradient penalty.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        score = discriminator(interpolated)
    grad = autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.reshape(B, -1).norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp


def generator_adversarial_loss(fake_score: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator adversarial loss.

    The generator wants the discriminator to classify generated data as real.

    Args:
        fake_score: Discriminator output for generated data [B, 1].
    """
    return F.binary_cross_entropy_with_logits(fake_score, torch.ones_like(fake_score))
