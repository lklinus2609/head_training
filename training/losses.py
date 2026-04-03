"""Loss functions for generator pretraining and adversarial co-training."""

import torch
import torch.nn.functional as F
from torch import autograd


def l1_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss between predicted and target FLAME expressions.

    Args:
        pred: Predicted expressions [B, T, D].
        target: Ground truth expressions [B, T, D].
    """
    return F.l1_loss(pred, target)


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
