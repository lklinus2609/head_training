"""Evaluation metrics: LVE, expression MSE, FGD."""

import numpy as np
import torch
from scipy.linalg import sqrtm


def lip_vertex_error(
    pred_expr: torch.Tensor,
    gt_expr: torch.Tensor,
    flame_decoder,
) -> float:
    """Compute Lip Vertex Error (LVE).

    Mean L2 distance between lip vertices of predicted and ground truth FLAME meshes.

    Args:
        pred_expr: Predicted expression params [B, 100] or [B, T, 100].
        gt_expr: Ground truth expression params, same shape as pred_expr.
        flame_decoder: FLAMEDecoder instance.

    Returns:
        Mean LVE in millimeters.
    """
    # Flatten temporal dimension if present
    if pred_expr.dim() == 3:
        B, T, D = pred_expr.shape
        pred_expr = pred_expr.reshape(B * T, D)
        gt_expr = gt_expr.reshape(B * T, D)

    # Decode to vertices
    pred_verts = flame_decoder.expression_to_vertices(pred_expr)
    gt_verts = flame_decoder.expression_to_vertices(gt_expr)

    # Extract lip vertices
    pred_lips = flame_decoder.get_lip_vertices(pred_verts)
    gt_lips = flame_decoder.get_lip_vertices(gt_verts)

    # L2 distance per vertex, averaged
    lve = torch.norm(pred_lips - gt_lips, dim=-1).mean().item()

    # Convert to mm (FLAME outputs in meters)
    return lve * 1000


def expression_mse(
    pred_expr: torch.Tensor,
    gt_expr: torch.Tensor,
) -> float:
    """Mean squared error between predicted and ground truth expression parameters.

    Args:
        pred_expr: Predicted [B, T, D] or [B, D].
        gt_expr: Ground truth, same shape.

    Returns:
        MSE scalar.
    """
    return torch.nn.functional.mse_loss(pred_expr, gt_expr).item()


def frechet_gesture_distance(
    real_features: np.ndarray,
    generated_features: np.ndarray,
) -> float:
    """Compute Frechet Gesture Distance (FGD).

    Analogous to FID for images: compares the distribution of real and generated
    motion feature embeddings using the Frechet distance between two Gaussians.

    Args:
        real_features: Feature embeddings of real sequences [N_real, D].
        generated_features: Feature embeddings of generated sequences [N_gen, D].

    Returns:
        FGD score (lower is better).
    """
    # Compute statistics
    mu_real = real_features.mean(axis=0)
    mu_gen = generated_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # Frechet distance
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)

    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fgd = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fgd)


def extract_motion_features(
    expression_sequences: np.ndarray,
    feature_dim: int = 64,
) -> np.ndarray:
    """Extract motion features for FGD computation.

    Uses PCA-based feature extraction as a simple baseline.
    For more accurate FGD, train a dedicated autoencoder on real motion data.

    Args:
        expression_sequences: Motion data [N, T, D].
        feature_dim: Output feature dimensionality.

    Returns:
        Feature embeddings [N, feature_dim].
    """
    from sklearn.decomposition import PCA

    N, T, D = expression_sequences.shape
    # Flatten temporal dimension
    flat = expression_sequences.reshape(N, T * D)

    # PCA reduction
    pca = PCA(n_components=min(feature_dim, flat.shape[0], flat.shape[1]))
    features = pca.fit_transform(flat)

    return features
