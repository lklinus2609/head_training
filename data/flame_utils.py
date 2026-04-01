"""FLAME expression parameter utilities: derivatives, normalization, resampling."""

import numpy as np
from scipy.interpolate import interp1d


def compute_velocity(expression: np.ndarray) -> np.ndarray:
    """Compute first-order temporal derivatives (velocity).

    Uses first-order finite differences: v_t = e_t - e_{t-1}.
    First frame uses forward difference.

    Args:
        expression: Array of shape [T, D].

    Returns:
        Velocity array of shape [T, D].
    """
    velocity = np.zeros_like(expression)
    velocity[1:] = expression[1:] - expression[:-1]
    velocity[0] = velocity[1]  # Forward difference for first frame
    return velocity


def compute_acceleration(expression: np.ndarray) -> np.ndarray:
    """Compute second-order temporal derivatives (acceleration).

    Uses second-order finite differences: a_t = e_{t+1} - 2*e_t + e_{t-1}.
    Boundary frames are copied from nearest interior frame.

    Args:
        expression: Array of shape [T, D].

    Returns:
        Acceleration array of shape [T, D].
    """
    acceleration = np.zeros_like(expression)
    if expression.shape[0] >= 3:
        acceleration[1:-1] = expression[2:] - 2 * expression[1:-1] + expression[:-2]
        acceleration[0] = acceleration[1]
        acceleration[-1] = acceleration[-2]
    return acceleration


def resample_to_fps(
    data: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample temporal data to a target frame rate using linear interpolation.

    Args:
        data: Array of shape [T_source, ...].
        source_fps: Original frame rate.
        target_fps: Desired frame rate.

    Returns:
        Resampled array of shape [T_target, ...].
    """
    if abs(source_fps - target_fps) < 0.01:
        return data

    T_source = data.shape[0]
    duration = (T_source - 1) / source_fps
    T_target = int(round(duration * target_fps)) + 1

    source_times = np.linspace(0, duration, T_source)
    target_times = np.linspace(0, duration, T_target)

    # Handle multi-dimensional data
    original_shape = data.shape
    flat_data = data.reshape(T_source, -1)

    interp_func = interp1d(source_times, flat_data, axis=0, kind="linear", fill_value="extrapolate")
    resampled = interp_func(target_times)

    return resampled.reshape(T_target, *original_shape[1:])


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    std_safe = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std_safe


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse Z-score normalization."""
    return data * std + mean
