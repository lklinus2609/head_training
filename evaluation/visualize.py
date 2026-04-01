"""Visualization utilities: expression trajectory plots and FLAME mesh rendering."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_expression_trajectories(
    pred_expr: np.ndarray,
    gt_expr: np.ndarray,
    dims: list[int] | None = None,
    dim_names: list[str] | None = None,
    save_path: str | None = None,
    title: str = "Expression Trajectories",
) -> plt.Figure:
    """Plot predicted vs ground truth expression parameter trajectories.

    Args:
        pred_expr: Predicted expressions [T, D].
        gt_expr: Ground truth expressions [T, D].
        dims: Which expression dimensions to plot (default: first 6).
        dim_names: Names for the plotted dimensions.
        save_path: If provided, save the figure to this path.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if dims is None:
        dims = list(range(min(6, pred_expr.shape[-1])))
    if dim_names is None:
        dim_names = [f"Expr {d}" for d in dims]

    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    T = pred_expr.shape[0]
    t = np.arange(T) / 30.0  # Convert to seconds at 30fps

    for ax, dim, name in zip(axes, dims, dim_names):
        ax.plot(t, gt_expr[:, dim], label="Ground Truth", color="blue", alpha=0.7)
        ax.plot(t, pred_expr[:, dim], label="Predicted", color="red", alpha=0.7, linestyle="--")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    metrics_history: dict[str, list[float]],
    save_path: str | None = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """Plot training loss curves from logged metrics.

    Args:
        metrics_history: Dict mapping metric names to lists of values.
        save_path: If provided, save the figure.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    n_metrics = len(metrics_history)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics_history.items()):
        ax.plot(values)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def render_flame_sequence(
    expression_params: np.ndarray,
    flame_decoder,
    resolution: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Render a sequence of FLAME meshes driven by expression parameters.

    Args:
        expression_params: Expression parameters [T, 100].
        flame_decoder: FLAMEDecoder instance.
        resolution: Output image resolution (H, W).

    Returns:
        Video frames [T, H, W, 3] as uint8.
    """
    try:
        import pyrender
        import trimesh
        import torch
    except ImportError:
        print("pyrender and trimesh are required for mesh rendering.")
        return np.zeros((expression_params.shape[0], *resolution, 3), dtype=np.uint8)

    T = expression_params.shape[0]
    frames = []

    # Set up scene
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 0.5  # Move camera back
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(*resolution)

    for t in range(T):
        expr = torch.from_numpy(expression_params[t:t+1]).float()
        vertices = flame_decoder.expression_to_vertices(expr)
        verts_np = vertices[0].cpu().numpy()

        # Create mesh (FLAME has a fixed topology)
        mesh = trimesh.Trimesh(vertices=verts_np)
        mesh_node = pyrender.Mesh.from_trimesh(mesh)

        node = scene.add(mesh_node)
        color, _ = renderer.render(scene)
        frames.append(color)
        scene.remove_node(node)

    renderer.delete()
    return np.stack(frames, axis=0)
