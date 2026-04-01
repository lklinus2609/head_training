"""FLAME mesh decoder wrapper for evaluation metrics (LVE)."""

import torch
import torch.nn as nn


# FLAME lip vertex indices (standard FLAME topology)
# These indices correspond to the inner and outer lip vertices in the FLAME mesh.
FLAME_LIP_VERTEX_IDS = [
    # Upper outer lip
    1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584,
    # Lower outer lip
    1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597,
    # Upper inner lip
    2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794,
    # Lower inner lip
    2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806,
]


class FLAMEDecoder:
    """Thin wrapper around the FLAME model for decoding expression parameters to mesh vertices.

    Used at evaluation time only (not during training backward passes).
    """

    def __init__(self, flame_model_path: str, device: str = "cpu"):
        """
        Args:
            flame_model_path: Path to the FLAME .pkl model file.
            device: Device to load the model on.
        """
        self.device = device
        self.flame = None
        self.flame_model_path = flame_model_path

    def _load_flame(self):
        """Lazy-load the FLAME model."""
        if self.flame is not None:
            return

        try:
            from flame_pytorch import FLAME, FLAMEConfig

            flame_config = FLAMEConfig(
                flame_model_path=self.flame_model_path,
                n_shape=300,
                n_exp=100,
            )
            self.flame = FLAME(flame_config).to(self.device).eval()
        except ImportError:
            raise ImportError(
                "flame-pytorch is required for FLAME mesh decoding. "
                "Install with: pip install flame-pytorch"
            )

    def expression_to_vertices(
        self, expression_params: torch.Tensor
    ) -> torch.Tensor:
        """Decode FLAME expression parameters to mesh vertices.

        Args:
            expression_params: Expression parameters [B, 100].

        Returns:
            Mesh vertices [B, V, 3] where V=5023.
        """
        self._load_flame()
        B = expression_params.size(0)

        with torch.no_grad():
            vertices, _ = self.flame(
                shape_params=torch.zeros(B, 300, device=self.device),
                expression_params=expression_params.to(self.device),
                pose_params=torch.zeros(B, 6, device=self.device),
            )
        return vertices

    def get_lip_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Extract lip vertices from full FLAME mesh.

        Args:
            vertices: Full mesh vertices [B, V, 3].

        Returns:
            Lip vertices [B, num_lip_verts, 3].
        """
        return vertices[:, FLAME_LIP_VERTEX_IDS, :]
