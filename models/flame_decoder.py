"""FLAME mesh decoder for evaluation metrics (LVE).

Loads the FLAME model directly from the pickle file, avoiding the flame-pytorch
package (which has a broken chumpy dependency). The expression decoding is a
simple linear operation: vertices = template + expr_basis @ expression_params.
"""

import pickle

import numpy as np
import torch


def _load_flame_pickle(flame_path: str) -> dict:
    """Load FLAME pickle. Requires chumpy to be installed."""
    with open(flame_path, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    return {k: np.array(v) if hasattr(v, "__array__") else v for k, v in model.items()}


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
    """Decodes FLAME expression parameters to mesh vertices.

    Loads the FLAME pickle directly and applies the expression blendshapes
    as a linear combination: vertices = template + expr_basis @ params.
    Used at evaluation time only (not during training backward passes).
    """

    def __init__(self, flame_model_path: str, device: str = "cpu"):
        self.device = device
        self.flame_model_path = flame_model_path
        self._template = None
        self._expr_basis = None

    def _load(self):
        """Lazy-load the FLAME model from pickle."""
        if self._template is not None:
            return

        model = _load_flame_pickle(self.flame_model_path)

        # Template vertices [5023, 3]
        v_template = np.array(model["v_template"], dtype=np.float32)
        self._template = torch.from_numpy(v_template).to(self.device)

        # Expression blendshapes: last 100 columns of shapedirs
        # shapedirs is [5023, 3, 400] where cols 300-399 are expression
        shapedirs = np.array(model["shapedirs"], dtype=np.float32)
        expr_basis = shapedirs[:, :, 300:400]  # [5023, 3, 100]
        # Reshape to [5023*3, 100] for matrix multiply
        expr_basis = expr_basis.reshape(-1, 100)
        self._expr_basis = torch.from_numpy(expr_basis).to(self.device)

    def expression_to_vertices(
        self, expression_params: torch.Tensor
    ) -> torch.Tensor:
        """Decode FLAME expression parameters to mesh vertices.

        Args:
            expression_params: Expression parameters [B, 100].

        Returns:
            Mesh vertices [B, V, 3] where V=5023.
        """
        self._load()
        B = expression_params.size(0)
        params = expression_params.to(self.device)

        # offsets = expr_basis @ params^T  ->  [5023*3, B]
        offsets = self._expr_basis @ params.T  # [5023*3, B]
        offsets = offsets.T.reshape(B, -1, 3)  # [B, 5023, 3]

        vertices = self._template.unsqueeze(0) + offsets  # [B, 5023, 3]
        return vertices

    def get_lip_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Extract lip vertices from full FLAME mesh.

        Args:
            vertices: Full mesh vertices [B, V, 3].

        Returns:
            Lip vertices [B, num_lip_verts, 3].
        """
        return vertices[:, FLAME_LIP_VERTEX_IDS, :]
