"""Export FLAME template mesh + expression basis to a binary file for the web viewer.

The FLAME model pickle contains:
  - v_template: [5023, 3] template vertex positions
  - f: [9976, 3] triangle face indices
  - shapedirs: [5023, 3, 400] where columns 300-399 are expression blendshapes

This script exports a binary file with:
  - Header: num_vertices(u32), num_faces(u32), num_expr(u32)
  - Template vertices: float32[num_verts * 3]
  - Face indices: uint32[num_faces * 3]
  - Expression basis: float32[num_expr * num_verts * 3]  (deltas from template)

Usage:
    python export_flame_mesh.py --flame_path $WORK/models/flame/generic_model.pkl --output static/flame_data.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import pickle


def load_flame_model(flame_path: str) -> dict:
    """Load the FLAME model from a pickle file without requiring chumpy.

    Uses a sys.modules hack to make pickle think chumpy is available,
    mapping all chumpy classes to a simple wrapper that stores data as numpy.
    """
    import types

    # Create fake chumpy modules so pickle can resolve class references
    class _Ch(np.ndarray):
        """Minimal chumpy.Ch stand-in that behaves as ndarray."""
        def __new__(cls, *args, **kwargs):
            return np.array(0.0).view(cls)
        def __setstate__(self, state):
            # chumpy pickles store data in state; just ignore non-array state
            if isinstance(state, tuple):
                # ndarray.__setstate__ expects (version, shape, dtype, fortran, data)
                try:
                    super().__setstate__(state)
                except Exception:
                    pass
            elif isinstance(state, dict):
                pass

    fake_ch = types.ModuleType("chumpy")
    fake_ch.Ch = _Ch
    fake_ch.ch = types.ModuleType("chumpy.ch")
    fake_ch.ch.Ch = _Ch
    fake_ch.ch.MatVecMult = _Ch
    fake_ch.utils = types.ModuleType("chumpy.utils")
    fake_ch.logic = types.ModuleType("chumpy.logic")
    fake_ch.reordering = types.ModuleType("chumpy.reordering")

    # Assign _Ch as fallback for any attribute access
    for mod in [fake_ch, fake_ch.ch, fake_ch.utils, fake_ch.logic, fake_ch.reordering]:
        mod.__dict__.setdefault("__getattr__", lambda name, _C=_Ch: _C)

    import sys
    saved = {}
    for name in list(sys.modules.keys()):
        if name.startswith("chumpy"):
            saved[name] = sys.modules[name]

    sys.modules["chumpy"] = fake_ch
    sys.modules["chumpy.ch"] = fake_ch.ch
    sys.modules["chumpy.utils"] = fake_ch.utils
    sys.modules["chumpy.logic"] = fake_ch.logic
    sys.modules["chumpy.reordering"] = fake_ch.reordering

    try:
        with open(flame_path, "rb") as f:
            model = pickle.load(f, encoding="latin1")
    finally:
        # Restore original modules
        for name in ["chumpy", "chumpy.ch", "chumpy.utils", "chumpy.logic", "chumpy.reordering"]:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)

    # Convert everything to numpy
    result = {}
    for key, val in model.items():
        try:
            result[key] = np.array(val)
        except Exception:
            result[key] = val
    return result


def export_binary(flame_path: str, output_path: str, n_expression: int = 100):
    """Export FLAME mesh data to a binary file for the web viewer."""
    print(f"Loading FLAME model from {flame_path}...")
    model = load_flame_model(flame_path)

    # Extract data
    v_template = np.array(model["v_template"], dtype=np.float32)  # [5023, 3]
    faces = np.array(model["f"], dtype=np.int32)  # [9976, 3]

    # Expression blendshapes: last n_expression columns of shapedirs
    shapedirs = np.array(model["shapedirs"], dtype=np.float32)  # [5023, 3, 400]
    expr_basis = shapedirs[:, :, 300 : 300 + n_expression]  # [5023, 3, n_expr]

    # Reshape expression basis to [n_expr, 5023 * 3]
    # Each row is one expression component's vertex deltas (flattened)
    expr_basis = expr_basis.transpose(2, 0, 1)  # [n_expr, 5023, 3]
    expr_basis_flat = expr_basis.reshape(n_expression, -1)  # [n_expr, 5023*3]

    num_verts = v_template.shape[0]
    num_faces = faces.shape[0]

    print(f"Vertices: {num_verts}, Faces: {num_faces}, Expression dims: {n_expression}")
    print(f"Template bounds: [{v_template.min():.4f}, {v_template.max():.4f}]")
    print(f"Expression basis range: [{expr_basis_flat.min():.4f}, {expr_basis_flat.max():.4f}]")

    # Write binary file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<III", num_verts, num_faces, n_expression))

        # Template vertices [num_verts * 3] as float32
        f.write(v_template.flatten().astype(np.float32).tobytes())

        # Face indices [num_faces * 3] as uint32
        f.write(faces.flatten().astype(np.uint32).tobytes())

        # Expression basis [n_expr * num_verts * 3] as float32
        f.write(expr_basis_flat.astype(np.float32).tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"Exported to {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export FLAME mesh for web viewer")
    parser.add_argument("--flame_path", type=str, required=True,
                        help="Path to FLAME generic_model.pkl")
    parser.add_argument("--output", type=str, default="static/flame_data.bin",
                        help="Output binary file path")
    parser.add_argument("--n_expression", type=int, default=100,
                        help="Number of expression components to export")
    args = parser.parse_args()
    export_binary(args.flame_path, args.output, args.n_expression)


if __name__ == "__main__":
    main()
