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
    """Load the FLAME model from a pickle file.

    Requires chumpy to be installed (the FLAME pickle contains chumpy.Ch objects).
    """
    with open(flame_path, "rb") as f:
        model = pickle.load(f, encoding="latin1")

    # Convert chumpy arrays to numpy
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
