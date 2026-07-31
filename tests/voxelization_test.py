"""
Usage:
    python tests/voxelization_test.py --voxel_path <path_to_voxel_grid> --output_path <path_to_output_image> [--level <level>] [--export_obj]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate.render import render_grid


def render_voxel(voxel_grid: np.ndarray, output_path: str, level: float = 0.5):
    """Render an occupancy grid, preserving the historic CLI defaults."""
    mesh = render_grid(
        voxel_grid,
        level=level,
        save_path=output_path,
        facecolor=(0.55, 0.55, 0.55),
        edgecolor=(0.1, 0.1, 0.1),
        alpha=0.9,
        bounds="grid",
        hide_axes=True,
        dpi=300,
    )
    if mesh is None:
        return None, None
    return mesh.vertices, mesh.faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--export_obj", action="store_true")
    args = parser.parse_args()

    grid = np.load(args.voxel_path)

    verts, faces = render_voxel(grid, args.output_path, level=args.level)

    if args.export_obj and verts is not None:
        import trimesh

        obj_path = args.output_path.replace(".png", ".obj")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(obj_path)


if __name__ == "__main__":
    main()
