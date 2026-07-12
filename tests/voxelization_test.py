import argparse

import numpy as np
import trimesh
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_voxel(voxel_grid: np.ndarray, output_path: str, level: float = 0.5):
    verts, faces, normals, values = marching_cubes(voxel_grid, level=level)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(verts[faces], alpha=0.9)
    poly.set_facecolor((0.55, 0.55, 0.55))
    poly.set_edgecolor((0.1, 0.1, 0.1))
    poly.set_linewidth(0.1)
    ax.add_collection3d(poly)

    ax.set_xlim(0, voxel_grid.shape[0])
    ax.set_ylim(0, voxel_grid.shape[1])
    ax.set_zlim(0, voxel_grid.shape[2])
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return verts, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--export_obj", action="store_true")
    args = parser.parse_args()

    grid = np.load(args.voxel_path)

    verts, faces = render_voxel(grid, args.output_path, level=args.level)

    if args.export_obj:
        obj_path = args.output_path.replace(".png", ".obj")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(obj_path)


if __name__ == "__main__":
    main()
