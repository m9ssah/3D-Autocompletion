"""
Utility functions for rendering 3D meshes from SDF grids.

Usage:
    check main block
"""

import warnings

import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def grid_to_mesh(sdf_grid, level=0.0):
    """
    extract the zero level-set surface of an sdf grid as a triangle mesh
    returns None if the grid never crosses `level` (e.g. an all-inside or
    all-outside prediction from an undertrained model) instead of raising

    params:
    - sdf_grid: (N, N, N) signed distance grid, negative == inside
    - level: iso-value to extract the surface at
    """
    try:
        verts, faces, _normals, _values = marching_cubes(sdf_grid, level=level)
    except ValueError:
        warnings.warn(
            f"grid_to_mesh: no surface found at level={level} "
            f"(grid range [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]), returning None"
        )
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces)


def render_comparison(pred_grid, gt_grid, level=0.0, save_path=None):
    """
    renders predicted vs ground-truth reconstructions side by side

    params:
    - pred_grid: (N, N, N) predicted sdf grid
    - gt_grid: (N, N, N) ground-truth sdf grid
    - level: iso-value to extract the surface at
    - save_path: if given, writes the figure to this path instead of showing it
    """
    fig = plt.figure(figsize=(10, 5))

    for i, (grid, title) in enumerate(
        [(pred_grid, "predicted"), (gt_grid, "ground truth")]
    ):
        mesh = grid_to_mesh(grid, level=level)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        if mesh is None:
            ax.text2D(0.5, 0.5, "no surface found", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        poly = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.7)
        poly.set_facecolor((0.6, 0.7, 0.9))
        ax.add_collection3d(poly)
        ax.set_xlim(mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max())
        ax.set_ylim(mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
        ax.set_zlim(mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max())
        ax.set_title(title)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import numpy as np

    N = 20
    center = (N - 1) / 2
    zz, yy, xx = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    dist_from_center = np.sqrt(
        (xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2
    )
    radius = N / 3
    gt_grid = (dist_from_center - radius).astype(np.float32)  # sphere SDF

    rng = np.random.default_rng(0)
    pred_grid = gt_grid + rng.normal(scale=0.3, size=gt_grid.shape).astype(np.float32)

    out_dir = Path(tempfile.mkdtemp())

    mesh = grid_to_mesh(gt_grid)
    print(f"marching cubes ok: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    mesh.export(out_dir / "dummy_sphere.obj")
    print(f"exported obj to {out_dir / 'dummy_sphere.obj'}")

    png_path = out_dir / "dummy_comparison.png"
    render_comparison(pred_grid, gt_grid, save_path=png_path)
    print(f"saved comparison render to {png_path}")

    print("DEGENERATE GRID (all-positive, no zero crossing, expect None + warning):")
    all_outside_grid = np.abs(gt_grid) + 1.0  # shift entirely positive
    degenerate_mesh = grid_to_mesh(all_outside_grid)
    print("grid_to_mesh returned:", degenerate_mesh)

    degenerate_png_path = out_dir / "dummy_degenerate_comparison.png"
    render_comparison(all_outside_grid, gt_grid, save_path=degenerate_png_path)
    print(f"saved degenerate comparison render (no crash) to {degenerate_png_path}")
