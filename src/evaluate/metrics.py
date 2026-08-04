"""
Utility functions for evaluating SDF predictions against ground-truth SDFs.
"""

import warnings

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def _check_shapes(pred_grid, gt_grid, mask):
    if pred_grid.shape != gt_grid.shape or pred_grid.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: pred_grid {pred_grid.shape}, gt_grid {gt_grid.shape}, "
            f"mask {mask.shape} must all match"
        )


def _iou(pred_inside, gt_inside):
    """Return binary IoU, treating two empty sets as a perfect match."""
    intersection = np.logical_and(pred_inside, gt_inside).sum()
    union = np.logical_or(pred_inside, gt_inside).sum()
    iou = 1.0 if union == 0 else intersection / union
    return float(iou)


def masked_iou(pred_grid, gt_grid, mask, threshold=0.0):
    """
    intersection-over-union between pred and gt, restricted to the occluded region
    mask == 1 marks observed voxels (per src/common/masking.py), so we score on mask == 0
    returns nan if mask marks nothing as occluded -- there's nothing to evaluate

    params:
    - pred_grid: (N, N, N) predicted SDF grid
    - gt_grid: (N, N, N) ground-truth SDF grid
    - mask: (N, N, N) binary array, 1 == observed, 0 == occluded
    - threshold: SDF value separating inside (< threshold) from outside
    """
    _check_shapes(pred_grid, gt_grid, mask)

    occluded = mask == 0
    if not occluded.any():
        warnings.warn("masked_iou: mask marks nothing as occluded, returning nan")
        return np.nan

    pred_inside = (pred_grid < threshold) & occluded
    gt_inside = (gt_grid < threshold) & occluded

    return _iou(pred_inside, gt_inside)[0]


def masked_sdf_error(pred_grid, gt_grid, mask, norm="l1"):
    """
    mean SDF error between pred and gt, restricted to the occluded region (mask == 0)
    returns nan if mask marks nothing as occluded -- there's nothing to evaluate

    params:
    - pred_grid: (N, N, N) predicted SDF grid
    - gt_grid: (N, N, N) ground-truth SDF grid
    - mask: (N, N, N) binary array, 1 == observed, 0 == occluded
    - norm: "l1" (mean absolute error) or "l2" (root mean squared error)
    """
    _check_shapes(pred_grid, gt_grid, mask)

    occluded = mask == 0
    if not occluded.any():
        warnings.warn("masked_sdf_error: mask marks nothing as occluded, returning nan")
        return np.nan

    diff = pred_grid[occluded] - gt_grid[occluded]

    if norm == "l1":
        return np.abs(diff).mean()
    elif norm == "l2":
        return np.sqrt((diff**2).mean())
    else:
        raise ValueError(f"Invalid norm {norm}, must be 'l1' or 'l2'")


def mesh_topology_metrics(sdf_grid):
    """
    Return marching-cubes component statistics for one SDF grid.

    A high component count or a small largest-component fraction flags floating
    fragments, which voxel error alone does not capture.
    """
    try:
        vertices, faces, _normals, _values = marching_cubes(sdf_grid, level=0.0)
    except ValueError:
        return {
            "mesh_vertices": 0,
            "mesh_faces": 0,
            "mesh_components": 0,
            "largest_component_face_fraction": 0.0,
        }

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    component_faces = [len(component.faces) for component in components]
    return {
        "mesh_vertices": len(vertices),
        "mesh_faces": len(faces),
        "mesh_components": len(components),
        "largest_component_face_fraction": float(max(component_faces) / len(faces)),
    }


def reconstruction_metrics(pred_grid, gt_grid, surface_band=0.05):
    """Evaluate a full SDF reconstruction, including zero-level-set topology."""
    if pred_grid.shape != gt_grid.shape:
        raise ValueError(
            f"shape mismatch: pred_grid {pred_grid.shape}, gt_grid {gt_grid.shape}"
        )
    if surface_band <= 0:
        raise ValueError("surface_band must be positive")

    error = pred_grid - gt_grid
    pred_inside = pred_grid < 0
    gt_inside = gt_grid < 0
    iou = _iou(pred_inside, gt_inside)
    surface = np.abs(gt_grid) < surface_band

    metrics = {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "iou": iou,
        "sign_accuracy": float((pred_inside == gt_inside).mean()),
        "surface_mae": (
            float(np.abs(error[surface]).mean()) if surface.any() else np.nan
        ),
        "surface_sign_accuracy": (
            float((pred_inside[surface] == gt_inside[surface]).mean())
            if surface.any()
            else np.nan
        ),
    }
    metrics.update(mesh_topology_metrics(pred_grid))
    return metrics


if __name__ == "__main__":
    N = 20
    center = (N - 1) / 2
    zz, yy, xx = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    dist_from_center = np.sqrt(
        (xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2
    )
    radius = N / 3
    gt_grid = (dist_from_center - radius).astype(
        np.float32
    )  # sphere SDF: negative inside

    rng = np.random.default_rng(42)
    mask = np.ones_like(gt_grid, dtype=np.uint8)
    mask[:, :, : N // 2] = 0  # occlude one half

    print("IDENTICAL PREDICTION:")
    pred_grid = gt_grid.copy()
    print("iou:", masked_iou(pred_grid, gt_grid, mask))
    print("sdf_error (l1):", masked_sdf_error(pred_grid, gt_grid, mask, norm="l1"))

    print("NOISY PREDICTION:")
    noisy_grid = gt_grid + rng.normal(scale=1.0, size=gt_grid.shape).astype(np.float32)
    print("iou:", masked_iou(noisy_grid, gt_grid, mask))
    print("sdf_error (l1):", masked_sdf_error(noisy_grid, gt_grid, mask, norm="l1"))

    print("FULLY-OBSERVED MASK (no occluded region, expect nan + warning):")
    full_mask = np.ones_like(gt_grid, dtype=np.uint8)
    print("iou:", masked_iou(noisy_grid, gt_grid, full_mask))
    print(
        "sdf_error (l1):", masked_sdf_error(noisy_grid, gt_grid, full_mask, norm="l1")
    )

    print("MISMATCHED SHAPES (expect ValueError):")
    try:
        masked_iou(noisy_grid, gt_grid, full_mask[:-1])
    except ValueError as e:
        print(f"  raised as expected: {e}")
