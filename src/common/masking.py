import numpy as np
from pathlib import Path


def apply_masking(grid, mask_type="halfspace", axis=2, seed=None):
    """
    returns (partial_grid, mask)
    mask == 1 marks observed voxels

    occluded voxels are set to the grid's own max value (not 0) because 0 is a valid SDF
    value and we want to avoid introducing new values into the grid

    params:
    - grid: (N, N, N) numpy array
    - mask_type: "halfspace" (single-view occlusion) or "block" (random contiguous occlusion)
    - axis: axis along which to apply halfspace masking
    - seed: random seed for reproducibility
    """
    assert (
        grid.ndim == 3 and grid.shape[0] == grid.shape[1] == grid.shape[2]
    ), "expected a cubic (N, N, N) grid"

    rng = np.random.default_rng(seed)
    N = grid.shape[0]
    mask = np.ones_like(grid, dtype=np.uint8)

    if mask_type == "halfspace":
        split = rng.integers(N // 4, 3 * N // 4)

        if axis == 0:
            mask[:split, :, :] = 0
        elif axis == 1:
            mask[:, :split, :] = 0
        elif axis == 2:
            mask[:, :, :split] = 0
        else:
            raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")

    elif mask_type == "block":
        size = rng.integers(N // 4, N // 2)
        start_x = rng.integers(0, N - size)
        start_y = rng.integers(0, N - size)
        start_z = rng.integers(0, N - size)
        mask[
            start_x : start_x + size, start_y : start_y + size, start_z : start_z + size
        ] = 0

    else:
        raise ValueError(
            f"Invalid mask_type '{mask_type}', must be 'halfspace' or 'block'"
        )

    fill_value = grid.max()
    partial_grid = np.where(mask == 1, grid, fill_value).astype(grid.dtype)
    return partial_grid, mask


if __name__ == "__main__":
    N = 48
    shape_path = Path("../../ModelNet40/sdf_conversion/monitor/train/monitor_0001.npy")
    grid = np.load(shape_path)
    print("block test:")
    partial_grid, mask = apply_masking(grid, mask_type="block", seed=42)
    print(f"grid shape={grid.shape}, visible voxels: {mask.sum()} / {mask.size}")

    out_path = Path("../../artifacts/masks/monitor_0001_partial_block.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, partial_grid)
    print(f"saved masked block partial grid to {out_path}")

    print("halfspace test:")
    partial_grid, mask = apply_masking(grid, mask_type="halfspace", axis=1, seed=42)
    print(f"grid shape={grid.shape}, visible voxels: {mask.sum()} / {mask.size}")

    out_path = Path("../../artifacts/masks/monitor_0001_partial_halfspace.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, partial_grid)
    print(f"saved masked halfspace partial grid to {out_path}")
