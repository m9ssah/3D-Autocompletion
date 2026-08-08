import numpy as np
from pathlib import Path


def apply_masking(
    grid,
    mask_type="halfspace",
    axis=2,
    seed=None,
    min_block_surface_fraction=0.10,
    surface_band=0.05,
    max_block_attempts=64,
):
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
    - min_block_surface_fraction: fraction of near-surface voxels that a block
      must hide, preventing a block from landing only in empty space
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
        surface_positions = np.argwhere(np.abs(grid) < surface_band)
        if len(surface_positions) == 0:
            raise ValueError("block masking requires at least one near-surface voxel")

        best_candidate, best_fraction = None, -1.0
        for _ in range(max_block_attempts):
            size = rng.integers(N // 4, N // 2)
            anchor = surface_positions[rng.integers(len(surface_positions))]
            starts = []
            for coordinate in anchor:
                low = max(0, coordinate - size + 1)
                high = min(coordinate, N - size)
                starts.append(rng.integers(low, high + 1))
            starts = np.asarray(starts)

            hidden_surface = np.all(
                (surface_positions >= starts) & (surface_positions < starts + size),
                axis=1,
            )
            hidden_fraction = hidden_surface.mean()
            if hidden_fraction > best_fraction:
                best_candidate = (starts, size)
                best_fraction = hidden_fraction
            if hidden_fraction >= min_block_surface_fraction:
                break
        else:
            raise RuntimeError(
                "could not sample a block hiding the requested surface fraction; "
                f"best was {best_fraction:.3f}"
            )

        starts, size = best_candidate
        mask[
            starts[0] : starts[0] + size,
            starts[1] : starts[1] + size,
            starts[2] : starts[2] + size,
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
    shape_path = Path("../../ModelNet40/sdf_conversion/monitor/test/monitor_0481.npy")
    grid = np.load(shape_path)
    print("block test:")
    partial_grid, mask = apply_masking(grid, mask_type="block", seed=42)
    print(f"grid shape={grid.shape}, visible voxels: {mask.sum()} / {mask.size}")

    out_path = Path("../../artifacts/masks/monitor_0002_partial_block.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, partial_grid)
    print(f"saved masked block partial grid to {out_path}")

    print("halfspace test:")
    partial_grid, mask = apply_masking(grid, mask_type="halfspace", axis=2, seed=42)
    print(f"grid shape={grid.shape}, visible voxels: {mask.sum()} / {mask.size}")

    out_path = Path("../../artifacts/masks/monitor_0002_partial_halfspace.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, partial_grid)
    print(f"saved masked halfspace partial grid to {out_path}")
