import numpy as np


def apply_masking(grid, mask_type="halfspace", axis=2, seed=None):
    """
    returns (partial_grid, mask)
    mask == 1 marks observed voxels

    params:
    - grid: (N, N, N) numpy array
    - mask_type: "halfspace" (single-view occlusion) or "grid" (random contiguous occlusion)
    - axis: axis along which to apply halfspace masking
    - seed: random seed for reproducibility
    """

    if seed is not None:
        np.random.default_rng(seed)

    if mask_type == "halfspace":  # randomly choose a halfspace along the given axis
        N = grid.shape[0]
        split = np.random.randint(1, N)
        mask = np.ones_like(grid, dtype=np.uint8)

        if axis == 0:
            mask[:split, :, :] = 0
        elif axis == 1:
            mask[:, :split, :] = 0
        elif axis == 2:
            mask[:, :, :split] = 0
        else:
            raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")

    elif mask_type == "grid":  # randomly choose a contiguous subgrid to mask out
        N = grid.shape[0]
        size = np.random.randint(N // 4, N // 2)  # size of the masked region
        start_1 = np.random.randint(0, N - size)
        start_2 = np.random.randint(0, N - size)
        start_3 = np.random.randint(0, N - size)
        mask = np.ones_like(grid, dtype=np.uint8)

        mask[
            start_1 : start_1 + size, start_2 : start_2 + size, start_3 : start_3 + size
        ] = 0

    return grid * mask, mask


if __name__ == "__main__":
    N = 48
    grid = np.random.rand(N, N, N)
    partial_grid, mask = apply_masking(grid, mask_type="halfspace", axis=0, seed=42)

    print(grid)
    print(partial_grid)

    print("PRACTICAL TEST:")

    # test using converted shapes
    grid = np.load("../../ModelNet40/sdf_conversion/monitor/train/monitor_0001.npy")
    partial_grid, mask = apply_masking(grid, mask_type="grid", seed=42)
    print(grid)
    print(partial_grid)
