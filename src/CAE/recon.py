import argparse
from pathlib import Path

import numpy as np
import torch

from common.dataset import SDFDataset, truncate_sdf

from .CAE import Conv3dAE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "cae" / "conv3d_ae_128_geometry_soft_w005_40ep.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "cae" / "reconstructions" / "expand_CAE"
)


def reconstruct(model, grid, device):
    """
    grid: (D, D, D) numpy array, a single SDF grid
    returns: reconstruction as a (D, D, D) numpy array
    """
    if grid.ndim != 3:
        raise ValueError(f"Expected a 3D SDF grid, got shape {grid.shape}")

    model.eval()
    tsdf = truncate_sdf(np.asarray(grid, dtype=np.float32))
    x = torch.from_numpy(tsdf).unsqueeze(0).unsqueeze(0)
    x = x.to(device)  # (1, 1, D, D, D)

    with torch.no_grad():
        recon = model(x)

    return truncate_sdf(recon.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct one or more test TSDFs.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    test_dataset = SDFDataset(split="test")
    indices = list(dict.fromkeys(args.test_indices))
    invalid = [index for index in indices if not 0 <= index < len(test_dataset)]
    if invalid:
        raise IndexError(
            f"test indices must be in [0, {len(test_dataset) - 1}]: {invalid}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv3dAE(input_size=48, latent_dim=args.latent_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in indices:
        original = test_dataset[index].squeeze(0).numpy()
        reconstruction = reconstruct(model, original, device)
        if np.isnan(reconstruction).any():
            raise RuntimeError("reconstruction contains NaNs; training likely diverged")

        shape_id = test_dataset.files[index].stem
        mse = float(np.mean((reconstruction - original) ** 2))
        output_path = args.output_dir / f"{shape_id}_reconstruction.npz"
        np.savez_compressed(
            output_path,
            original=original,
            reconstruction=reconstruction,
            mse=mse,
            checkpoint=np.asarray(str(checkpoint)),
        )
        print(f"{shape_id}: MSE={mse:.6f} -> {output_path}")
