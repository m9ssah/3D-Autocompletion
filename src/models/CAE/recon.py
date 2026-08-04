from pathlib import Path

import numpy as np
import torch

from common.dataset import SDFDataset, truncate_sdf

from .CAE import Conv3dAE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "cae" / "conv3d_ae_20_signweight_1.pt"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cae" / "renders"


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conv3dAE(input_size=48, latent_dim=64)
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Run models.CAE.train first."
        )
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.to(device)

    test_dataset = SDFDataset(split="test")
    original = test_dataset[0].squeeze(0).numpy()  # (1, D, D, D) -> (D, D, D)

    reconstruction = reconstruct(model, original, device)

    print(f"original range: [{original.min():.4f}, {original.max():.4f}]")
    print(
        f"reconstruction range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]"
    )
    assert not np.isnan(
        reconstruction
    ).any(), "reconstruction contains NaNs, training likely diverged"

    mse = np.mean((reconstruction - original) ** 2)
    print(f"full reconstruction MSE: {mse}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "test_conv3d_ae_20_signweight_1.npy", reconstruction)
    print(f"saved both grids to {OUTPUT_DIR}")
