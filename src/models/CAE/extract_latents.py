"""
Extract aligned latent codes for the train, validation, and test splits.

Usage:
    python -m models.CAE.extract_latents
    python -m models.CAE.extract_latents --checkpoint artifacts/cae/conv3d_ae_64_geometry_loss.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common.dataset import SDFDataset

from .CAE import Conv3dAE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "cae" / "conv3d_ae_64_geometry_loss.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cae" / "latents"


def extract_split_latents(model, dataset, batch_size, device):
    """Return latent vectors in the same sorted order as ``dataset.files``."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_batches = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            latent_batches.append(model.encode(batch.to(device)).cpu().numpy())

    codes = np.concatenate(latent_batches, axis=0).astype(np.float32, copy=False)
    shape_ids = np.asarray([path.stem for path in dataset.files], dtype=str)

    if len(codes) != len(shape_ids):
        raise RuntimeError("latent-code count does not match dataset file count")
    
    return codes, shape_ids


def extract_latents(checkpoint_path, latent_dim, batch_size, device, output_dir):
    model = Conv3dAE(input_size=48, latent_dim=latent_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for split in ("train", "validation", "test"):
        dataset = SDFDataset(split=split)
        codes, shape_ids = extract_split_latents(model, dataset, batch_size, device)
        path = output_dir / f"{checkpoint_path.stem}_{split}_latents.npz"
        np.savez_compressed(
            path,
            codes=codes,
            shape_ids=shape_ids,
            split=np.asarray(split),
            checkpoint=np.asarray(str(checkpoint_path)),
        )
        output_paths[split] = path
    return output_paths


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_paths = extract_latents(
        checkpoint, args.latent_dim, args.batch_size, device, args.output_dir
    )

    for split, path in output_paths.items():
        with np.load(path) as data:
            print(f"{split}: {data['codes'].shape} -> {path}")
