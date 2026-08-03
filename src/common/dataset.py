from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TSDF_TRUNCATION = 0.1


def truncate_sdf(sdf, truncation=TSDF_TRUNCATION):
    """Convert a signed-distance grid to a truncated SDF (TSDF)."""
    if truncation <= 0:
        raise ValueError("truncation must be positive")
    return np.clip(sdf, -truncation, truncation)


class SDFDataset(Dataset):
    def __init__(self, split="train", truncation=TSDF_TRUNCATION):
        root = Path(__file__).resolve().parents[2]
        self.files = sorted(
            (root / "ModelNet40" / "sdf_conversion" / "monitor" / split).glob("*.npy")
        )
        if not self.files:
            raise FileNotFoundError(f"No .npy files found for split: {split}")
        self.truncation = truncation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sdf = np.load(self.files[index]).astype(np.float32)
        sdf = truncate_sdf(sdf, self.truncation)
        return torch.from_numpy(sdf).unsqueeze(0)
