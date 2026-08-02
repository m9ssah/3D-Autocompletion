from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SDFDataset(Dataset):
    def __init__(self, split="train"):
        root = Path(__file__).resolve().parents[3]
        self.files = sorted(
            (root / "ModelNet40" / "sdf_conversion" / "monitor" / split).glob("*.npy")
        )
        if not self.files:
            raise FileNotFoundError(f"No .npy files found for split: {split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sdf = np.load(self.files[index]).astype(np.float32)
        return torch.from_numpy(sdf).unsqueeze(0)


train_dataset = SDFDataset("train")
val_dataset = SDFDataset("validation")
test_dataset = SDFDataset("test")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
