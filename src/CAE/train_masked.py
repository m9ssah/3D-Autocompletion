"""
Train a denoising CAE to reconstruct complete TSDFs from masked inputs.

Each training example is a ``(partial_tsdf, full_tsdf)`` pair.  The partial
input is regenerated every epoch, using an even mix of halfspace and block
masks, while the target is always the unmodified full TSDF.
"""

import argparse
from pathlib import Path

import matplotlib
import torch
from torch import optim
from torch.utils.data import Dataset

from common.dataset import SDFDataset
from common.masking import apply_masking

from .CAE import Conv3dAE
from .train import geometry_aware_composite_loss, plot_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cae" / "masked_autoencoder"


class MaskedSDFDataset(Dataset):
    """Create deterministic, epoch-varying partial/full TSDF training pairs."""

    def __init__(self, dataset, seed, training):
        self.dataset = dataset
        self.seed = seed
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        target = self.dataset[index]
        # validation stays fixed
        # training masks change every epoch without making results depend on DataLoader worker timing
        epoch_offset = self.epoch * len(self) if self.training else 0
        mask_seed = self.seed + epoch_offset + index
        mask_type = "halfspace" if (self.epoch + index) % 2 == 0 else "block"
        partial, _mask = apply_masking(
            target.squeeze(0).numpy(),
            mask_type=mask_type,
            axis=2,
            seed=mask_seed,
        )
        return torch.from_numpy(partial).unsqueeze(0), target


def train_masked(
    model,
    epochs,
    batch_size,
    learning_rate,
    train_dataset,
    val_dataset,
    device,
    loss_kwargs=None,
):
    model.to(device)
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for partial, target in train_loader:
            partial, target = partial.to(device), target.to(device)
            optimizer.zero_grad()
            loss = geometry_aware_composite_loss(model(partial), target, **loss_kwargs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * target.size(0)
        history["train_loss"].append(train_loss / len(train_dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for partial, target in val_loader:
                partial, target = partial.to(device), target.to(device)
                loss = geometry_aware_composite_loss(
                    model(partial), target, **loss_kwargs
                )
                val_loss += loss.item() * target.size(0)
        history["val_loss"].append(val_loss / len(val_dataset))
        print(
            f"epoch {epoch + 1:03d}/{epochs}: "
            f"train={history['train_loss'][-1]:.6f}, val={history['val_loss'][-1]:.6f}"
        )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.latent_dim) <= 0
        or args.learning_rate <= 0
    ):
        raise ValueError(
            "epochs, batch-size, latent-dim, and learning-rate must be positive"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MaskedSDFDataset(SDFDataset("train"), args.seed, training=True)
    val_dataset = MaskedSDFDataset(SDFDataset("validation"), args.seed, training=False)
    model = Conv3dAE(input_size=48, latent_dim=args.latent_dim)
    history = train_masked(
        model,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        train_dataset,
        val_dataset,
        device,
        loss_kwargs={"sign_target": "soft", "sign_weight": 0.05},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "conv3d_ae_masked_mixed_40ep.pt"
    torch.save(model.state_dict(), checkpoint)
    plot_history(history, args.output_dir / "history.png")
    print(f"checkpoint saved to: {checkpoint}")
