import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from .CAE import Conv3dAE
from common.dataset import SDFDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cae"


def train(model, epochs, batch_size, learning_rate, train_dataset, val_dataset, device):
    torch.manual_seed(42)
    model.to(device)

    criterion_val = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon = model(batch)
            loss = weighted_mse_loss(recon, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion_val(recon, batch)
                val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history


def weighted_mse_loss(recon, target, truncation=0.1, surface_weight=5.0):
    """
    MSE that upweights voxels near the SDF zero-crossing (the actual surface).
    Far-field SDF values are geometrically uninteresting but dominate a naive
    MSE by sheer voxel count -- this is what causes thin structures (the
    monitor's arm) to get neglected relative to large flat regions (the screen).
    """
    weight = torch.where(target.abs() < truncation, surface_weight, 1.0)
    return (weight * (recon - target) ** 2).mean()


def plot_history(history):
    path = Path(ARTIFACT_DIR / "cae_training_history.png")

    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    train_dataset = SDFDataset(split="train")
    val_dataset = SDFDataset(split="validation")
    model = Conv3dAE(input_size=48, latent_dim=32)

    history = train(
        model,
        epochs=100,
        batch_size=8,
        learning_rate=1e-3,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )
    plot_history(history)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / "conv3d_ae_v2.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"checkpoint saved to: {checkpoint_path}")
