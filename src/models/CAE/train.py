from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim

from ...common.dataset import TSDF_TRUNCATION, SDFDataset
from .CAE import Conv3dAE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "cae"


def train(model, epochs, batch_size, learning_rate, train_dataset, val_dataset, device):
    torch.manual_seed(42)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon = model(batch)
            loss = geometry_aware_composite_loss(recon, batch)
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
                loss = geometry_aware_composite_loss(recon, batch)
                val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history


def geometry_aware_composite_loss(
    recon,
    target,
    truncation=TSDF_TRUNCATION,
    surface_weight=5.0,
    huber_beta=0.02,
    sign_weight=0.1,
    sign_temperature=0.02,
):
    """Optimize the TSDF values and the topology of their zero level-set.

    The weighted Huber term retains accurate distances near the surface while
    reducing the influence of larger TSDF errors.  The narrow-band sign term
    treats -recon as an inside-logit: it directly penalizes local
    inside/outside flips, which otherwise become holes or detached pieces after
    marching cubes.

    Note:
    A soft target has the correct zero-gradient at recon == target.  A hard
    occupancy label would instead push an exactly reconstructed, near-zero
    SDF farther inside or outside in pursuit of an infinite classification
    margin.
    """
    surface_band = truncation / 2
    surface_mask = target.abs() < surface_band

    huber = F.smooth_l1_loss(recon, target, beta=huber_beta, reduction="none")
    weights = 1.0 + (surface_weight - 1.0) * surface_mask.to(recon.dtype)
    distance_loss = (weights * huber).mean()

    inside_probability = torch.sigmoid(-target / sign_temperature)
    sign_loss = F.binary_cross_entropy_with_logits(
        -recon / sign_temperature, inside_probability, reduction="none"
    )
    narrow_band_sign_loss = (
        sign_loss * surface_mask.to(recon.dtype)
    ).sum() / surface_mask.sum().clamp_min(1)

    return distance_loss + sign_weight * narrow_band_sign_loss


def plot_history(history):
    path = Path(ARTIFACT_DIR / "cae_training_history_v4.png")

    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Composite Loss")
    plt.title("Training and Validation Composite Loss")
    plt.legend()
    plt.savefig(path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    train_dataset = SDFDataset(split="train")
    val_dataset = SDFDataset(split="validation")
    model = Conv3dAE(input_size=48, latent_dim=64)

    history = train(
        model,
        epochs=40,
        batch_size=8,
        learning_rate=1e-3,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )
    plot_history(history)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / "conv3d_ae_64_geometry_loss_v4.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"checkpoint saved to: {checkpoint_path}")
