"""Run a reproducible controlled sign-loss experiment for the Conv3dAE."""

import argparse

import torch

from common.dataset import SDFDataset

from .CAE import Conv3dAE
from .train import ARTIFACT_DIR, plot_history, train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sign-target", choices=("soft", "hard"), required=True)
    parser.add_argument("--sign-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conv3dAE(input_size=48, latent_dim=64)
    history = train(
        model,
        epochs=args.epochs,
        batch_size=8,
        learning_rate=1e-3,
        train_dataset=SDFDataset(split="train"),
        val_dataset=SDFDataset(split="validation"),
        device=device,
        loss_kwargs={"sign_target": args.sign_target, "sign_weight": args.sign_weight},
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / f"{args.run_name}.pt"
    history_path = ARTIFACT_DIR / f"{args.run_name}_history.png"
    torch.save(model.state_dict(), checkpoint_path)
    plot_history(history, history_path)
    print(f"checkpoint: {checkpoint_path}")
    print(f"history: {history_path}")
