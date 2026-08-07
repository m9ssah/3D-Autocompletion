"""Overfit a fixed small TSDF subset to test CAE reconstruction capacity.

This deliberately uses the same examples for training and validation.  It is a
sanity check, not a generalization experiment: a healthy autoencoder should
nearly memorize this subset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from common.dataset import SDFDataset, TSDF_TRUNCATION
from evaluate import ResultRecorder, reconstruction_metrics

from .CAE import Conv3dAE
from .train import plot_history, train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cae" / "overfit_8"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shapes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_shapes <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("num-shapes, epochs, and learning-rate must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train_dataset = SDFDataset(split="train")
    if args.num_shapes > len(full_train_dataset):
        raise ValueError(f"num-shapes must be at most {len(full_train_dataset)}")
    indices = list(range(args.num_shapes))
    subset = Subset(full_train_dataset, indices)

    model = Conv3dAE(input_size=48, latent_dim=64)
    history = train(
        model,
        epochs=args.epochs,
        batch_size=args.num_shapes,
        learning_rate=args.learning_rate,
        train_dataset=subset,
        val_dataset=subset,
        device=device,
        loss_kwargs={"sign_target": "soft", "sign_weight": 0.05},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    plot_history(history, args.output_dir / "history.png")

    originals, reconstructions, shape_ids = [], [], []
    recorder = ResultRecorder()
    model.eval()
    with torch.no_grad():
        for index in indices:
            original = full_train_dataset[index].squeeze(0).numpy()
            reconstruction = torch.clamp(
                model(full_train_dataset[index][None].to(device)),
                min=-TSDF_TRUNCATION,
                max=TSDF_TRUNCATION,
            ).squeeze().cpu().numpy()
            shape_id = full_train_dataset.files[index].stem
            metrics = reconstruction_metrics(reconstruction, original)
            recorder.add(shape_id=shape_id, **metrics)
            originals.append(original)
            reconstructions.append(reconstruction)
            shape_ids.append(shape_id)

    recorder.save(args.output_dir / "per_shape_metrics.csv")
    np.savez_compressed(
        args.output_dir / "reconstructions.npz",
        originals=np.asarray(originals),
        reconstructions=np.asarray(reconstructions),
        shape_ids=np.asarray(shape_ids, dtype=str),
    )
    summary = {
        key: float(np.nanmean([row[key] for row in recorder.rows]))
        for key in recorder.rows[0]
        if key != "shape_id"
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"overfit {args.num_shapes} train shapes on {device}")
    print(f"mean IoU: {summary['iou']:.6f}")
    print(f"mean MSE: {summary['mse']:.6f}")
    print(f"mean components: {summary['mesh_components']:.6f}")
    print(f"artifacts: {args.output_dir}")
