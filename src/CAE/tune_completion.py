"""Evaluate Gaussian-prior latent completion on a chosen dataset split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.dataset import TSDF_TRUNCATION, SDFDataset
from common.masking import apply_masking
from evaluate import ResultRecorder, masked_iou, masked_sdf_error

from .complete import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TRAIN_LATENTS,
    complete,
    fit_single_gaussian,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cae" / "completion_tuning"


def decode_partial_baseline(model, partial_grid, device):
    """Decode the encoded partial TSDF without latent optimization."""
    partial_tensor = torch.from_numpy(partial_grid).to(device)[None, None]
    with torch.no_grad():
        reconstruction = torch.clamp(
            model(partial_tensor), -TSDF_TRUNCATION, TSDF_TRUNCATION
        )
    return reconstruction.squeeze().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-latents", type=Path, default=DEFAULT_TRAIN_LATENTS)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 0.00001, 0.0001, 0.001]
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--mask-type", choices=("halfspace", "block"), default="halfspace"
    )
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.learning_rate <= 0
        or any(value < 0 for value in args.lambdas)
    ):
        raise ValueError(
            "steps and learning-rate must be positive; lambdas non-negative"
        )

    checkpoint = args.checkpoint.resolve()
    train_latents = args.train_latents.resolve()
    if not checkpoint.is_file() or not train_latents.is_file():
        raise FileNotFoundError("checkpoint or train latent file not found")

    dataset = SDFDataset(split=args.split)
    num_shapes = len(dataset) if args.max_shapes is None else args.max_shapes
    if not 0 < num_shapes <= len(dataset):
        raise ValueError(f"max-shapes must be in [1, {len(dataset)}]")

    with np.load(train_latents) as data:
        gmm = fit_single_gaussian(data["codes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    recorder = ResultRecorder()

    for index in range(num_shapes):
        ground_truth = dataset[index].squeeze(0).numpy()
        partial, mask = apply_masking(
            ground_truth, args.mask_type, axis=args.axis, seed=args.mask_seed + index
        )
        shape_id = dataset.files[index].stem

        baseline = decode_partial_baseline(model, partial, device)
        recorder.add(
            shape_id=shape_id,
            method="encoded_partial",
            lambda_prior=np.nan,
            masked_iou=masked_iou(baseline, ground_truth, mask),
            masked_sdf_l1=masked_sdf_error(baseline, ground_truth, mask),
        )

        for lambda_prior in args.lambdas:
            completion, _z_initial, _z_optimized, _history = complete(
                model,
                partial,
                mask,
                gmm,
                device,
                lambda_prior=lambda_prior,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
            recorder.add(
                shape_id=shape_id,
                method="latent_optimized",
                lambda_prior=lambda_prior,
                masked_iou=masked_iou(completion, ground_truth, mask),
                masked_sdf_l1=masked_sdf_error(completion, ground_truth, mask),
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.split}_{args.mask_type}_per_shape.csv"
    recorder.save(csv_path)
    summaries = []
    configurations = [("encoded_partial", np.nan)] + [
        ("latent_optimized", value) for value in args.lambdas
    ]
    for method, lambda_prior in configurations:
        rows = [
            row
            for row in recorder.rows
            if row["method"] == method
            and (np.isnan(lambda_prior) or row["lambda_prior"] == lambda_prior)
        ]
        summaries.append(
            {
                "method": method,
                "lambda_prior": None if np.isnan(lambda_prior) else lambda_prior,
                "mean_masked_iou": float(np.mean([row["masked_iou"] for row in rows])),
                "mean_masked_sdf_l1": float(
                    np.mean([row["masked_sdf_l1"] for row in rows])
                ),
            }
        )
    summary_path = args.output_dir / f"{args.split}_{args.mask_type}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"evaluated {num_shapes} {args.split} shapes using {args.mask_type} masks")
    for summary in summaries:
        print(summary)
    print(f"saved: {csv_path}")
