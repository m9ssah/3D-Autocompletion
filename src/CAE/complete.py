"""Latent-space TSDF completion with a frozen Conv3dAE decoder and Gaussian prior.

Example:
    python -m CAE.complete --mask-type halfspace --lambda-prior 0.1
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common.dataset import TSDF_TRUNCATION, SDFDataset, truncate_sdf
from common.gmm import GaussianMixtureEM
from common.masking import apply_masking
from evaluate import masked_iou, masked_sdf_error

from .CAE import Conv3dAE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "cae" / "conv3d_ae_64_geometry_loss.pt"
)
DEFAULT_TRAIN_LATENTS = (
    PROJECT_ROOT
    / "artifacts"
    / "cae"
    / "latents"
    / "conv3d_ae_64_geometry_loss_train_latents.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cae" / "completions"


def fit_single_gaussian(latent_codes, reg_covar=1e-6):
    """Fit the single-Gaussian latent prior using train codes only."""
    codes = np.asarray(latent_codes, dtype=np.float64)
    if codes.ndim != 2 or len(codes) < 2:
        raise ValueError("latent_codes must have shape (n_samples >= 2, latent_dim)")

    return GaussianMixtureEM(
        n_components=1, n_init=1, reg_covar=reg_covar, random_state=42
    ).fit(codes)


def complete(
    model,
    partial_grid,
    mask,
    gmm,
    device,
    lambda_prior=1e-5,
    steps=500,
    learning_rate=1e-2,
    truncation=TSDF_TRUNCATION,
):
    """
    Optimize a latent code to fit observed voxels while staying plausible.

    mask == 1 denotes observed voxels. The ground truth in the masked
    region is never used by the optimization; it is only needed later for
    evaluation.
    """
    partial_grid = np.asarray(partial_grid, dtype=np.float32)
    mask = np.asarray(mask)
    if partial_grid.ndim != 3 or partial_grid.shape != mask.shape:
        raise ValueError("partial_grid and mask must be same-shape 3D arrays")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be binary: 1 observed, 0 occluded")
    if gmm.n_components != 1 or gmm.means_ is None or gmm.covariances_ is None:
        raise ValueError("gmm must be a fitted GaussianMixtureEM(n_components=1)")
    if lambda_prior < 0 or steps <= 0 or learning_rate <= 0:
        raise ValueError(
            "lambda_prior must be non-negative; steps and learning_rate positive"
        )

    model = model.to(device)
    model.eval()
    parameter_states = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():  # freeze ae params for completion optimization
        parameter.requires_grad_(False)

    try:
        partial_tsdf = truncate_sdf(partial_grid, truncation)
        target = torch.from_numpy(partial_tsdf).to(device)[
            None, None
        ]  # [1, 1, D, H, W]
        observed = torch.from_numpy(mask.astype(bool)).to(device)[
            None, None
        ]  # [1, 1, D, H, W]

        with torch.no_grad():  # encode the partial TSDF to get an initial latent code while keeping the model frozen
            z_initial = model.encode(target)
        z = (
            z_initial.detach().clone().requires_grad_(True)
        )  # new starting point for optimization, requires gradient for optimization

        mean = torch.as_tensor(gmm.means_[0], dtype=z.dtype, device=device)
        covariance = torch.as_tensor(gmm.covariances_[0], dtype=z.dtype, device=device)
        covariance = covariance + gmm.reg_covar * torch.eye(
            len(mean), device=device
        )  # add regularization to ensure positive-definite covar (for cholesky)
        covariance_cholesky = torch.linalg.cholesky(
            covariance
        )  # cholesky decomposition for mahalanobis distance computation

        optimizer = torch.optim.Adam([z], lr=learning_rate)
        history = {"total_loss": [], "observed_mse": [], "prior_energy": []}
        for _ in range(steps):
            optimizer.zero_grad()
            decoded = model.decode(z)
            observed_mse = ((decoded - target)[observed] ** 2).mean()

            difference = (z - mean).unsqueeze(-1)  # [1, latent_dim, 1]
            mahalanobis = torch.cholesky_solve(difference, covariance_cholesky)
            prior_energy = 0.5 * (difference.transpose(1, 2) @ mahalanobis).squeeze()
            prior_energy = (
                prior_energy / z.shape[1]
            )  # normalize by latent dimension for comparability

            total_loss = observed_mse + lambda_prior * prior_energy
            total_loss.backward()
            optimizer.step()

            history["total_loss"].append(float(total_loss.detach().cpu()))
            history["observed_mse"].append(float(observed_mse.detach().cpu()))
            history["prior_energy"].append(float(prior_energy.detach().cpu()))

        with torch.no_grad():
            completion = torch.clamp(model.decode(z), -truncation, truncation)

        return (
            completion.squeeze().cpu().numpy(),
            z_initial.squeeze().cpu().numpy(),
            z.detach().squeeze().cpu().numpy(),
            {
                name: np.asarray(values, dtype=np.float32)
                for name, values in history.items()
            },
        )
    finally:
        for parameter, requires_grad in zip(model.parameters(), parameter_states):
            parameter.requires_grad_(requires_grad)


def load_model(checkpoint_path, device, latent_dim=64):
    model = Conv3dAE(input_size=48, latent_dim=latent_dim)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-latents", type=Path, default=DEFAULT_TRAIN_LATENTS)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument(
        "--mask-type", choices=("halfspace", "block"), default="halfspace"
    )
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--lambda-prior", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    train_latents = args.train_latents.resolve()
    if not train_latents.is_file():
        raise FileNotFoundError(f"train latent file not found: {train_latents}")

    test_dataset = SDFDataset(split="test")
    ground_truth = test_dataset[args.test_index].squeeze(0).numpy()
    partial, mask = apply_masking(
        ground_truth,
        mask_type=args.mask_type,
        axis=args.axis,
        seed=args.mask_seed,
    )

    with np.load(train_latents) as data:
        gmm = fit_single_gaussian(data["codes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completion, z_initial, z_optimized, history = complete(
        load_model(checkpoint, device),
        partial,
        mask,
        gmm,
        device,
        lambda_prior=args.lambda_prior,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )

    shape_id = test_dataset.files[args.test_index].stem
    metrics = {
        "masked_iou": masked_iou(completion, ground_truth, mask),
        "masked_sdf_l1": masked_sdf_error(completion, ground_truth, mask, norm="l1"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{shape_id}_{args.mask_type}_completion.npz"
    np.savez_compressed(
        output_path,
        completion=completion,
        partial=partial,
        mask=mask,
        z_initial=z_initial,
        z_optimized=z_optimized,
        **history,
        **metrics,
    )
    print(f"completed {shape_id} on {device}")
    print(f"masked IoU: {metrics['masked_iou']:.6f}")
    print(f"masked SDF L1: {metrics['masked_sdf_l1']:.6f}")
    print(f"saved: {output_path}")
