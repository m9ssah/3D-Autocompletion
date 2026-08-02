import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.gmm import GaussianMixtureEM 


def fit_single_gaussian_prior(latent_codes):
    """fits a single gaussian on the latent codes, just reusing our GMM class with k=1"""
    gmm = GaussianMixtureEM(n_components=1, n_init=1)
    gmm.fit(latent_codes)
    return gmm.means_[0], gmm.covariances_[0]


def closed_form_complete(partial_grid, mask, pca, lam=0.0, prior_mean=None, prior_cov=None, reg_covar=1e-6):
    """
    solves for the latent code that best explains the visible voxels, then
    decodes it back into a full grid

    params:
    - partial_grid: grid with the occluded voxels filled in (see src/common/masking.py)
    - mask: same shape as partial_grid, 1 = observed, 0 = occluded
    - pca: a fitted src.pca.pca.PCA instance
    - lam: how much to trust the prior. 0 = don't use it at all
    - prior_mean, prior_cov: only needed if lam > 0
    - reg_covar: tiny jitter added to prior_cov before inverting it. same idea as
      GaussianMixtureEM's reg_covar, just applied here too since this function might
      get called with a covariance from a GMM component that wasn't fit on much data

    returns (completed_grid, z)
    """
    grid_shape = partial_grid.shape
    x = partial_grid.reshape(-1).astype(np.float64)
    visible = mask.reshape(-1).astype(bool)

    A = pca.components_[:, visible].T
    b = x[visible] - pca.mean_[visible]

    if lam == 0:
        z, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        if prior_mean is None or prior_cov is None:
            raise ValueError("need prior_mean and prior_cov when lam > 0")
        prior_cov_reg = prior_cov + reg_covar * np.eye(prior_cov.shape[0])
        precision = np.linalg.inv(prior_cov_reg)
        lhs = A.T @ A + lam * precision
        rhs = A.T @ b + lam * precision @ prior_mean
        z = np.linalg.solve(lhs, rhs)

    completed = pca.mean_ + z @ pca.components_
    return completed.reshape(grid_shape).astype(np.float32), z


if __name__ == "__main__":
    from src.pca.pca import PCA 

    rng = np.random.default_rng(0)
    n_samples, true_rank, d = 200, 5, 400

    basis = rng.standard_normal((true_rank, d))
    mean_shape = rng.standard_normal(d)
    codes = rng.standard_normal((n_samples, true_rank))
    X = mean_shape + codes @ basis

    pca = PCA(n_components=true_rank)
    pca.fit(X)

    z_true = pca.transform(X[:1])[0]
    x_true = X[0]

    print("WELL-DETERMINED CASE (80% visible, lam=0, expect near-exact recovery):")
    mask = (rng.random(d) < 0.8).astype(np.uint8)
    partial = np.where(mask == 1, x_true, 0.0)
    recon, z_hat = closed_form_complete(partial, mask, pca, lam=0.0)
    print("z error:", np.abs(z_hat - z_true).max())
    print("recon error on the hidden voxels:", np.abs(recon[mask == 0] - x_true[mask == 0]).max())

    print("\nUNDER-DETERMINED CASE (only 3 voxels visible, fewer than k=5):")
    mask_sparse = np.zeros(d, dtype=np.uint8)
    mask_sparse[:3] = 1
    partial_sparse = np.where(mask_sparse == 1, x_true, 0.0)

    _recon_noprior, z_noprior = closed_form_complete(partial_sparse, mask_sparse, pca, lam=0.0)
    print("lam=0, z error (expect large - not enough info to pin down z):", np.abs(z_noprior - z_true).max())

    prior_mean, prior_cov = z_true, np.eye(true_rank) * 0.01
    _recon_prior, z_prior = closed_form_complete(
        partial_sparse, mask_sparse, pca, lam=10.0, prior_mean=prior_mean, prior_cov=prior_cov
    )
    print("lam=10, z error (expect much smaller - prior pulls it back):", np.abs(z_prior - z_true).max())
