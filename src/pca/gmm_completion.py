import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.gmm import _gaussian_logpdf 
from src.pca.completion import closed_form_complete 


def gmm_weighted_complete(partial_grid, mask, pca, gmm, lam=0.0, reg_covar=1e-6):
    """
    solves once per component, then scores and blends the candidates instead of 
    picking a component up front

    params:
    - partial_grid, mask, pca: same as closed_form_complete
    - gmm: a fitted src.common.gmm.GaussianMixtureEM
    - lam: same as before, 0 = no prior
    - reg_covar: jitter added before inverting each component's covariance

    returns (completed_grid, z, blend_weights)
    """
    x = partial_grid.reshape(-1).astype(np.float64)
    visible = mask.reshape(-1).astype(bool)
    A = pca.components_[:, visible].T
    b = x[visible] - pca.mean_[visible]

    candidates = []
    scores = []
    for k in range(gmm.n_components):
        _recon_k, z_k = closed_form_complete(
            partial_grid,
            mask,
            pca,
            lam=lam,
            prior_mean=gmm.means_[k],
            prior_cov=gmm.covariances_[k],
            reg_covar=reg_covar,
        )
        candidates.append(z_k)

        data_fit = np.sum((A @ z_k - b) ** 2)
        # lower score = more plausible candidate
        log_prior = np.log(gmm.weights_[k]) + _gaussian_logpdf(
            z_k[None, :], gmm.means_[k], gmm.covariances_[k], reg_covar
        )[0]
        scores.append(data_fit - 2 * lam * log_prior)

    scores = np.array(scores)
    blend_weights = softmax(-scores / 2)
    z = np.sum(blend_weights[:, None] * np.array(candidates), axis=0)

    completed = pca.mean_ + z @ pca.components_
    return completed.reshape(partial_grid.shape).astype(np.float32), z, blend_weights


if __name__ == "__main__":
    from src.common.gmm import GaussianMixtureEM 
    from src.pca.pca import PCA 

    rng = np.random.default_rng(0)
    true_rank, d = 4, 300

    basis, _ = np.linalg.qr(rng.standard_normal((d, true_rank)))  # orthonormal, so PCA's axes stay aligned with ours
    basis = basis.T
    mean_shape = rng.standard_normal(d)

    n_per_cluster = 150
    codes_a = np.array([5.0, 0.0, 0.0, 0.0]) + rng.standard_normal((n_per_cluster, true_rank)) * 0.5
    codes_b = np.array([-5.0, 0.0, 0.0, 0.0]) + rng.standard_normal((n_per_cluster, true_rank)) * 0.5
    codes = np.vstack([codes_a, codes_b])
    X = mean_shape + codes @ basis

    pca = PCA(n_components=true_rank)
    pca.fit(X)
    z_train = pca.transform(X)

    single_gmm = GaussianMixtureEM(n_components=1, n_init=1, random_state=0).fit(z_train)
    mixture_gmm = GaussianMixtureEM(n_components=2, n_init=5, random_state=0).fit(z_train)

    x_true = X[0]
    z_true = pca.transform(X[:1])[0]

    print("WELL-DETERMINED CASE (80% visible, expect both priors to do fine):")
    mask = (rng.random(d) < 0.8).astype(np.uint8)
    partial = np.where(mask == 1, x_true, 0.0)
    _r1, z1 = closed_form_complete(
        partial, mask, pca, lam=5.0, prior_mean=single_gmm.means_[0], prior_cov=single_gmm.covariances_[0]
    )
    _r2, z2, _w2 = gmm_weighted_complete(partial, mask, pca, mixture_gmm, lam=5.0)
    print("single-gaussian z error:", np.abs(z1 - z_true).max())
    print("gmm-weighted z error:", np.abs(z2 - z_true).max())

    print("\nAMBIGUOUS CASE (only reveal voxels that barely depend on the style-defining dim):") 
    influence = np.abs(basis[0])
    visible_idx = np.argsort(influence)[: d // 3]
    mask_amb = np.zeros(d, dtype=np.uint8)
    mask_amb[visible_idx] = 1
    partial_amb = np.where(mask_amb == 1, x_true, 0.0)

    _r3, z3 = closed_form_complete(
        partial_amb, mask_amb, pca, lam=5.0, prior_mean=single_gmm.means_[0], prior_cov=single_gmm.covariances_[0]
    )
    _r4, z4, blend_weights = gmm_weighted_complete(partial_amb, mask_amb, pca, mixture_gmm, lam=5.0)

    a_component = np.argmin([np.linalg.norm(mixture_gmm.means_[k] - z_true) for k in range(mixture_gmm.n_components)])

    print("single-gaussian (blind to the two styles) z error:", np.abs(z3 - z_true).max())
    print("gmm-weighted z error:", np.abs(z4 - z_true).max())
    print(f"blend weight on the correct-style component (index {a_component}):", blend_weights[a_component])
