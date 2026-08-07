import numpy as np
from scipy.linalg import svd


class PCA:
    """
    PCA model for flattened SDF grids.

    Uses SVD after subtracting the training mean. This avoids building the full
    feature covariance matrix, which would be too large for 48x48x48 grids.

    params:
    - n_components: number of PCA components to keep
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        _U, S, Vt = svd(Xc, full_matrices=False)

        k = self.n_components
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]

        total_variance = np.sum(S ** 2)
        self.explained_variance_ratio_ = (S[:k] ** 2) / total_variance
        return self

    def transform(self, X):
        """turn grid vectors into k-dim latent codes"""
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        """turn latent codes back into grid vectors"""
        Z = np.asarray(Z, dtype=np.float64)
        return self.mean_ + Z @ self.components_

    def reconstruct(self, X):
        """encode then decode in one step, so we can check reconstruction error"""
        return self.inverse_transform(self.transform(X))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_samples, true_rank, d = 200, 5, 50

    basis = rng.standard_normal((true_rank, d))
    mean = rng.standard_normal(d) * 3
    codes = rng.standard_normal((n_samples, true_rank))
    X = mean + codes @ basis

    print("FITTING PCA AT k == true rank (expect ~exact reconstruction):")
    pca = PCA(n_components=true_rank)
    pca.fit(X)
    recon = pca.reconstruct(X)
    print(f"max abs reconstruction error: {np.abs(X - recon).max():.2e} (expect ~1e-9)")
    print(f"explained variance ratio sums to: {pca.explained_variance_ratio_.sum():.6f} (expect ~1.0)")

    print("\nFITTING PCA AT k < true rank (expect noticeably nonzero error):")
    pca_under = PCA(n_components=2)
    pca_under.fit(X)
    recon_under = pca_under.reconstruct(X)
    print(f"mean abs reconstruction error: {np.abs(X - recon_under).mean():.4f} (expect clearly > 0)")

    print("\nCHECKING mean_ MATCHES DIRECT X.mean(axis=0):")
    print("max diff (expect ~0):", np.abs(pca.mean_ - X.mean(axis=0)).max())

    print("\nCHECKING RECONSTRUCTION ERROR DECREASES MONOTONICALLY AS k GROWS:")
    errors = []
    for k in [1, 2, 3, 4, 5]:
        p = PCA(n_components=k).fit(X)
        errors.append(np.abs(X - p.reconstruct(X)).mean())
        print(f"  k={k}: mean abs error={errors[-1]:.4f}")
    n_increases = sum(1 for a, b in zip(errors, errors[1:]) if b > a + 1e-9)
    print(f"increases in error as k grows: {n_increases} (expect 0)")
