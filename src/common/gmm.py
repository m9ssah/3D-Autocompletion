import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp


def _gaussian_logpdf(X, mean, cov, reg_covar):
    """
    log density of a single multivariate gaussian at each row of X, via cholesky

    params:
    - X: (n, d) points
    - mean: (d,) component mean
    - cov: (d, d) component covariance
    - reg_covar: added to the diagonal of cov for numerical stability
    """
    d = X.shape[1]
    cov_reg = cov + reg_covar * np.eye(d)
    L = np.linalg.cholesky(cov_reg)

    diff = (X - mean).T  # (d, n)
    y = solve_triangular(L, diff, lower=True)  # (d, n), y = L^-1 (x - mean)
    squared_maha = np.sum(y**2, axis=0)  # (n,)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    return -0.5 * (d * np.log(2 * np.pi) + log_det + squared_maha)


class GaussianMixtureEM:
    """
    from-scratch gaussian mixture model fit via expectation-maximization
    n_components == 1 reduces to a single full-covariance gaussian (MLE)

    params (init):
    - n_components: number of mixture components K
    - max_iter: max EM iterations per restart
    - tol: a restart stops early once log-likelihood improves by less than this
    - reg_covar: added to the diagonal of every covariance for numerical stability
    - n_init: number of random restarts, keeps the one with the best log-likelihood
    - random_state: seed for reproducibility

    after fit(), the fitted params are in weights_, means_, covariances_,
    and log_likelihood_history_ holds the winning restart's trace (for
    checking that EM increased the log-likelihood every iteration)
    """

    def __init__(
        self,
        n_components,
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-6,
        n_init=5,
        random_state=None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_history_ = None

    def _init_means(self, X, rng):
        # kmeans++-style seeding: spread initial means out instead of picking
        # them uniformly at random, which materially reduces how often EM
        # gets stuck in a bad local optimum
        n_samples = X.shape[0]
        first = rng.integers(n_samples)
        means = [X[first]]

        for _ in range(1, self.n_components):
            sq_dists = np.min([np.sum((X - m) ** 2, axis=1) for m in means], axis=0)
            probs = sq_dists / sq_dists.sum()
            next_idx = rng.choice(n_samples, p=probs)
            means.append(X[next_idx])

        return np.array(means)

    def _e_step(self, X, weights, means, covariances):
        log_prob_matrix = np.stack(
            [
                np.log(weights[k])
                + _gaussian_logpdf(X, means[k], covariances[k], self.reg_covar)
                for k in range(self.n_components)
            ],
            axis=1,
        )
        log_norm = logsumexp(log_prob_matrix, axis=1)  # (n,) per-sample log p(x)
        responsibilities = np.exp(log_prob_matrix - log_norm[:, None])
        return log_norm, responsibilities

    def _fit_single(self, X, rng):
        n_samples, d = X.shape

        means = self._init_means(X, rng)
        data_cov = np.cov(X.T, bias=True).reshape(d, d)
        covariances = np.array([data_cov.copy() for _ in range(self.n_components)])
        weights = np.full(self.n_components, 1.0 / self.n_components)

        history = []
        for iteration in range(self.max_iter):
            log_norm, responsibilities = self._e_step(X, weights, means, covariances)
            log_likelihood = log_norm.sum()
            history.append(log_likelihood)

            if iteration > 0 and log_likelihood - history[-2] < self.tol:
                break  # weights/means/covariances already match this log-likelihood

            Nk = responsibilities.sum(axis=0)
            for k in range(self.n_components):
                if Nk[k] < 1e-6:  # component collapsed -- reinitialize it
                    means[k] = X[rng.integers(n_samples)]
                    covariances[k] = data_cov.copy()
                    weights[k] = 1.0 / n_samples
                    continue

                weights[k] = Nk[k] / n_samples
                means[k] = (responsibilities[:, k, None] * X).sum(axis=0) / Nk[k]
                diff = X - means[k]
                covariances[k] = (
                    np.einsum("n,ni,nj->ij", responsibilities[:, k], diff, diff) / Nk[k]
                )
            weights /= weights.sum()  # renormalize in case some components collapsed

        log_norm, _ = self._e_step(X, weights, means, covariances)
        history.append(
            log_norm.sum()
        )  # so that the final element matches the returned params

        return weights, means, covariances, history

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        best_log_likelihood = -np.inf
        best_params, best_history = None, None

        for _ in range(self.n_init):
            weights, means, covariances, history = self._fit_single(X, rng)
            if history[-1] > best_log_likelihood:
                best_log_likelihood = history[-1]
                best_params = (weights, means, covariances)
                best_history = history

        self.weights_, self.means_, self.covariances_ = best_params
        self.log_likelihood_history_ = best_history
        return self

    def score_samples(self, X):
        """per-sample log p(x) under the fitted mixture -- the MAP prior term"""
        X = np.asarray(X, dtype=np.float64)
        log_norm, _ = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return log_norm

    def predict_proba(self, X):
        """per-sample, per-component responsibilities under the fitted mixture"""
        X = np.asarray(X, dtype=np.float64)
        _, responsibilities = self._e_step(
            X, self.weights_, self.means_, self.covariances_
        )
        return responsibilities

    def sample(self, n_samples, random_state=None):
        """draws n_samples points from the fitted mixture"""
        rng = np.random.default_rng(
            random_state if random_state is not None else self.random_state
        )
        d = self.means_.shape[1]
        component_choices = rng.choice(
            self.n_components, size=n_samples, p=self.weights_
        )

        samples = np.empty((n_samples, d))
        for k in range(self.n_components):
            idx = component_choices == k
            n_k = idx.sum()
            if n_k == 0:
                continue
            cov_reg = self.covariances_[k] + self.reg_covar * np.eye(d)
            L = np.linalg.cholesky(cov_reg)
            z = rng.standard_normal((n_k, d))
            samples[idx] = self.means_[k] + z @ L.T

        return samples


if __name__ == "__main__":
    import itertools

    rng = np.random.default_rng(0)

    true_weights = np.array([0.5, 0.3, 0.2])
    true_means = np.array([[0.0, 0.0], [5.0, 5.0], [-4.0, 4.0]])
    true_covariances = np.array(
        [
            [[1.0, 0.3], [0.3, 1.0]],
            [[0.5, -0.2], [-0.2, 0.5]],
            [[1.5, 0.0], [0.0, 0.3]],
        ]
    )

    n_samples = 2000
    component_choices = rng.choice(3, size=n_samples, p=true_weights)
    X = np.empty((n_samples, 2))
    for k in range(3):
        idx = component_choices == k
        X[idx] = rng.multivariate_normal(
            true_means[k], true_covariances[k], size=idx.sum()
        )

    print("FITTING K=3 GMM ON SYNTHETIC DATA (ground truth known):")
    gmm = GaussianMixtureEM(n_components=3, n_init=5, random_state=42)
    gmm.fit(X)

    history = np.array(gmm.log_likelihood_history_)
    n_decreases = (np.diff(history) < -1e-8).sum()
    print(
        f"log-likelihood trace length: {len(history)}, decreases: {n_decreases} (expect 0)"
    )

    # match recovered components to true ones by nearest mean (EM doesn't preserve ordering)
    best_perm, best_cost = None, np.inf
    for perm in itertools.permutations(range(3)):
        cost = sum(np.sum((gmm.means_[perm[k]] - true_means[k]) ** 2) for k in range(3))
        if cost < best_cost:
            best_cost, best_perm = cost, perm

    for k in range(3):
        j = best_perm[k]
        print(f"component {k}:")
        print(f"  true weight={true_weights[k]:.3f}   recovered={gmm.weights_[j]:.3f}")
        print(f"  true mean={true_means[k]}   recovered={np.round(gmm.means_[j], 3)}")
        print(
            f"  true cov=\n{true_covariances[k]}\n  recovered=\n{np.round(gmm.covariances_[j], 3)}"
        )

    print(
        "\nFITTING K=1 (single-gaussian special case) AND CHECKING AGAINST DIRECT MLE:"
    )
    single = GaussianMixtureEM(n_components=1, n_init=1, random_state=0)
    single.fit(X)
    direct_mean = X.mean(axis=0)
    direct_cov = np.cov(
        X.T, bias=True
    )  # bias=True: EM's M-step is the population (MLE) covariance
    print("mean diff (expect ~0):", np.abs(single.means_[0] - direct_mean).max())
    print("cov diff (expect ~0):", np.abs(single.covariances_[0] - direct_cov).max())

    print("\nsample() sanity check -- drawing 5 points from the fitted K=3 mixture:")
    print(gmm.sample(5, random_state=1))

    print(
        "\nscore_samples() / predict_proba() sanity check on the first 3 training points:"
    )
    print("log p(x):", gmm.score_samples(X[:3]))
    print("responsibilities:\n", np.round(gmm.predict_proba(X[:3]), 3))
