import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.gmm import GaussianMixtureEM 
from src.common.masking import apply_masking 
from src.evaluate import ResultRecorder, Timer, masked_iou, masked_sdf_error
from src.pca.gmm_completion import gmm_weighted_complete 
from src.pca.pca import PCA 

DATA_DIR = REPO_ROOT / "ModelNet40" / "sdf_conversion" / "monitor"
RESULTS_PATH = REPO_ROOT / "artifacts" / "results" / "gmm_completion_baseline.csv"
DIAGNOSTICS_PATH = REPO_ROOT / "artifacts" / "results" / "gmm_completion_diagnostics.csv"
K_PCA = 64  # same latent size as the completion baseline
K_GMM_VALUES = [1, 2, 3, 5]
LAM = 1.0  # lam=0 would make every k_gmm identical, since there'd be no prior to tell components apart
MASK_TYPES = ["halfspace", "block"]


def load_split(split):
    files = sorted((DATA_DIR / split).glob("*.npy"))
    grids = np.stack([np.load(f) for f in files]).astype(np.float32)
    shape_ids = [f.stem for f in files]
    return grids, shape_ids


def main():
    train_grids, _ = load_split("train")
    val_grids, val_ids = load_split("validation")

    X_train = train_grids.reshape(len(train_grids), -1)
    X_val = val_grids.reshape(len(val_grids), -1)

    pca = PCA(n_components=K_PCA)
    pca.fit(X_train)
    z_train = pca.transform(X_train)
    z_val = pca.transform(X_val)

    recorder = ResultRecorder()
    diagnostics = ResultRecorder()

    for k_gmm in K_GMM_VALUES:
        gmm = GaussianMixtureEM(n_components=k_gmm, n_init=5, random_state=0)
        gmm.fit(z_train)

        # train vs val log-lik tells us if more components generalize or just overfit
        train_ll = gmm.score_samples(z_train).mean()
        val_ll = gmm.score_samples(z_val).mean()
        print(
            f"k_gmm={k_gmm:<3} train_log_lik={train_ll:.2f}  val_log_lik={val_ll:.2f}  "
            f"weights={np.round(gmm.weights_, 3)}"
        )
        for component, weight in enumerate(gmm.weights_):
            diagnostics.add(
                k_gmm=k_gmm, component=component, weight=weight, train_log_lik=train_ll, val_log_lik=val_ll
            )

        for mask_type in MASK_TYPES:
            ious, sdf_errors, solve_times = [], [], []

            for idx, (shape_id, gt_grid) in enumerate(zip(val_ids, val_grids)):
                partial_grid, mask = apply_masking(gt_grid, mask_type=mask_type, seed=idx)

                with Timer() as t:
                    recon, _z, _blend = gmm_weighted_complete(partial_grid, mask, pca, gmm, lam=LAM)

                iou = masked_iou(recon, gt_grid, mask)
                sdf_error = masked_sdf_error(recon, gt_grid, mask, norm="l1")

                ious.append(iou)
                sdf_errors.append(sdf_error)
                solve_times.append(t.elapsed)

                recorder.add(
                    shape_id=shape_id,
                    method="gmm_completion",
                    mask_type=mask_type,
                    k_gmm=k_gmm,
                    lam=LAM,
                    k_pca=K_PCA,
                    iou=iou,
                    sdf_error=sdf_error,
                    solve_time_s=t.elapsed,
                )

            print(
                f"k_gmm={k_gmm:<3} mask={mask_type:<10} "
                f"iou={np.mean(ious):.4f}  sdf_error={np.mean(sdf_errors):.5f}  "
                f"solve_time={np.mean(solve_times) * 1000:.3f}ms"
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    recorder.save(RESULTS_PATH)
    print(f"saved results to {RESULTS_PATH}")

    diagnostics.save(DIAGNOSTICS_PATH)
    print(f"saved gmm diagnostics (train/val log-lik, weights per component) to {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
