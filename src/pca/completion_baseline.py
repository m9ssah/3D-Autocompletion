import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.masking import apply_masking 
from src.evaluate import ResultRecorder, Timer, masked_iou, masked_sdf_error 
from src.pca.completion import closed_form_complete, fit_single_gaussian_prior  
from src.pca.pca import PCA 

DATA_DIR = REPO_ROOT / "ModelNet40" / "sdf_conversion" / "monitor"
RESULTS_PATH = REPO_ROOT / "artifacts" / "results" / "pca_completion_baseline.csv"
K = 64  # best among the k values tested in the reconstruction baseline
LAM_VALUES = [0.0, 1.0]
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

    pca = PCA(n_components=K)
    pca.fit(X_train)

    z_train = pca.transform(X_train)
    prior_mean, prior_cov = fit_single_gaussian_prior(z_train)

    recorder = ResultRecorder()

    for mask_type in MASK_TYPES:
        for lam in LAM_VALUES:
            ious, sdf_errors, solve_times = [], [], []

            for idx, (shape_id, gt_grid) in enumerate(zip(val_ids, val_grids)):
                partial_grid, mask = apply_masking(gt_grid, mask_type=mask_type, seed=idx)

                with Timer() as t:
                    recon, _z = closed_form_complete(
                        partial_grid, mask, pca, lam=lam, prior_mean=prior_mean, prior_cov=prior_cov
                    )

                iou = masked_iou(recon, gt_grid, mask)
                sdf_error = masked_sdf_error(recon, gt_grid, mask, norm="l1")

                ious.append(iou)
                sdf_errors.append(sdf_error)
                solve_times.append(t.elapsed)

                recorder.add(
                    shape_id=shape_id,
                    method="pca_completion",
                    mask_type=mask_type,
                    lam=lam,
                    k=K,
                    iou=iou,
                    sdf_error=sdf_error,
                    solve_time_s=t.elapsed,
                )

            print(
                f"mask={mask_type:<10} lam={lam:<4} "
                f"iou={np.mean(ious):.4f}  sdf_error={np.mean(sdf_errors):.5f}  "
                f"solve_time={np.mean(solve_times) * 1000:.3f}ms"
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    recorder.save(RESULTS_PATH)
    print(f"saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
