import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import ResultRecorder, masked_iou, masked_sdf_error 
from src.pca.pca import PCA 

DATA_DIR = REPO_ROOT / "ModelNet40" / "sdf_conversion" / "monitor"
RESULTS_PATH = REPO_ROOT / "artifacts" / "results" / "pca_reconstruction_baseline.csv"
K_VALUES = [16, 32, 64]


def load_split(split):
    files = sorted((DATA_DIR / split).glob("*.npy"))
    grids = np.stack([np.load(f) for f in files]).astype(np.float32)
    shape_ids = [f.stem for f in files]
    return grids, shape_ids


def main():
    train_grids, _ = load_split("train")
    val_grids, val_ids = load_split("validation")
    grid_shape = train_grids.shape[1:]

    print(f"train: {len(train_grids)} shapes, validation: {len(val_grids)} shapes, grid shape: {grid_shape}")

    X_train = train_grids.reshape(len(train_grids), -1)
    X_val = val_grids.reshape(len(val_grids), -1)

    full_mask = np.zeros(grid_shape, dtype=np.uint8) 
    recorder = ResultRecorder()

    for k in K_VALUES:
        pca = PCA(n_components=k)
        pca.fit(X_train)

        recon_val = pca.reconstruct(X_val).reshape(len(val_grids), *grid_shape).astype(np.float32)

        ious, sdf_errors = [], []
        for shape_id, pred_grid, gt_grid in zip(val_ids, recon_val, val_grids):
            iou = masked_iou(pred_grid, gt_grid, full_mask)
            sdf_error = masked_sdf_error(pred_grid, gt_grid, full_mask, norm="l1")
            ious.append(iou)
            sdf_errors.append(sdf_error)
            recorder.add(shape_id=shape_id, method="pca_reconstruction", k=k, iou=iou, sdf_error=sdf_error)

        print(
            f"k={k:>3}: explained_variance={pca.explained_variance_ratio_.sum():.4f}  "
            f"val_iou={np.mean(ious):.4f}  val_sdf_error={np.mean(sdf_errors):.5f}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    recorder.save(RESULTS_PATH)
    print(f"saved per-shape, per-k results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
