import argparse
from pathlib import Path

import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing mesh files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to save .npy SDF grids"
    )
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument(
        "--pattern", default="*.off", help="Glob pattern, e.g. *.off or *.obj"
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(in_dir.rglob(args.pattern))
    print(f"Found {len(mesh_files)} mesh files matching {args.pattern} under {in_dir}")

    n_ok, n_fail = 0, 0
    for f in tqdm(mesh_files):
        try:
            mesh = trimesh.load(f, force="mesh")
            sdf_grid = mesh_to_voxels(
                mesh,
                voxel_resolution=args.resolution,
                surface_point_method="sample",
                sign_method="normal",
            )
            np.save(out_dir / (f.stem + ".npy"), sdf_grid.astype(np.float32))
            n_ok += 1
        except Exception as e:
            print(f"  [FAILED] {f.name}: {e}")
            n_fail += 1

    print(f"Done. {n_ok} succeeded, {n_fail} failed. SDF grids saved to {out_dir}")


if __name__ == "__main__":
    main()
