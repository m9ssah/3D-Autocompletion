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
    parser.add_argument(
        "--surface_point_method",
        choices=["scan", "sample"],
        default="scan",
        help="'scan' uses virtual depth-camera renders -- needs an OpenGL context "
        "but is far more robust for non-watertight / multi-part meshes. "
        "'sample' is faster but is what produced the floating-fragment artifacts.",
    )
    parser.add_argument(
        "--sign_method",
        choices=["depth", "normal"],
        default="depth",
        help="'depth' determines inside/outside from the virtual scans directly, "
        "robust to holes and disconnected parts. 'normal' is faster but unreliable "
        "on exactly this kind of geometry.",
    )
    parser.add_argument("--scan_count", type=int, default=100)
    parser.add_argument("--scan_resolution", type=int, default=400)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(in_dir.rglob(args.pattern))
    print(f"Found {len(mesh_files)} mesh files matching {args.pattern} under {in_dir}")

    n_ok, n_fail, n_non_watertight = 0, 0, 0
    for f in tqdm(mesh_files):
        try:
            mesh = trimesh.load(f, force="mesh")
            if not mesh.is_watertight:
                n_non_watertight += 1
            sdf_grid = mesh_to_voxels(
                mesh,
                voxel_resolution=args.resolution,
                surface_point_method=args.surface_point_method,
                sign_method=args.sign_method,
                scan_count=args.scan_count,
                scan_resolution=args.scan_resolution,
            )
            np.save(out_dir / (f.stem + ".npy"), sdf_grid.astype(np.float32))
            n_ok += 1
        except Exception as e:
            print(f"  [FAILED] {f.name}: {e}")
            n_fail += 1

    print(f"Done. {n_ok} succeeded, {n_fail} failed. SDF grids saved to {out_dir}")
    if n_ok + n_fail > 0:
        print(
            f"{n_non_watertight}/{n_ok + n_fail} meshes were non-watertight -- if that's most of "
            f"them, it explains the earlier artifacts and is worth mentioning in your report's "
            f"SDF-upgrade section."
        )


if __name__ == "__main__":
    main()
