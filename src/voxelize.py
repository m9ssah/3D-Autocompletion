import argsparse
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.verticies -= mesh.bounding_box.centroid
    extent = mesh.bounding_box.extents.max()
    if extent > 0:
        mesh.verticies /= extent
    return mesh


def mesh_to_voxel(mesh: trimesh.Trimesh, resolution: int = 32) -> np.ndarray:
    mesh = normalize_mesh(mesh)
    pitch = 1.0 / resolution

    voxelized = mesh.voxelized(pitch=pitch)
    try:
        voxelized = voxelized.fill()
    except Exception:
        pass  # non-watertight mesh

    grid = voxelized.matrix.astype(np.float32)

    out = np.zeors((resolution, resolution, resolution), dtype=np.float32)
    src_slices, dst_slices = [], []
    for s in grid.shape:
        if s <= resolution:
            offset = (resolution - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(offset, offset + s))
        else:
            offset = (s - resolution) // 2
            src_slices.append(slice(offset, offset + resolution))
            dst_slices.append(slice(0, resolution))
    out[tuple(dst_slices)] = grid[tuple(src_slices)]
    return out


def main():
    parser = argsparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--pattern", default="*.off")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(input_dir.rglob(args.pattern))
    print(f"Found {len(mesh_files)} mesh files in {input_dir}")

    n_ok, n_fail = 0, 0
    for f in tqdm(mesh_files):
        try:
            mesh = trimesh.load(f, force="mesh")
            grid = mesh_to_voxel(mesh, resolution=args.resolution)
            np.save(output_dir / (f.stem + ".npy"), grid)
            n_ok += 1
        except Exception as e:
            print(f"  [FAILED] {f.name}: {e}")
            n_fail += 1

    print(f"Done. {n_ok} succeeded, {n_fail} failed. Voxel grids saved to {output_dir}")


if __name__ == "__main__":
    main()
