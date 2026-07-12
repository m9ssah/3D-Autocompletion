import argparse
from pathlib import Path

import trimesh
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.off")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(in_dir.rglob(args.pattern))
    print(f"Found {len(mesh_files)} files matching {args.pattern} under {in_dir}")

    n_ok, n_fail = 0, 0
    for f in tqdm(mesh_files):
        try:
            mesh = trimesh.load(f, force="mesh")
            mesh.export(out_dir / (f.stem + ".obj"))
            n_ok += 1
        except Exception as e:
            print(f"  [FAILED] {f.name}: {e}")
            n_fail += 1

    print(f"Done. {n_ok} succeeded, {n_fail} failed. .obj files saved to {out_dir}")


if __name__ == "__main__":
    main()
