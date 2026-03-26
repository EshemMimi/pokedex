"""Build artifacts/manifest.csv from folder-per-class image roots (Gen 1).

Example (after placing Kaggle extract under data/raw/kaggle/):

  python scripts/build_manifest.py --data-root data/raw/kaggle

Repeat --data-root for multiple sources.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pokedex.ingest import build_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build train/val/test manifest for Gen 1 images.")
    p.add_argument(
        "--data-root",
        action="append",
        dest="data_roots",
        metavar="DIR",
        help="Directory whose *subfolders* are class names (or dex numbers). Repeat for multiple datasets.",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "artifacts" / "label_map.json",
        help="Path to label_map.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "manifest.csv",
        help="Output CSV path",
    )
    p.add_argument("--train", type=float, default=0.7, help="Train fraction")
    p.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    p.add_argument("--test", type=float, default=0.15, help="Test fraction")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for splits")
    args = p.parse_args()
    if not args.data_roots:
        p.error("Pass at least one --data-root DIR (folder containing per-species subfolders).")

    roots = [Path(r).resolve() for r in args.data_roots]
    warnings, stats = build_manifest(
        label_map_path=args.label_map.resolve(),
        data_roots=roots,
        out_csv=args.out.resolve(),
        project_root=ROOT,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    print(f"Wrote {args.out} ({stats['images']} rows)")
    print(f"Classes with at least one image: {stats['classes_with_samples']}/151")
    print(f"Exact duplicates dropped (SHA-256): {stats['dedupe_dropped']}")
    if warnings:
        print("\nWarnings:")
        for w in warnings[:50]:
            print(f"  - {w}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more")


if __name__ == "__main__":
    main()
