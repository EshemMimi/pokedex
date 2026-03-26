"""Train Gen 1 classifier from artifacts/manifest.csv.

Requires PyTorch + torchvision. Example:

  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  python scripts/train.py

Use your GPU/CUDA install from https://pytorch.org/get-started/locally/ when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pokedex.training import run_training  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train Pokémon Gen 1 image classifier.")
    p.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    p.add_argument("--label-map", type=Path, default=ROOT / "artifacts" / "label_map.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backbone", type=str, default="resnet18", choices=("resnet18",))
    args = p.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(
            f"Missing {args.manifest}. Build it with: python scripts/build_manifest.py --data-root data/raw/<your_dataset>"
        )

    run_training(
        project_root=ROOT,
        manifest_path=args.manifest.resolve(),
        label_map_path=args.label_map.resolve(),
        out_dir=args.out_dir.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        backbone=args.backbone,
    )


if __name__ == "__main__":
    main()
