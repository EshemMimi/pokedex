"""Train a Gen 1 Pokémon classifier from artifacts/manifest.csv (transfer learning)."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestImageDataset(Dataset):
    """Loads images listed in manifest CSV (columns: path, dex, class_index, split)."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        split: str,
        transform: transforms.Compose,
    ) -> None:
        self.project_root = project_root.resolve()
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                rel = Path(row["path"])
                path = self.project_root / rel
                y = int(row["class_index"])
                self.samples.append((path, y))
        if not self.samples:
            raise ValueError(
                f"No rows with split={split!r} in {manifest_path}. "
                "Run scripts/build_manifest.py on your image folders first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(num_classes: int, backbone: str) -> nn.Module:
    if backbone != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone}")
    weights = ResNet18_Weights.IMAGENET1K_V1
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    n = 0
    correct1 = 0.0
    correct3 = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, pred = logits.topk(3, 1, True, True)
        pred = pred.t()
        ok = pred.eq(y.view(1, -1).expand_as(pred))
        correct1 += ok[:1].any(dim=0).float().sum().item()
        correct3 += ok[:3].any(dim=0).float().sum().item()
        n += y.size(0)
    if n == 0:
        return 0.0, 0.0
    return correct1 / n, correct3 / n


def run_training(
    *,
    project_root: Path,
    manifest_path: Path,
    label_map_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
    image_size: int,
    backbone: str,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm = json.loads(label_map_path.read_text(encoding="utf-8"))
    num_classes = int(lm["num_classes"])

    train_ds = ManifestImageDataset(
        project_root,
        manifest_path,
        "train",
        build_transforms(image_size, train=True),
    )
    val_ds = ManifestImageDataset(
        project_root,
        manifest_path,
        "val",
        build_transforms(image_size, train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes, backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_top1 = -1.0
    best_path = out_dir / "best_model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "version": 1,
        "backbone": backbone,
        "num_classes": num_classes,
        "image_size": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "label_map_path": "artifacts/label_map.json",
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2), encoding="utf-8"
    )

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v1, v3 = evaluate(model, val_loader, device)
        scheduler.step()
        history.append(
            {"epoch": float(epoch), "train_loss": t_train, "val_top1": v1, "val_top3": v3}
        )
        if v1 > best_top1:
            best_top1 = v1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_top1": v1,
                    "val_top3": v3,
                    "backbone": backbone,
                    "num_classes": num_classes,
                    "image_size": image_size,
                },
                best_path,
            )
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={t_train:.4f}  val_top1={v1*100:.2f}%  val_top3={v3*100:.2f}%"
        )

    elapsed = time.perf_counter() - t0
    summary = {
        "best_val_top1": best_top1,
        "epochs_ran": epochs,
        "seconds": elapsed,
        "checkpoint": str(best_path),
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    (out_dir / "training_history.json").write_text(
        json.dumps({"history": history, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone in {elapsed:.1f}s. Best val Top-1: {best_top1*100:.2f}%  Saved: {best_path}")
    return summary
