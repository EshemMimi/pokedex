"""Tests for pokedex/training.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from pokedex.training import (
    ManifestImageDataset,
    build_model,
    build_transforms,
    evaluate,
    train_one_epoch,
)


def test_build_transforms_val_shape() -> None:
    tf = build_transforms(224, train=False)
    img = Image.new("RGB", (400, 300), color=(128, 64, 32))
    t = tf(img)
    assert t.shape == (3, 224, 224)


def test_build_model_resnet18_head(project_root: Path, label_map_path: Path) -> None:
    """Small integration: ResNet-18 with ImageNet weights (may download once)."""
    lm = json.loads(label_map_path.read_text(encoding="utf-8"))
    n = int(lm["num_classes"])
    m = build_model(n, "resnet18")
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, n)


def test_manifest_dataset_reads_csv(
    label_map_path: Path, project_root: Path, tmp_path: Path
) -> None:
    img_dir = tmp_path / "imgs" / "a"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "x.png"
    Image.new("RGB", (32, 32), color="red").save(img_path)
    manifest = tmp_path / "m.csv"
    rel = img_path.relative_to(tmp_path)
    rel_s = str(rel).replace("\\", "/")
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "dex", "class_index", "split"])
        w.writerow([rel_s, 1, 0, "train"])
    tf = build_transforms(224, train=False)
    ds = ManifestImageDataset(tmp_path, manifest, "train", tf)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (3, 224, 224)
    assert y == 0


def test_evaluate_perfect_linear_model() -> None:
    """evaluate() Top-1 / Top-3 with a trivial model."""

    class Fixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3 * 4 * 4, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.size(0)
            logits = torch.zeros(b, 5, device=x.device)
            logits[:, 2] = 100.0
            return logits

    device = torch.device("cpu")
    model = Fixed().to(device)
    x = torch.randn(4, 3, 4, 4)
    y = torch.tensor([2, 2, 2, 2])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    t1, t3 = evaluate(model, loader, device)
    assert t1 == 1.0
    assert t3 == 1.0


def test_train_one_epoch_runs() -> None:
    model = nn.Linear(2, 3)
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.tensor([0, 1])),
        batch_size=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loss = train_one_epoch(model, loader, loss_fn, opt, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss >= 0.0
