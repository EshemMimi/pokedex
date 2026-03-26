"""Tests for pokedex/inference.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pokedex.inference import TorchPredictor, try_load_torch_predictor
from pokedex.training import build_model


def _tiny_classes(n: int) -> list[dict]:
    return [{"index": i, "dex": i + 1, "name": f"P{i}"} for i in range(n)]


def test_try_load_no_checkpoint(tmp_path: Path) -> None:
    classes = _tiny_classes(3)
    pred, status = try_load_torch_predictor(classes, tmp_path)
    assert pred is None
    assert status == "no_checkpoint"


def test_torch_predictor_roundtrip(tmp_path: Path, label_map_path: Path) -> None:
    classes = json.loads(label_map_path.read_text(encoding="utf-8"))["classes"]
    n = len(classes)
    model = build_model(n, "resnet18")
    ckpt_path = tmp_path / "c.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": n,
            "image_size": 224,
            "backbone": "resnet18",
        },
        ckpt_path,
    )
    pred = TorchPredictor(classes, ckpt_path, None)
    from PIL import Image

    im = Image.new("RGB", (300, 200), color="blue")
    out = pred.predict(im)
    assert len(out) == 3
    assert all(isinstance(x[0], int) and isinstance(x[1], str) and 0 <= x[2] <= 1 for x in out)


def test_try_load_torch_success(tmp_path: Path, label_map_path: Path) -> None:
    classes = json.loads(label_map_path.read_text(encoding="utf-8"))["classes"]
    n = len(classes)
    model = build_model(n, "resnet18")
    ckpt_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": n,
            "image_size": 224,
            "backbone": "resnet18",
        },
        ckpt_path,
    )
    pred, status = try_load_torch_predictor(classes, tmp_path, ckpt_path, None)
    assert status == "torch"
    assert pred is not None


def test_torch_predictor_wrong_class_count(tmp_path: Path) -> None:
    classes = _tiny_classes(5)
    model = build_model(5, "resnet18")
    ckpt_path = tmp_path / "c.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 5,
            "image_size": 224,
            "backbone": "resnet18",
        },
        ckpt_path,
    )
    with pytest.raises(ValueError, match="num_classes"):
        TorchPredictor(classes[:3], ckpt_path, None)
