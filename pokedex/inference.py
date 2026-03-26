"""Load trained checkpoint and run Top-k predictions (same preprocessing as validation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from pokedex.training import build_model, build_transforms


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class TorchPredictor:
    """ResNet-18 classifier; preprocessing matches `build_transforms(..., train=False)`."""

    def __init__(
        self,
        classes: list[dict],
        checkpoint_path: Path,
        train_config_path: Path | None = None,
    ) -> None:
        self._classes = classes
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )
        cfg: dict[str, Any] = {}
        if train_config_path is not None and train_config_path.is_file():
            cfg = _read_json(train_config_path)

        raw_nc = ck.get("num_classes") if "num_classes" in ck else cfg.get("num_classes")
        if raw_nc is None:
            raise ValueError("num_classes missing from checkpoint and train_config.json")
        num_classes = int(raw_nc)
        if num_classes != len(classes):
            raise ValueError(
                f"num_classes={num_classes} but label_map has {len(classes)} classes"
            )
        backbone = str(ck.get("backbone") or cfg.get("backbone", "resnet18"))
        raw_sz = ck.get("image_size") if "image_size" in ck else cfg.get("image_size")
        if raw_sz is None:
            raise ValueError("image_size missing from checkpoint and train_config.json")
        image_size = int(raw_sz)

        self._model = build_model(num_classes, backbone).to(self._device)
        self._model.load_state_dict(ck["model_state_dict"])
        self._model.eval()
        self._transform = build_transforms(image_size, train=False)

    @torch.inference_mode()
    def predict(self, image) -> list[tuple[int, str, float]]:
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        x = self._transform(image).unsqueeze(0).to(self._device)
        logits = self._model(x)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(3, probs.numel())
        top = torch.topk(probs, k=k)
        rows: list[tuple[int, str, float]] = []
        for idx, p in zip(top.indices.tolist(), top.values.tolist()):
            c = self._classes[int(idx)]
            rows.append((int(c["dex"]), str(c["name"]), float(p)))
        return rows


def try_load_torch_predictor(
    classes: list[dict],
    project_root: Path,
    checkpoint_path: Path | None = None,
    train_config_path: Path | None = None,
) -> tuple[TorchPredictor | None, str]:
    """
    Returns (predictor, status). status is 'torch', 'no_checkpoint', or 'load_error: ...'.
    """
    ckpt = checkpoint_path or (project_root / "artifacts" / "best_model.pt")
    cfg_path = train_config_path or (project_root / "artifacts" / "train_config.json")
    if not ckpt.is_file():
        return None, "no_checkpoint"
    cfg_arg = cfg_path if cfg_path.is_file() else None
    try:
        pred = TorchPredictor(classes, ckpt, cfg_arg)
    except ValueError as e:
        err = str(e).lower()
        if "missing" in err or "num_classes" in err or "image_size" in err:
            return None, "missing_train_config"
        return None, f"load_error: {e}"
    except Exception as e:
        return None, f"load_error: {e}"
    return pred, "torch"
