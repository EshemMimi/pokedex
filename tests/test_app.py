"""Tests for app.py predictor wiring (no Gradio server)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app as app_module


def test_load_label_map(project_root: Path) -> None:
    classes = app_module.load_label_map()
    assert len(classes) == 151
    assert classes[0]["name"] == "Bulbasaur"


def test_dummy_predictor_top3(project_root: Path) -> None:
    classes = app_module.load_label_map()
    d = app_module.DummyPredictor(classes)
    class _Img:
        pass

    out = d.predict(_Img())
    assert len(out) == 3
    dexes = {x[0] for x in out}
    assert len(dexes) == 3
    assert abs(sum(x[2] for x in out) - 1.0) < 1e-5


def test_resolve_predictor_no_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    classes = [{"index": i, "dex": i + 1, "name": f"P{i}"} for i in range(151)]
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(app_module, "LABEL_MAP_PATH", tmp_path / "lm.json")
    (tmp_path / "artifacts").mkdir(parents=True)
    (tmp_path / "artifacts" / "label_map.json").write_text(
        json.dumps({"num_classes": 151, "classes": classes}), encoding="utf-8"
    )
    pred, msg = app_module._resolve_predictor(classes)
    assert isinstance(pred, app_module.DummyPredictor)
    assert "dummy" in msg.lower() or "no checkpoint" in msg.lower() or "best_model" in msg.lower()
