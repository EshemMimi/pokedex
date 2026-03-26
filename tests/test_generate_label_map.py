"""Tests for scripts/generate_label_map.py (151 canonical names)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_generate_module(project_root: Path):
    path = project_root / "scripts" / "generate_label_map.py"
    spec = importlib.util.spec_from_file_location("generate_label_map", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_names_list_has_151_entries(project_root: Path) -> None:
    gen = _load_generate_module(project_root)
    assert len(gen.NAMES) == 151


def test_main_writes_valid_json(tmp_path: Path, project_root: Path, monkeypatch) -> None:
    gen = _load_generate_module(project_root)
    out = tmp_path / "label_map.json"
    monkeypatch.setattr(gen, "OUT", out)
    gen.main()
    data = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert data["num_classes"] == 151
    assert len(data["classes"]) == 151
