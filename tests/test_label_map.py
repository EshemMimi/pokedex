"""Tests for artifacts/label_map.json and Gen 1 taxonomy."""

from __future__ import annotations

import json
from pathlib import Path


def test_label_map_exists_and_shape(label_map_path: Path) -> None:
    assert label_map_path.is_file(), "Run: python scripts/generate_label_map.py"


def test_label_map_gen1_coverage(label_map_path: Path) -> None:
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    assert data.get("generation") == 1
    assert data.get("num_classes") == 151
    classes = data["classes"]
    assert len(classes) == 151
    dexes = [c["dex"] for c in classes]
    assert dexes == list(range(1, 152))
    indices = [c["index"] for c in classes]
    assert indices == list(range(151))
    names = [c["name"] for c in classes]
    assert len(set(names)) == 151
    assert names[0] == "Bulbasaur"
    assert names[-1] == "Mew"
