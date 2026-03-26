"""Tests for pokedex/ingest.py."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
import pytest

from pokedex.ingest import (
    build_manifest,
    build_name_to_dex,
    dedupe_by_hash,
    load_classes,
    resolve_folder_name,
    stratified_split,
)


def test_slug_and_resolve_common_folders(label_map_path: Path) -> None:
    classes = load_classes(label_map_path)
    lookup = build_name_to_dex(classes)
    assert resolve_folder_name("Bulbasaur", lookup) == 1
    assert resolve_folder_name("bulbasaur", lookup) == 1
    assert resolve_folder_name("001", lookup) == 1
    assert resolve_folder_name("025", lookup) == 25
    assert resolve_folder_name("Mr_Mime", lookup) == 122
    assert resolve_folder_name("nidoran-f", lookup) == 29
    assert resolve_folder_name("Farfetch'd", lookup) == 83
    assert resolve_folder_name("unknown", lookup) is None
    assert resolve_folder_name("999", lookup) is None


def test_stratified_split_counts_per_class() -> None:
    class_index = {1: 0, 2: 1}
    rows = [
        (Path(f"/a{i}.png"), 1, "tag") for i in range(10)
    ] + [(Path(f"/b{i}.png"), 2, "tag") for i in range(10)]
    out = stratified_split(rows, 0.7, 0.15, 0.15, seed=0, class_index=class_index)
    by_split: dict[str, int] = {}
    for _, _, _, sp in out:
        by_split[sp] = by_split.get(sp, 0) + 1
    assert by_split.get("train", 0) == 14  # 7+7 per class
    assert by_split.get("val", 0) == 2
    assert by_split.get("test", 0) == 4


def test_dedupe_by_hash_keeps_one(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    same = b"a" * 100
    a.write_bytes(same)
    b.write_bytes(same)
    rows = [(a, 1, "t"), (b, 1, "t")]
    kept, dropped = dedupe_by_hash(rows)
    assert dropped == 1
    assert len(kept) == 1


def test_build_manifest_minimal_dataset(
    label_map_path: Path, project_root: Path, tmp_path: Path
) -> None:
    """Two class folders with one image each; dedupe none."""
    root = tmp_path / "data"
    for name, dex in [("Bulbasaur", 1), ("Pikachu", 25)]:
        d = root / name
        d.mkdir(parents=True)
        Image.new("RGB", (16, 16), color=(dex, 0, 0)).save(d / f"{dex}.png")
    out_csv = tmp_path / "manifest.csv"
    warnings, stats = build_manifest(
        label_map_path=label_map_path,
        data_roots=[root],
        out_csv=out_csv,
        project_root=tmp_path,
        train_ratio=0.0,
        val_ratio=0.5,
        test_ratio=0.5,
        seed=42,
    )
    assert stats["images"] == 2
    assert stats["classes_with_samples"] == 2
    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    splits = {r["split"] for r in rows}
    assert splits <= {"train", "val", "test"}
    assert {int(r["dex"]) for r in rows} == {1, 25}


def test_train_val_split_must_sum_to_one() -> None:
    class_index = {1: 0}
    rows = [(Path("/x.png"), 1, "t")]
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(rows, 0.5, 0.5, 0.5, 0, class_index)
