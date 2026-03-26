"""Build a stratified train/val/test manifest from folder-per-class image datasets (Gen 1 only)."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def load_classes(label_map_path: Path) -> list[dict]:
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    return data["classes"]


def _slug(name: str) -> str:
    n = name.replace("♀", "-f").replace("♂", "-m")
    n = n.replace(".", "").replace("'", "")
    n = re.sub(r"[^a-zA-Z0-9]+", "-", n)
    n = n.strip("-").lower()
    return re.sub(r"-+", "-", n)


def _compact_slug(name: str) -> str:
    return _slug(name).replace("-", "")


def build_name_to_dex(classes: list[dict]) -> dict[str, int]:
    """Map various normalized folder-name keys to National Dex number (1–151)."""
    m: dict[str, int] = {}
    for c in classes:
        dex = int(c["dex"])
        name = str(c["name"])
        keys = {
            name.lower(),
            _slug(name),
            _compact_slug(name),
            str(dex),
            f"{dex:03d}",
        }
        for k in keys:
            if k:
                m[k] = dex
    # Common dataset variants not covered by slug alone
    extra = {
        "mr-mime": 122,
        "mrmime": 122,
        "mr.mime": 122,
        "farfetchd": 83,
        "farfetch'd": 83,
        "nidoran-f": 29,
        "nidoran-female": 29,
        "nidoranf": 29,
        "nidoran-m": 32,
        "nidoran-male": 32,
        "nidoranm": 32,
    }
    for k, dex in extra.items():
        if 1 <= dex <= 151:
            m.setdefault(k, dex)
    return m


def resolve_folder_name(folder: str, lookup: dict[str, int]) -> int | None:
    raw = folder.strip()
    if not raw:
        return None
    # Numeric: 1, 01, 001
    if raw.isdigit():
        d = int(raw)
        if 1 <= d <= 151:
            return d
    key = raw.lower().replace("_", " ").replace(".", " ")
    key = re.sub(r"\s+", " ", key).strip()
    if key in lookup:
        return lookup[key]
    key2 = key.replace(" ", "-")
    if key2 in lookup:
        return lookup[key2]
    slug = _slug(raw)
    if slug in lookup:
        return lookup[slug]
    compact = _compact_slug(raw)
    if compact in lookup:
        return lookup[compact]
    return None


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_labeled_images(
    data_roots: list[Path],
    lookup: dict[str, int],
) -> tuple[list[tuple[Path, int, str]], list[str]]:
    """Returns (rows of absolute path, dex, source_tag), warnings."""
    rows: list[tuple[Path, int, str]] = []
    warnings: list[str] = []
    for root in data_roots:
        root = root.resolve()
        if not root.is_dir():
            warnings.append(f"Skip missing directory: {root}")
            continue
        tag = root.name
        # Case A: root contains class subfolders
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if subdirs:
            for sub in sorted(subdirs):
                dex = resolve_folder_name(sub.name, lookup)
                if dex is None:
                    warnings.append(f"Unmapped folder (skipped): {sub}")
                    continue
                for img in iter_images(sub):
                    rows.append((img, dex, tag))
        else:
            # Case B: images directly under root — try to infer from filename (not supported)
            warnings.append(
                f"No class subfolders under {root}; expected one folder per species. Skipped."
            )
    return rows, warnings


def dedupe_by_hash(
    rows: list[tuple[Path, int, str]],
) -> tuple[list[tuple[Path, int, str]], int]:
    """Drop duplicate files by SHA-256 (exact). Keeps first occurrence per hash."""
    seen: dict[str, tuple[Path, int, str]] = {}
    dropped = 0
    for row in rows:
        path, dex, tag = row
        try:
            digest = sha256_file(path)
        except OSError:
            dropped += 1
            continue
        if digest not in seen:
            seen[digest] = row
        else:
            dropped += 1
    return list(seen.values()), dropped


def stratified_split(
    rows: list[tuple[Path, int, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    class_index: dict[int, int],
) -> list[tuple[Path, int, int, str]]:
    """Assign split per dex; returns list of (path, dex, class_index, split_name)."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")
    rng = random.Random(seed)
    by_dex: dict[int, list[tuple[Path, int, str]]] = defaultdict(list)
    for path, dex, tag in rows:
        by_dex[dex].append((path, dex, tag))

    out: list[tuple[Path, int, int, str]] = []
    for dex, group in by_dex.items():
        if not group:
            continue
        rng.shuffle(group)
        n = len(group)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val
        idx = class_index[dex]
        for i, item in enumerate(group):
            path, _, tag = item
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            out.append((path, dex, idx, split))
    return out


def write_manifest(
    rows: list[tuple[Path, int, int, str]],
    out_csv: Path,
    project_root: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "dex", "class_index", "split"])
        for path, dex, idx, split in rows:
            try:
                rel = path.relative_to(project_root.resolve())
            except ValueError:
                rel = path
            w.writerow([str(rel).replace("\\", "/"), dex, idx, split])


def build_manifest(
    *,
    label_map_path: Path,
    data_roots: list[Path],
    out_csv: Path,
    project_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], dict[str, int]]:
    """End-to-end: scan roots, dedupe, stratified split, write CSV. Returns (warnings, stats)."""
    classes = load_classes(label_map_path)
    lookup = build_name_to_dex(classes)
    class_index = {int(c["dex"]): int(c["index"]) for c in classes}

    rows, warnings = collect_labeled_images(data_roots, lookup)
    rows, dropped_dupes = dedupe_by_hash(rows)
    rng_rows = stratified_split(
        rows, train_ratio, val_ratio, test_ratio, seed, class_index
    )
    write_manifest(rng_rows, out_csv, project_root)

    stats = {
        "images": len(rng_rows),
        "dedupe_dropped": dropped_dupes,
        "classes_with_samples": len({d for _, d, _, _ in rng_rows}),
    }
    return warnings, stats
