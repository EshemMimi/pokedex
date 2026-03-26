"""Shared fixtures for Pokédex tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def label_map_path(project_root: Path) -> Path:
    return project_root / "artifacts" / "label_map.json"
