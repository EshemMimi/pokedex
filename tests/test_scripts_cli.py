"""Smoke tests for CLI entry points (--help)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(script: str, args: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    return subprocess.call(
        [sys.executable, str(root / script)] + args,
        cwd=root,
    )


def test_train_help_exits_zero() -> None:
    assert _run("scripts/train.py", ["--help"]) == 0


def test_build_manifest_help_exits_zero() -> None:
    assert _run("scripts/build_manifest.py", ["--help"]) == 0
