"""Project paths and settings, read from config.yaml at the project root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> SimpleNamespace:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = _ns(yaml.safe_load(f))
    cfg.paths.raw_dir = ROOT / cfg.paths.raw_dir
    cfg.paths.artifacts_dir = ROOT / cfg.paths.artifacts_dir
    return cfg
