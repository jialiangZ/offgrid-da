"""Configuration loading and utilities.

All data paths, variable names and split definitions are managed here
(plan section 0.4: configuration-driven development).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "wb2_metar.yaml"


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a copy)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the data configuration YAML, optionally merged with overrides."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    # Resolve cache dir relative to the project root
    cache_dir = Path(cfg["data"]["cache_dir"])
    if not cache_dir.is_absolute():
        cfg["data"]["cache_dir"] = str((PROJECT_ROOT / cache_dir).resolve())
    return cfg


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
