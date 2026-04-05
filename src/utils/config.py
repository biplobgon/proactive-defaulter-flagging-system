"""
utils/config.py
---------------
YAML configuration loader with dot-notation access.
Mirrors the structure from product-recommendation-system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a YAML dict that supports attribute-style access."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and merge model_config.yaml and pipeline_config.yaml.

    When *config_path* is None both configs are loaded relative to the
    project root (detected by locating the configs/ directory).

    As a side-effect, the process working directory is changed to the
    project root so that all relative paths in the config (e.g.
    ``data/raw/application_train.csv``) resolve correctly regardless of
    where the calling script or notebook is located.
    """
    import os

    project_root = _find_project_root()
    os.chdir(project_root)

    if config_path is not None:
        raw = _read_yaml(Path(config_path))
    else:
        model_cfg = _read_yaml(project_root / "configs" / "model_config.yaml")
        pipeline_cfg = _read_yaml(project_root / "configs" / "pipeline_config.yaml")
        raw = {**model_cfg, **pipeline_cfg}

    return Config(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk upward from this file until a configs/ directory is found."""
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        "Could not locate project root (no configs/ directory found). "
        "Run from the project root or set PYTHONPATH appropriately."
    )


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
