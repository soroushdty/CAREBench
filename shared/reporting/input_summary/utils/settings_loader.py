"""Load summary defaults from configs/summary_config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml


@lru_cache(maxsize=1)
def load_default_summary() -> dict[str, Any]:
    """Load default summary settings from the package YAML file."""
    config_path = Path(__file__).resolve().parents[4] / "configs" / "summary_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError("configs/summary_config.yaml must be a mapping at top level")
    return payload


