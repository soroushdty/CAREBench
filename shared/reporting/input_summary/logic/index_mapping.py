"""Index mapping output writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.utils.json_utils import save_json_file
from ..outputs.notes_outputs import _resolve_reproducibility_dir


def save_index_mapping_outputs(
    *,
    cfg: dict[str, Any],
    split_index_maps: dict[str, dict[int, Any]],
) -> list[str]:
    # Single gate: REPRODUCIBILITY.artifacts.index_map_json in config/config.yaml.
    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    artifact_toggles = reproducibility_cfg.get("artifacts", {}) or {}
    if not bool(artifact_toggles.get("index_map_json", True)):
        return []

    summary_dir = Path(cfg.get("DIR_SUMMARY", "output/summary"))
    summary_dir.mkdir(parents=True, exist_ok=True)
    reproducibility_dir = _resolve_reproducibility_dir(summary_dir, cfg)
    output_path = reproducibility_dir / "index_map.json"
    payload = {
        split_name: {str(k): v for k, v in index_map.items()}
        for split_name, index_map in split_index_maps.items()
    }
    save_json_file(payload, output_path, ensure_ascii=False, indent=2)
    return [str(output_path)]
