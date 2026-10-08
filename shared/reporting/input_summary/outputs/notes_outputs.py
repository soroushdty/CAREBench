"""README and reproducibility summary index writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.utils.json_utils import save_json_file
from ..utils.io_utils import output_enabled, record, warn


def _resolve_reproducibility_dir(summary_dir: Path, cfg: dict[str, Any]) -> Path:
    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    reproducibility_dir_raw = cfg.get("DIR_REPRODUCIBILITY")
    if reproducibility_dir_raw:
        return Path(reproducibility_dir_raw)

    folder_path = str(reproducibility_cfg.get("folder_path", "reproducibility_artifacts")).strip()
    folder_candidate = Path(folder_path) if folder_path else Path("reproducibility_artifacts")
    return folder_candidate if folder_candidate.is_absolute() else summary_dir.parent / folder_candidate



def write_summary_index(
    *,
    result: dict[str, Any],
    strict: bool,
    cfg: dict[str, Any],
    summary_dir: Path,
) -> None:
    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    if not bool(reproducibility_cfg.get("folder_enabled", True)):
        return
    artifact_toggles = reproducibility_cfg.get("artifacts", {}) or {}
    if not bool(artifact_toggles.get("summary_index_json", True)):
        return

    reproducibility_dir = _resolve_reproducibility_dir(summary_dir, cfg)
    reproducibility_dir.mkdir(parents=True, exist_ok=True)

    project_root = cfg.get("PROJECT_ROOT")

    def _to_rel(path_str: str) -> str:
        if not project_root:
            return path_str
        try:
            return str(Path(path_str).relative_to(project_root))
        except ValueError:
            return path_str

    summary_index_path = reproducibility_dir / "summary_index.json"
    summary_index_payload = {
        "summary_dir": _to_rel(result["summary_dir"]),
        "generated_files": [_to_rel(p) for p in result["generated_files"]],
        "removed_files": [_to_rel(p) for p in result["removed_files"]],
        "warnings": result["warnings"],
        "skipped": result["skipped"],
        "stage_stats": result["stage_stats"],
    }
    try:
        save_json_file(summary_index_payload, summary_index_path, ensure_ascii=False, indent=2)
        record(result, summary_index_path)
    except Exception as exc:
        warn(result, strict, f"Failed to write summary index JSON: {exc}")


def write_runtime_summary_config(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    cfg: dict[str, Any],
) -> None:
    if not output_enabled(summary_cfg, "runtime_summary_config_json", default=True):
        return

    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    artifact_toggles = reproducibility_cfg.get("artifacts", {}) or {}
    if not bool(artifact_toggles.get("summary_config_json", True)):
        return

    if not bool(reproducibility_cfg.get("folder_enabled", True)):
        return

    reproducibility_dir = _resolve_reproducibility_dir(summary_dir, cfg)
    reproducibility_dir.mkdir(parents=True, exist_ok=True)

    project_root = cfg.get("PROJECT_ROOT")

    def _to_rel(path_str: str | None) -> str | None:
        if not path_str or not project_root:
            return path_str
        try:
            return str(Path(path_str).relative_to(project_root))
        except ValueError:
            return path_str

    summary_config_path = Path(__file__).resolve().parents[4] / "configs" / "summary_config.yaml"
    config_source = str(
        summary_config_path.relative_to(Path(project_root))
        if project_root
        else summary_config_path
    )

    runtime_config_path = reproducibility_dir / "summary_config.json"
    runtime_config_payload = {
        "runtime_summary_config": summary_cfg,
        "provenance": {
            "default_summary_config_source": config_source,
            "runtime_overrides_source": "configs/main_config.yaml: SUMMARY",
            "main_config_path": _to_rel(cfg.get("CONFIG_PATH")),
            "run_id": cfg.get("RUN_ID"),
        },
    }
    try:
        save_json_file(runtime_config_payload, runtime_config_path, ensure_ascii=False, indent=2)
        record(result, runtime_config_path)
    except Exception as exc:
        warn(result, strict, f"Failed to write runtime summary config JSON: {exc}")
