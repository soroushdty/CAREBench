"""Utilities for creating unique run folders and updating config paths."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _build_run_id() -> str:
    now = datetime.now(UTC)
    return f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"


def apply_timestamped_run_folder(cfg: dict, output_root: str | Path | None = None) -> dict:
    """
    Apply unique run folder to config, updating DIR_MODEL and DIR_SUMMARY.

    If DIR_SUMMARY is configured as a relative path in main config, it is resolved
    under the generated run folder. If absolute, it is used as-is.
    """
    if output_root is None:
        output_root = cfg.get("DIR_OUTPUT", "output")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = _build_run_id()
    run_folder_path = output_root / run_id
    run_folder_path.mkdir(parents=True, exist_ok=False)

    configured_summary_dir = str(cfg.get("DIR_SUMMARY", "input_summary")).strip()
    summary_dir_candidate = Path(configured_summary_dir) if configured_summary_dir else Path("input_summary")
    summary_dir_path = (
        summary_dir_candidate
        if summary_dir_candidate.is_absolute()
        else run_folder_path / summary_dir_candidate
    )

    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    reproducibility_enabled = bool(reproducibility_cfg.get("folder_enabled", True))
    reproducibility_path_raw = str(reproducibility_cfg.get("folder_path", "reproducibility_artifacts")).strip()
    reproducibility_dir_candidate = (
        Path(reproducibility_path_raw) if reproducibility_path_raw else Path("reproducibility_artifacts")
    )
    reproducibility_dir_path = (
        reproducibility_dir_candidate
        if reproducibility_dir_candidate.is_absolute()
        else run_folder_path / reproducibility_dir_candidate
    )

    if reproducibility_enabled:
        reproducibility_dir_path.mkdir(parents=True, exist_ok=True)

    cfg["RUN_ID"] = run_id
    cfg["DIR_MODEL"] = str(run_folder_path / "model")
    cfg["DIR_SUMMARY"] = str(summary_dir_path)
    if reproducibility_enabled:
        cfg["DIR_REPRODUCIBILITY"] = str(reproducibility_dir_path)
    else:
        cfg.pop("DIR_REPRODUCIBILITY", None)

    return cfg
