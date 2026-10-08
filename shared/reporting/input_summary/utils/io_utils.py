"""Shared utilities for summary output writing and toggle evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

from shared.utils.file_utils import ensure_parent_dir, remove_file_if_exists, write_csv_file

_logger = logging.getLogger("shared.reporting.input_summary")


def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def output_enabled(
    summary_cfg: dict[str, Any],
    key: str,
    *,
    default: bool = True,
) -> bool:
    outputs = summary_cfg.get("outputs") or {}
    if key in outputs:
        return bool(outputs[key])
    return default


def record(result: dict[str, Any], file_path: Path, *, plot: bool = False) -> None:
    path = str(file_path)
    result["generated_files"].append(path)
    if plot:
        result["generated_plots"].append(path)
    else:
        result["generated_tables"].append(path)


def warn(result: dict[str, Any], strict: bool, message: str) -> None:
    if strict:
        raise RuntimeError(message)
    result["warnings"].append(message)


def skip(result: dict[str, Any], reason: str) -> None:
    """Record a skipped artifact and emit a WARNING log entry immediately."""
    result["skipped"].append(reason)
    _logger.warning("Skipped: %s", reason)


def write_csv(result: dict[str, Any], strict: bool, frame: pd.DataFrame, path: Path) -> None:
    try:
        write_csv_file(frame, path, index=False)
        record(result, path)
    except Exception as exc:
        warn(result, strict, f"Failed to write CSV {path}: {exc}")


def remove_if_exists(result: dict[str, Any], strict: bool, path: Path) -> None:
    try:
        if remove_file_if_exists(path):
            result["removed_files"].append(str(path))
    except Exception as exc:
        warn(result, strict, f"Failed to remove obsolete file {path}: {exc}")


def write_missingness_workbook(
    result: dict[str, Any],
    strict: bool,
    workbook_path: Path,
    stage_matrices: dict[str, pd.DataFrame],
) -> None:
    if not stage_matrices:
        return
    try:
        workbook_path = ensure_parent_dir(workbook_path)
        with pd.ExcelWriter(workbook_path) as writer:
            for stage, matrix in stage_matrices.items():
                sheet_name = str(stage)[:31]
                matrix.to_excel(writer, sheet_name=sheet_name, index=False)
        record(result, workbook_path)
    except Exception as exc:
        warn(result, strict, f"Failed to write missingness workbook {workbook_path}: {exc}")


def save_plot(result: dict[str, Any], strict: bool, fig: Figure, path: Path) -> None:
    try:
        path = ensure_parent_dir(path)
        fig.tight_layout()
        fig.savefig(path)
        record(result, path, plot=True)
    except Exception as exc:
        warn(result, strict, f"Failed to write plot {path}: {exc}")
    finally:
        plt.close(fig)
