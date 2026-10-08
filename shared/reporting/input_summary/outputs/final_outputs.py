"""Final-stage outputs (final CSVs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..logic.divisions import combine_splits
from ..utils.io_utils import output_enabled, write_csv


def write_final_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    stage_data: dict[str, dict[str, pd.DataFrame]],
) -> None:
    if not output_enabled(summary_cfg, "final_combined_csv", default=True):
        return

    final_stage = stage_data.get("final", {})
    combined = final_stage.get("combined")
    if combined is None:
        combined = combine_splits(final_stage)
    write_csv(result, strict, combined, summary_dir / "final" / "final_combined.csv")
