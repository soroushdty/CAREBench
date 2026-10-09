"""Cleanup of obsolete summary artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import remove_if_exists


def remove_obsolete_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_dir: Path,
    selected_stages: list[str],
    stage_data: dict[str, dict[str, pd.DataFrame]],
) -> None:
    remove_if_exists(result, strict, summary_dir / "missingness_matrix.xlsx")
    for obsolete_name in [
        "stage_audit.csv",
        "standardization_impact_summary.csv",
        "raw_to_standardized_mapping.csv",
        "standardization_many_to_one_collapses.csv",
        "split_overlap_raw.csv",
        "split_overlap_standardized.csv",
        "split_overlap_final.csv",
        "physician_multirow_groups.csv",
        "physician_disagreement_rates.csv",
        "physician_conflicting_groups.csv",
        "physician_collapse_counts.csv",
        "summary_notes_after.md",
        "summary_notes_after.txt",
        "summary_index.json",
    ]:
        remove_if_exists(result, strict, summary_dir / "notes" / obsolete_name)
    for stage in selected_stages:
        remove_if_exists(result, strict, summary_dir / stage / "missingness_matrix.png")
        remove_if_exists(result, strict, summary_dir / stage / "missingness_matrix.csv")
        for split in stage_data[stage].keys():
            out_dir = summary_dir / stage / split
            for obsolete_name in [
                "missing_counts.csv",
                "missing_percentage.csv",
                "rows_per_patient_distribution.png",
                "top_patients_by_row_count.csv",
                "top_patients_by_positive_labels.csv",
                "label_cardinality_summary.csv",
                "labels_per_row_distribution.csv",
                "label_cardinality.png",
                "text_length_stats.csv",
                "text_length_distribution.png",
                "per_label_global_counts.csv",
                "fractional_values.png",
            ]:
                remove_if_exists(result, strict, out_dir / obsolete_name)
