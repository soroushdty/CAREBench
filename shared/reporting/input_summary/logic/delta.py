"""Delta output generation for paired test rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.io_utils import output_enabled, skip, write_csv


DELTA_METRIC_COLUMNS: list[str] = [
    "positive_delta",
    "negative_delta",
    "zero_delta",
    "delta_-1",
    "delta_+1",
    "delta_-0.5",
    "delta_+0.5",
]


def _safe_percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _delta_metric_counts(delta_values: pd.Series) -> tuple[dict[str, int], int]:
    numeric = pd.to_numeric(delta_values, errors="coerce").dropna()
    total = int(len(numeric))
    counts = {
        "positive_delta": int((numeric > 0).sum()),
        "negative_delta": int((numeric < 0).sum()),
        "zero_delta": int((numeric == 0).sum()),
        "delta_-1": int((numeric == -1).sum()),
        "delta_+1": int((numeric == 1).sum()),
        "delta_-0.5": int((numeric == -0.5).sum()),
        "delta_+0.5": int((numeric == 0.5).sum()),
    }
    return counts, total


def _delta_metric_percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    return {name: _safe_percentage(float(counts[name]), float(total)) for name in DELTA_METRIC_COLUMNS}


def write_delta_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    cfg: dict[str, Any],
    stage_data: dict[str, dict[str, pd.DataFrame]],
) -> None:
    final_combined = stage_data.get("final", {}).get("combined")
    if final_combined is None:
        skip(result, "final/delta/*.csv (missing final/combined snapshot)")
        return

    configured_classes = [str(c) for c in list(cfg.get("classes") or [])]
    if not configured_classes:
        skip(result, "final/delta/*.csv (missing cfg classes)")
        return

    available_classes = [
        class_name
        for class_name in configured_classes
        if f"{class_name}_survey" in final_combined.columns and f"{class_name}_interview" in final_combined.columns
    ]
    if not available_classes:
        skip(result, "final/delta/*.csv (missing survey/interview class pairs)")
        return

    survey_cols = [f"{class_name}_survey" for class_name in available_classes]
    interview_cols = [f"{class_name}_interview" for class_name in available_classes]
    if "split" not in final_combined.columns:
        skip(result, "final/delta/*.csv (missing final split column)")
        return

    paired_mask = final_combined["split"].eq("test") & final_combined[interview_cols].notna().any(axis=1)
    paired_test = final_combined.loc[paired_mask].copy(deep=True)

    delta_dir = summary_dir / "final" / "delta"

    count_rows_per_class: list[dict[str, Any]] = []
    pct_rows_per_class: list[dict[str, Any]] = []
    class_valid_totals: list[int] = []

    for class_name in configured_classes:
        if class_name in available_classes:
            delta_values = paired_test[f"{class_name}_interview"] - paired_test[f"{class_name}_survey"]
            counts, total = _delta_metric_counts(delta_values)
        else:
            counts = {name: 0 for name in DELTA_METRIC_COLUMNS}
            total = 0

        count_row: dict[str, Any] = {"class": class_name}
        count_row.update(counts)
        count_rows_per_class.append(count_row)

        pct_row: dict[str, Any] = {"class": class_name}
        pct_row.update(_delta_metric_percentages(counts, total))
        pct_rows_per_class.append(pct_row)
        class_valid_totals.append(total)

    total_count_row: dict[str, Any] = {
        "class": "Total",
        **{name: int(sum(int(r[name]) for r in count_rows_per_class)) for name in DELTA_METRIC_COLUMNS},
    }
    grand_total = int(sum(class_valid_totals))
    total_pct_row: dict[str, Any] = {
        "class": "Total",
        **{name: _safe_percentage(float(total_count_row[name]), float(grand_total)) for name in DELTA_METRIC_COLUMNS},
    }
    count_rows_per_class.append(total_count_row)
    pct_rows_per_class.append(total_pct_row)

    per_class_cols = ["class", *DELTA_METRIC_COLUMNS]
    count_per_class_df = pd.DataFrame(count_rows_per_class)[per_class_cols]
    pct_per_class_df = pd.DataFrame(pct_rows_per_class)[per_class_cols]

    if output_enabled(summary_cfg, "delta_percentage_per_class_csv", default=True):
        write_csv(result, strict, pct_per_class_df, delta_dir / "percentage_per_class.csv")
    if output_enabled(summary_cfg, "delta_count_per_class_csv", default=True):
        write_csv(result, strict, count_per_class_df, delta_dir / "count_per_class.csv")

    strata = ["repeated", "novel"]
    count_rows_per_stratum: list[dict[str, Any]] = []
    pct_rows_per_stratum: list[dict[str, Any]] = []

    for class_name in configured_classes:
        for stratum_name in strata:
            if class_name in available_classes and "stratum" in paired_test.columns:
                stratum_mask = paired_test["stratum"] == stratum_name
                stratum_df = paired_test.loc[stratum_mask]
                delta_values = stratum_df[f"{class_name}_interview"] - stratum_df[f"{class_name}_survey"]
                counts, total = _delta_metric_counts(delta_values)
            else:
                counts = {name: 0 for name in DELTA_METRIC_COLUMNS}
                total = 0

            count_row = {"class": class_name, "stratum": stratum_name}
            count_row.update(counts)
            count_rows_per_stratum.append(count_row)

            pct_row = {"class": class_name, "stratum": stratum_name}
            pct_row.update(_delta_metric_percentages(counts, total))
            pct_rows_per_stratum.append(pct_row)

    for stratum_name in strata:
        stratum_count_rows = [r for r in count_rows_per_stratum if r["stratum"] == stratum_name]
        stratum_total_count_row: dict[str, Any] = {
            "class": "Total",
            "stratum": stratum_name,
            **{name: int(sum(int(r[name]) for r in stratum_count_rows)) for name in DELTA_METRIC_COLUMNS},
        }
        stratum_valid_total = int(
            sum(
                _delta_metric_counts(
                    (
                        paired_test.loc[paired_test["stratum"] == stratum_name, f"{class_name}_interview"]
                        - paired_test.loc[paired_test["stratum"] == stratum_name, f"{class_name}_survey"]
                    )
                    if class_name in available_classes and "stratum" in paired_test.columns
                    else pd.Series(dtype=float)
                )[1]
                for class_name in configured_classes
            )
        )
        stratum_total_pct_row: dict[str, Any] = {
            "class": "Total",
            "stratum": stratum_name,
            **{
                name: _safe_percentage(float(stratum_total_count_row[name]), float(stratum_valid_total))
                for name in DELTA_METRIC_COLUMNS
            },
        }
        count_rows_per_stratum.append(stratum_total_count_row)
        pct_rows_per_stratum.append(stratum_total_pct_row)

    per_stratum_cols = ["class", "stratum", *DELTA_METRIC_COLUMNS]
    count_per_stratum_df = pd.DataFrame(count_rows_per_stratum)[per_stratum_cols]
    pct_per_stratum_df = pd.DataFrame(pct_rows_per_stratum)[per_stratum_cols]

    if output_enabled(summary_cfg, "delta_percentage_per_stratum_csv", default=True):
        write_csv(result, strict, pct_per_stratum_df, delta_dir / "percentage_per_stratum.csv")
    if output_enabled(summary_cfg, "delta_count_per_stratum_csv", default=True):
        write_csv(result, strict, count_per_stratum_df, delta_dir / "count_per_stratum.csv")
