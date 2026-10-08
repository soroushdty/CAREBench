"""Cross-stage audit outputs: stage audit, standardization, overlap, physician, and missingness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .divisions import item_set, resolve_class_cols, trim_raw_item
from ..utils.io_utils import output_enabled, skip, write_csv, write_missingness_workbook


def write_stage_audit(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    stage_stats: list[dict[str, Any]],
) -> None:
    if output_enabled(summary_cfg, "stage_audit_csv", default=True):
        write_csv(result, strict, pd.DataFrame(stage_stats), summary_dir / "audits" / "stage_audit.csv")


def write_standardization_impact(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    split_names: list[str],
    stage_data: dict[str, dict[str, pd.DataFrame]],
    item_col: str,
) -> None:
    if not (
        output_enabled(summary_cfg, "standardization_impact_summary_csv", default=True)
        or output_enabled(summary_cfg, "raw_to_standardized_mapping_csv", default=True)
        or output_enabled(summary_cfg, "standardization_many_to_one_collapses_csv", default=True)
    ):
        return

    impact_rows: list[dict[str, Any]] = []
    maps: list[pd.DataFrame] = []
    for split in [s for s in split_names if s != "combined"]:
        raw_df = stage_data.get("raw", {}).get(split)
        std_df = stage_data.get("standardized", {}).get(split)
        if raw_df is None or std_df is None or item_col not in raw_df.columns or item_col not in std_df.columns:
            continue
        pair = pd.DataFrame(
            {
                "split": split,
                "raw_item_trimmed": raw_df[item_col].map(trim_raw_item),
                "standardized_item": std_df[item_col].map(lambda v: None if pd.isna(v) else str(v).strip()),
            }
        )
        maps.append(pair)
        impact_rows.append(
            {
                "split": split,
                "raw_unique_items_count": int(pair["raw_item_trimmed"].dropna().nunique()),
                "standardized_unique_items_count": int(pair["standardized_item"].dropna().nunique()),
                "rows_changed": int((pair["raw_item_trimmed"].fillna("") != pair["standardized_item"].fillna("")).sum()),
            }
        )

    if impact_rows and output_enabled(
        summary_cfg,
        "standardization_impact_summary_csv",
        default=True,
    ):
        write_csv(result, strict, pd.DataFrame(impact_rows), summary_dir / "audits" / "standardization_impact_summary.csv")

    if maps and (
        output_enabled(summary_cfg, "raw_to_standardized_mapping_csv", default=True)
        or output_enabled(summary_cfg, "standardization_many_to_one_collapses_csv", default=True)
    ):
        mapping_df = pd.concat(maps, axis=0, ignore_index=True)
        mapping_counts = (
            mapping_df.value_counts(["split", "raw_item_trimmed", "standardized_item"], dropna=False)
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        if output_enabled(summary_cfg, "raw_to_standardized_mapping_csv", default=True):
            write_csv(result, strict, mapping_counts, summary_dir / "audits" / "raw_to_standardized_mapping.csv")

        many_to_one = (
            mapping_counts.groupby(["split", "standardized_item"], dropna=False)
            .agg(raw_forms=("raw_item_trimmed", "nunique"), total_rows=("count", "sum"))
            .reset_index()
        )
        many_to_one = many_to_one[many_to_one["raw_forms"] > 1].sort_values(["raw_forms", "total_rows"], ascending=False)
        if output_enabled(summary_cfg, "standardization_many_to_one_collapses_csv", default=True):
            write_csv(result, strict, many_to_one, summary_dir / "audits" / "standardization_many_to_one_collapses.csv")


def write_split_overlap(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    selected_stages: list[str],
    stage_data: dict[str, dict[str, pd.DataFrame]],
    item_col: str,
) -> None:
    if not (
        output_enabled(summary_cfg, "split_overlap_raw_csv", default=True)
        or output_enabled(summary_cfg, "split_overlap_standardized_csv", default=True)
        or output_enabled(summary_cfg, "split_overlap_post_physician_merge_csv", default=True)
    ):
        return

    for stage in ["raw", "standardized", "post_physician_merge"]:
        if stage not in selected_stages:
            continue
        stage_key = f"split_overlap_{stage}_csv"
        if not output_enabled(summary_cfg, stage_key, default=True):
            continue

        rows: list[dict[str, Any]] = []
        for left, right in [("train", "test"), ("train", "interview"), ("test", "interview")]:  # PAIRED-CONTEXT SCHEMA
            left_df = stage_data.get(stage, {}).get(left)
            right_df = stage_data.get(stage, {}).get(right)
            if left_df is None or right_df is None:
                continue
            left_set = item_set(left_df, item_col, raw_mode=(stage == "raw"))
            right_set = item_set(right_df, item_col, raw_mode=(stage == "raw"))
            overlap = left_set.intersection(right_set)
            union = left_set.union(right_set)
            rows.append(
                {
                    "stage": stage,
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": len(overlap),
                    "union_count": len(union),
                    "jaccard_similarity": (len(overlap) / len(union)) if union else 0.0,
                    "unique_to_left_count": len(left_set - right_set),
                    "unique_to_right_count": len(right_set - left_set),
                }
            )
        if rows:
            write_csv(result, strict, pd.DataFrame(rows), summary_dir / "audits" / f"split_overlap_{stage}.csv")


def write_physician_disagreement(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    stage_data: dict[str, dict[str, pd.DataFrame]],
    patient_col: str | None,
    physician_col: str | None,
    item_col: str,
    metadata: dict[str, Any],
) -> None:
    """Write physician disagreement diagnostics to the audits directory.

    Computes disagreement rates and conflicting label groups from the
    standardized/train split. Collapse counts cover train, test, and interview.
    Requires patient_col, physician_col, and item_col to all be present in the
    standardized/train DataFrame; silently skips if any are missing.
    """
    if not (
        output_enabled(summary_cfg, "physician_multirow_groups_csv", default=True)
        or output_enabled(summary_cfg, "physician_disagreement_rates_csv", default=True)
        or output_enabled(summary_cfg, "physician_conflicting_groups_csv", default=True)
        or output_enabled(summary_cfg, "physician_collapse_counts_csv", default=True)
    ):
        return

    standard_train = stage_data.get("standardized", {}).get("train")
    post_phys_train = stage_data.get("post_physician_merge", {}).get("train")
    if (
        standard_train is not None
        and patient_col
        and physician_col
        and item_col in standard_train.columns
        and patient_col in standard_train.columns
        and physician_col in standard_train.columns
    ):
        class_cols = resolve_class_cols(standard_train, metadata, "standardized", "train")
        groups = standard_train.groupby([patient_col, item_col], dropna=False)
        multi_phys = groups.size().rename("rows_in_group").reset_index()
        multi_phys = multi_phys[multi_phys["rows_in_group"] > 1]
        if output_enabled(summary_cfg, "physician_multirow_groups_csv", default=True):
            write_csv(result, strict, multi_phys, summary_dir / "audits" / "physician_multirow_groups.csv")

        denom = max(len(multi_phys), 1)
        disagreement_rows: list[dict[str, Any]] = []
        conflicts: list[pd.DataFrame] = []
        for col in class_cols:
            nunique = groups[col].nunique(dropna=True).rename("nunique").reset_index()
            conflict = nunique[nunique["nunique"] > 1]
            conflict = conflict.merge(multi_phys[[patient_col, item_col]], on=[patient_col, item_col], how="inner")
            disagreement_rows.append(
                {
                    "class": col,
                    "disagreement_group_count": int(len(conflict)),
                    "disagreement_rate": float(len(conflict) / denom),
                }
            )
            if not conflict.empty:
                conflict["class"] = col
                conflicts.append(conflict)

        if output_enabled(summary_cfg, "physician_disagreement_rates_csv", default=True):
            write_csv(result, strict, pd.DataFrame(disagreement_rows), summary_dir / "audits" / "physician_disagreement_rates.csv")
        if conflicts and output_enabled(
            summary_cfg,
            "physician_conflicting_groups_csv",
            default=True,
        ):
            # NOTE: physician_conflicting_groups.csv grows with the number of conflicting
            # label pairs per (patient, item, class) and can become large at scale.
            write_csv(
                result,
                strict,
                pd.concat(conflicts, axis=0, ignore_index=True),
                summary_dir / "audits" / "physician_conflicting_groups.csv",
            )

        if output_enabled(summary_cfg, "physician_collapse_counts_csv", default=True):
            post_phys_test = stage_data.get("post_physician_merge", {}).get("test")
            post_phys_interview = stage_data.get("post_physician_merge", {}).get("interview")  # PAIRED-CONTEXT SCHEMA
            standard_test = stage_data.get("standardized", {}).get("test")
            standard_interview = stage_data.get("standardized", {}).get("interview")  # PAIRED-CONTEXT SCHEMA
            collapsed_train = len(standard_train) - len(post_phys_train) if post_phys_train is not None else None
            collapsed_test = (
                len(standard_test) - len(post_phys_test)
                if standard_test is not None and post_phys_test is not None
                else None
            )
            collapsed_interview = (
                len(standard_interview) - len(post_phys_interview)
                if standard_interview is not None and post_phys_interview is not None
                else None
            )
            write_csv(
                result,
                strict,
                pd.DataFrame([{
                    "rows_collapsed_by_physician_merge_train": collapsed_train,
                    "rows_collapsed_by_physician_merge_test": collapsed_test,
                    "rows_collapsed_by_physician_merge_interview": collapsed_interview,
                }]),
                summary_dir / "audits" / "physician_collapse_counts.csv",
            )
    else:
        skip(result, "audits/physician_disagreement_* (missing required columns)")


def write_missingness_matrix(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    stage_data: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Write a consolidated missingness summary workbook.

    Generates a single audits/missingness_summary.xlsx with one sheet per stage
    (raw, standardized, post_physician_merge, final). Rows represent splits,
    columns represent dataset column names, and cell values are missing percentages.
    """
    if not output_enabled(summary_cfg, "missingness_summary_xlsx", default=True):
        return

    workbook_path = summary_dir / "audits" / "missingness_summary.xlsx"
    stage_order = ["raw", "standardized", "post_physician_merge", "final"]
    stage_matrices: dict[str, pd.DataFrame] = {}

    for stage in stage_order:
        splits_data = stage_data.get(stage)
        if not splits_data:
            continue
        rows: list[dict[str, Any]] = []
        for split, df in splits_data.items():
            row: dict[str, Any] = {"split": split}
            pct = (df.isna().sum() / max(len(df), 1)) * 100.0
            for col, value in pct.items():
                col_name = str(col)
                if col_name == "split":
                    row["split_column_missing_percentage"] = float(value)
                else:
                    row[col_name] = float(value)
            rows.append(row)
        if rows:
            stage_matrices[stage] = pd.DataFrame(rows).fillna(0.0)

    write_missingness_workbook(result, strict, workbook_path, stage_matrices)
