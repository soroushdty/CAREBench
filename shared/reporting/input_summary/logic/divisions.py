"""Split/stage/column handling utilities for summary generation."""

from __future__ import annotations

from typing import Any
import re
import string

import pandas as pd


FIXED_STAGES = ["raw", "standardized", "post_physician_merge", "final"]


def copy_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {k: v.copy(deep=True) for k, v in splits.items()}


def combine_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not splits:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for split, frame in splits.items():
        chunk = frame.copy(deep=True)
        chunk["split"] = split
        chunks.append(chunk)
    return pd.concat(chunks, axis=0, ignore_index=True)


def trim_raw_item(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return ""
    punctuation = re.escape(string.punctuation)
    return re.sub(rf"^[{punctuation}]+|[{punctuation}]+$", "", text)


def numeric_binary(df: pd.DataFrame, class_cols: list[str]) -> pd.DataFrame:
    if not class_cols:
        return pd.DataFrame(index=df.index)
    numeric = df[class_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (numeric > 0).astype(int)


def resolve_class_cols(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    stage: str,
    split: str | None = None,
) -> list[str]:
    """Return the list of class columns applicable to a given stage and split.

    Columns are resolved in priority order: stage_class_cols[stage][split],
    then raw_class_cols (for non-final stages) or final_class_cols / _survey
    suffixed columns (for the final stage). The '_survey' suffix stripping
    convention means that a configured class name 'Foo' will match 'Foo_survey'
    when the base name is absent from the final-stage DataFrame.
    """
    stage_class_cols = metadata.get("stage_class_cols")
    if isinstance(stage_class_cols, dict) and split is not None:
        stage_map = stage_class_cols.get(stage)
        if isinstance(stage_map, dict):
            preferred = list(stage_map.get(split) or [])
        else:
            preferred = []
    elif stage == "final":
        preferred = list(metadata.get("final_class_cols") or [])
        if not preferred:
            preferred = [col[:-7] for col in df.columns if col.endswith("_survey")]
    else:
        preferred = list(metadata.get("raw_class_cols") or [])

    if not preferred and split == "combined":
        if stage == "final":
            preferred = list(metadata.get("final_class_cols") or [])
            if not preferred:
                preferred = [col[:-7] for col in df.columns if col.endswith("_survey")]
        else:
            preferred = list(metadata.get("raw_class_cols") or [])

    resolved: list[str] = []
    for col in preferred:
        if col in df.columns:
            resolved.append(col)
        elif stage == "final" and f"{col}_survey" in df.columns:
            resolved.append(f"{col}_survey")
    return resolved


def collect_stage_stats(
    stage_data: dict[str, dict[str, pd.DataFrame]],
    *,
    item_col: str,
    patient_col: str | None,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, splits in stage_data.items():
        for split, df in splits.items():
            class_cols = resolve_class_cols(df, metadata, stage, split)
            entry: dict[str, Any] = {
                "stage": stage,
                "split": split,
                "row_count": int(len(df)),
                "unique_items": int(df[item_col].nunique(dropna=True)) if item_col in df.columns else None,
                "unique_patients": None,
                "class_column_count": len(class_cols),
                "missing_item_count": int(df[item_col].isna().sum()) if item_col in df.columns else None,
            }
            if patient_col and patient_col in df.columns:
                entry["unique_patients"] = int(df[patient_col].nunique(dropna=True))
            rows.append(entry)
    return rows


def item_set(df: pd.DataFrame, item_col: str, *, raw_mode: bool) -> set[str]:
    if item_col not in df.columns:
        return set()
    if raw_mode:
        values = df[item_col].map(trim_raw_item)
    else:
        values = df[item_col].map(lambda v: None if pd.isna(v) else str(v).strip())
    return {v for v in values.dropna().astype(str) if v != ""}


def prepare_stage_data(
    *,
    raw_splits: dict[str, pd.DataFrame],
    standardized_splits: dict[str, pd.DataFrame],
    post_physician_splits: dict[str, pd.DataFrame],
    final_splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[str], list[str]]:
    """Build stage-keyed DataFrames from preprocessing snapshots.

    Creates copies of all input splits, adds a 'combined' entry for all non-final
    stages, and patches the 'combined' split entries in a shallow copy of
    metadata['stage_class_cols'] (the caller's dict is not mutated).
    """
    stage_data: dict[str, dict[str, pd.DataFrame]] = {
        "raw": copy_splits(raw_splits),
        "standardized": copy_splits(standardized_splits),
        "post_physician_merge": copy_splits(post_physician_splits),
        "final": {"combined": combine_splits({k: v for k, v in final_splits.items() if k != "combined"})}
        if "combined" not in final_splits
        else {"combined": final_splits["combined"].copy(deep=True)},
    }

    selected_stages = list(FIXED_STAGES)
    for stage in selected_stages[:-1]:
        stage_data[stage]["combined"] = combine_splits(stage_data[stage])

    if "stage_class_cols" in metadata and isinstance(metadata["stage_class_cols"], dict):
        # Work on a shallow copy so we don't mutate the caller's dict.
        patched_stage_class_cols: dict[str, Any] = dict(metadata["stage_class_cols"])
        for stage in selected_stages:
            if stage == "final":
                combined_df = stage_data[stage]["combined"]
                configured_cols = list(metadata.get("final_class_cols") or [])
                stage_entry = dict(patched_stage_class_cols.get(stage) or {})
                stage_entry["combined"] = [col for col in configured_cols if col in combined_df.columns]
                patched_stage_class_cols[stage] = stage_entry
                continue
            if stage in patched_stage_class_cols and "combined" in stage_data.get(stage, {}):
                combined_df = stage_data[stage]["combined"]
                configured_cols = list(metadata.get("raw_class_cols") or [])
                stage_entry = dict(patched_stage_class_cols.get(stage) or {})
                stage_entry["combined"] = [col for col in configured_cols if col in combined_df.columns]
                patched_stage_class_cols[stage] = stage_entry
        metadata["stage_class_cols"] = patched_stage_class_cols

    split_names = ["train", "test", "interview", "combined"]  # PAIRED-CONTEXT SCHEMA
    return stage_data, selected_stages, split_names
