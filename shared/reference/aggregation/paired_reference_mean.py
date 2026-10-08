"""Paired reference observer aggregation: collapse per-observer rows into one row per (context_entity_id, task_instance).

Terminology mapping (canonical → dataset-specific term):
    reference_observer  ← physician / human rater
    context_entity_id   ← Patient  # PAIRED-CONTEXT SCHEMA
    task_instance       ← Item     # PAIRED-CONTEXT SCHEMA

This module is part of the shared reference layer and is reusable by any evaluation track.
"""

from __future__ import annotations

from typing import Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _aggregate_physicians(
    df: pd.DataFrame,
    *,
    patient_col: str,  # PAIRED-CONTEXT SCHEMA — canonical: context_entity_id
    physician_col: str | None,  # PAIRED-CONTEXT SCHEMA — canonical: reference_observer
    item_col: str,  # PAIRED-CONTEXT SCHEMA — canonical: task_instance
    class_cols: list[str],
    physician_count: int = 2,
    mismatch_error: bool = False,
) -> pd.DataFrame:
    """Collapse per-reference-observer rows into one row per (context_entity_id, task_instance).

    For each label column the value is the arithmetic mean of the original
    binary {0, 1} reference observer labels.

    **Mismatch handling** (when a group's row count ≠ ``physician_count``):

    * Condition A – resolvable duplicate mismatch:
        Every row in the group has at least one exact replica in the group,
        AND the number of *unique* rows equals ``physician_count``.
        Action: deduplicate (keep lowest-index occurrence), emit a WARNING.

    * Condition B – unresolvable mismatch:
        Condition A criteria are not met.
        Action: if ``mismatch_error`` is ``True``, raise a ``ValueError``
        (logged to logger + printed to stdout); otherwise skip (drop) the
        associated rows and emit a WARNING.

    The scalar reference_observer column is dropped; a ``physician_ids`` column (sorted
    list of observer IDs from the group) is added for audit purposes.
    """
    group_cols = [patient_col, item_col]
    present_class_cols = [c for c in class_cols if c in df.columns]

    # ── Mismatch detection and resolution ───────────────────────────────────
    sizes = df.groupby(group_cols, dropna=False).size()
    rows_to_drop: list[int] = []  # original integer positions to drop

    for group_key, count in sizes.items():
        if count == physician_count:
            continue

        patient_val, item_val = group_key

        # Locate the group rows (by original DataFrame position).
        mask = (df[patient_col] == patient_val) & (df[item_col] == item_val)
        group_df = df[mask]
        group_indices = group_df.index.tolist()

        deduplicated = group_df.drop_duplicates()
        unique_count = len(deduplicated)
        all_rows_have_replica = all(group_df.duplicated(keep=False).loc[group_indices])

        if all_rows_have_replica and unique_count == physician_count:
            # Condition A: invariant satisfied — safe to deduplicate.
            dup_mask_in_group = group_df.duplicated(keep="first")
            dropped_indices = [idx for idx, is_dup in zip(group_indices, dup_mask_in_group.tolist()) if is_dup]
            num_dropped = len(dropped_indices)
            rows_to_drop.extend(dropped_indices)
            msg = (
                f'[INFO] Reference observer count mismatch: context_entity_id={patient_val!r}, '
                f'task_instance={item_val!r}, Total count={count} '
                f'(Unique: {unique_count}) -> {num_dropped} ROWS DROPPED.'
            )
            logger.info(msg)
        else:
            # Condition B: invariant not satisfied.
            msg = (
                f"[{'ERROR' if mismatch_error else 'WARNING'}] Reference observer count mismatch: "
                f"context_entity_id={patient_val!r}, "
                f"task_instance={item_val!r}, actual count={count} (expected {physician_count}). "
                f"Unique rows={unique_count}. Cannot resolve mismatch. "
                + ("Halting." if mismatch_error else "Skipping associated rows.")
            )
            if mismatch_error:
                logger.error(msg)
                raise ValueError(msg)
            else:
                rows_to_drop.extend(group_indices)
                logger.warning(msg)

    # Drop duplicate rows identified in Condition A resolutions.
    if rows_to_drop:
        df = df.drop(index=rows_to_drop).reset_index(drop=True)

    # ── Aggregation ──────────────────────────────────────────────────────────
    skip_cols = set(group_cols + present_class_cols)
    if physician_col and physician_col in df.columns:
        skip_cols.add(physician_col)
    extra_cols = [c for c in df.columns if c not in skip_cols]

    agg_specs: dict[str, Any] = {}
    for col in present_class_cols:
        agg_specs[col] = "mean"
    for col in extra_cols:
        agg_specs[col] = "first"

    grouped = df.groupby(group_cols, dropna=False).agg(agg_specs).reset_index()

    if physician_col and physician_col in df.columns:
        physician_ids_series = (
            df.groupby(group_cols, dropna=False)[physician_col]
            .agg(lambda x: sorted(int(v) for v in x.dropna().unique()))
            .rename("physician_ids")
            .reset_index()
        )
        grouped = grouped.merge(physician_ids_series, on=group_cols, how="left")

    return grouped.reset_index(drop=True)
