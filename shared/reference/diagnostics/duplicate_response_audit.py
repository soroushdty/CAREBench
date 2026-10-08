"""Duplicate response audit: merge test and interview reference observer data.

Terminology mapping (canonical → dataset-specific term):
    reference_context_free    ← physician_survey / survey  # PAIRED-CONTEXT SCHEMA
    reference_correct_context ← physician_interview / interview  # PAIRED-CONTEXT SCHEMA
    context_entity_id         ← Patient  # PAIRED-CONTEXT SCHEMA
    task_instance             ← Item     # PAIRED-CONTEXT SCHEMA

This module contains audit and diagnostic logic for the two-condition reference design
(context-free survey vs. correct-context interview). It is more dataset-specific than
the aggregation primitive in paired_reference_mean.py and may require adaptation for
other datasets.
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _merge_test_interview(
    test_df: pd.DataFrame,
    interview_df: pd.DataFrame,
    *,
    patient_col: str,  # PAIRED-CONTEXT SCHEMA — canonical: context_entity_id
    item_col: str,  # PAIRED-CONTEXT SCHEMA — canonical: task_instance
    class_cols: list[str],
) -> pd.DataFrame:
    """Merge collapsed test (reference_context_free) with collapsed interview (reference_correct_context).

    Performs a LEFT OUTER JOIN on (context_entity_id, task_instance).
    Interview label columns are renamed to ``{col}_interview``.
    The interview ``physician_ids`` column is renamed to ``physician_ids_interview``.

    * Test rows with no matching interview row → NaN in all *_interview cols.
    * Interview rows with no matching test row → DROPPED; WARNING emitted.
    """
    merge_cols = [patient_col, item_col]

    # Identify which interview rows have no matching test row.
    test_keys = set(
        zip(test_df[patient_col].tolist(), test_df[item_col].tolist())
    )
    interview_keys = set(
        zip(interview_df[patient_col].tolist(), interview_df[item_col].tolist())
    )
    dropped_keys = interview_keys - test_keys
    if dropped_keys:
        msg = (
            f"WARNING: {len(dropped_keys)} collapsed interview row(s) dropped "
            f"(no corresponding test row): {sorted(dropped_keys)}. Continuing."
        )
        logger.warning(msg)

    # Build interview subset with renamed columns.
    interview_class_cols = [c for c in class_cols if c in interview_df.columns]
    rename_map: dict[str, str] = {col: f"{col}_interview" for col in interview_class_cols}
    if "physician_ids" in interview_df.columns:
        rename_map["physician_ids"] = "physician_ids_interview"

    keep_cols = merge_cols + [c for c in interview_df.columns if c in rename_map]
    interview_subset = interview_df[keep_cols].rename(columns=rename_map)

    merged = test_df.merge(interview_subset, on=merge_cols, how="left")
    return merged.reset_index(drop=True)
