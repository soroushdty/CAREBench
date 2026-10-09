"""
Paired cell delta builder for the analysis bundle pipeline.

Merges physician consensus and LLM scores into one long-format DataFrame with
one row per model × patient × item × category, then computes all deltas,
alignments, and indicator columns.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONDITIONS = ("context_free", "correct_context", "shuffled_context")


def build_paired_cell_deltas(
    physician_df: pd.DataFrame,
    scores_long_df: pd.DataFrame,
    categories: List[str],
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Build one row per model × patient × item × category with all deltas.

    Parameters
    ----------
    physician_df:
        Long-format DataFrame from ``load_physician_consensus``.
        Columns: patient_id, item_text, item_id, category,
        physician_survey_consensus, physician_interview_consensus,
        delta_physician.
    scores_long_df:
        Long-format LLM scores from ``load_llm_scores``.
        Columns: model, patient_id, item_text, condition, {categories}.
    categories:
        Canonical category names.
    epsilon:
        Threshold for LLM-changed indicators.

    Returns
    -------
    pd.DataFrame
        One row per model × patient × item × category.  Unmatched rows are
        logged and excluded (they appear in validation failed_rows instead).
    """
    cats_present = [c for c in categories if c in scores_long_df.columns]
    if not cats_present:
        raise ValueError("No category columns found in scores_long_df.")

    # ------------------------------------------------------------------
    # 1. Pivot LLM scores wide: one row per model × patient × item, with
    #    columns {cat}_{condition} for each (category, condition) pair.
    # ------------------------------------------------------------------
    pivot_parts: list[pd.DataFrame] = []
    for cond in _CONDITIONS:
        sub = scores_long_df[scores_long_df["condition"] == cond][
            ["model", "patient_id", "item_text"] + cats_present
        ].copy()
        rename = {cat: f"{cat}__{cond}" for cat in cats_present}
        sub = sub.rename(columns=rename)
        pivot_parts.append(sub)

    if not pivot_parts:
        raise ValueError("No condition rows found in scores_long_df.")

    scores_wide = pivot_parts[0]
    for part in pivot_parts[1:]:
        scores_wide = scores_wide.merge(
            part, on=["model", "patient_id", "item_text"], how="outer"
        )

    # ------------------------------------------------------------------
    # 2. Melt scores_wide to long: one row per model × patient × item × category
    # ------------------------------------------------------------------
    id_vars = ["model", "patient_id", "item_text"]
    score_cols = [c for c in scores_wide.columns if c not in id_vars]

    # Build a structured melt: extract (cat, condition) from column name
    records: list[dict] = []
    scores_wide = scores_wide.reset_index(drop=True)
    for _, row in scores_wide.iterrows():
        model = row["model"]
        pid = str(row["patient_id"])
        itext = str(row["item_text"])
        for cat in cats_present:
            cf_col = f"{cat}__context_free"
            cc_col = f"{cat}__correct_context"
            sc_col = f"{cat}__shuffled_context"
            records.append(
                {
                    "model": model,
                    "patient_id": pid,
                    "item_text": itext,
                    "category": cat,
                    "llm_context_free": row.get(cf_col, np.nan),
                    "llm_correct_context": row.get(cc_col, np.nan),
                    "llm_shuffled_context": row.get(sc_col, np.nan),
                }
            )

    scores_long_cat = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 3. Merge with physician_df on (patient_id, item_text, category)
    # ------------------------------------------------------------------
    phys_cols = [
        "patient_id",
        "item_text",
        "item_id",
        "category",
        "physician_survey_consensus",
        "physician_interview_consensus",
        "delta_physician",
    ]
    phys_sub = physician_df[phys_cols].copy()
    phys_sub["patient_id"] = phys_sub["patient_id"].astype(str)
    phys_sub["item_text"] = phys_sub["item_text"].astype(str)

    merged = scores_long_cat.merge(
        phys_sub, on=["patient_id", "item_text", "category"], how="inner"
    )

    n_unmatched = len(scores_long_cat) - len(merged)
    if n_unmatched > 0:
        logger.warning(
            "%d score rows had no matching physician consensus row and were excluded.",
            n_unmatched,
        )

    if merged.empty:
        raise ValueError(
            "No rows remain after merging LLM scores with physician consensus. "
            "Check that patient_id and item_text keys match between files."
        )

    # ------------------------------------------------------------------
    # 4. Compute deltas, alignments, indicators
    # ------------------------------------------------------------------
    merged["delta_physician"] = merged["delta_physician"].astype(float)
    merged["llm_context_free"] = pd.to_numeric(merged["llm_context_free"], errors="coerce")
    merged["llm_correct_context"] = pd.to_numeric(merged["llm_correct_context"], errors="coerce")
    merged["llm_shuffled_context"] = pd.to_numeric(merged["llm_shuffled_context"], errors="coerce")

    merged["delta_llm_correct"] = merged["llm_correct_context"] - merged["llm_context_free"]
    merged["delta_llm_shuffled"] = merged["llm_shuffled_context"] - merged["llm_context_free"]

    sign_dp = np.sign(merged["delta_physician"].to_numpy())
    merged["alignment_correct"] = sign_dp * merged["delta_llm_correct"].to_numpy()
    merged["alignment_shuffled"] = sign_dp * merged["delta_llm_shuffled"].to_numpy()
    merged["alignment_difference"] = merged["alignment_correct"] - merged["alignment_shuffled"]

    merged["physician_shift_indicator"] = merged["delta_physician"] != 0.0
    merged["llm_correct_changed_indicator"] = merged["delta_llm_correct"].abs() > epsilon
    merged["llm_shuffled_changed_indicator"] = merged["delta_llm_shuffled"].abs() > epsilon

    # ------------------------------------------------------------------
    # 5. Final column order
    # ------------------------------------------------------------------
    out_cols = [
        "model",
        "patient_id",
        "item_id",
        "item_text",
        "category",
        "physician_survey_consensus",
        "physician_interview_consensus",
        "delta_physician",
        "llm_context_free",
        "llm_correct_context",
        "llm_shuffled_context",
        "delta_llm_correct",
        "delta_llm_shuffled",
        "alignment_correct",
        "alignment_shuffled",
        "alignment_difference",
        "physician_shift_indicator",
        "llm_correct_changed_indicator",
        "llm_shuffled_changed_indicator",
    ]
    # Rename patient_id → patient for output spec
    merged = merged.rename(columns={"patient_id": "patient"})
    out_cols = [c.replace("patient_id", "patient") for c in out_cols]

    available = [c for c in out_cols if c in merged.columns]
    result = merged[available].reset_index(drop=True)

    logger.info(
        "Built paired cell deltas: %d rows (%d models, %d patients, %d categories)",
        len(result),
        result["model"].nunique(),
        result["patient"].nunique(),
        result["category"].nunique(),
    )
    return result
