"""
Hypothesis summary and model comparison summary generation.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_AGGREGATE_LABEL = "aggregate"


def make_hypothesis_summary(
    h1_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    h3_corr_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Create one-row-per-model hypothesis summary with interpretation flags.

    Parameters
    ----------
    h1_df, h2_df, h3_corr_df, h4_df:
        Concatenated results across all models from compute_h1/h2/h3/h4.
    cells_df:
        Paired cell deltas (all models).
    failed_df:
        Failed/missing rows from validate_inputs.
    epsilon:
        Context-sensitivity threshold.

    Returns
    -------
    pd.DataFrame
        One row per model with all key aggregate statistics and
        ``interpretation_flag``.
    """
    models = sorted(cells_df["model"].unique())
    patient_col = "patient" if "patient" in cells_df.columns else "patient_id"

    rows: list[dict] = []
    for model in models:
        mc = cells_df[cells_df["model"] == model]
        mf = failed_df if "model" not in failed_df.columns else failed_df[failed_df.get("model", pd.Series(dtype=str)) == model]

        # Counts
        n_patients = int(mc[patient_col].nunique())
        n_patient_items = int(mc.groupby([patient_col, "item_text"]).ngroups if "item_text" in mc.columns else len(mc) // mc["category"].nunique())
        n_cells = len(mc)
        n_shift = int((mc["delta_physician"] != 0.0).sum()) if "delta_physician" in mc.columns else 0

        # Failure rate
        total_possible = n_cells  # rough estimate
        n_failed = len(mf)
        failure_rate = float(n_failed / max(total_possible + n_failed, 1))

        def _agg(df: pd.DataFrame, col: str, default=float("nan")):
            if df is None or df.empty or col not in df.columns:
                return default
            agg_rows = df[(df["model"] == model) & (df["category"] == _AGGREGATE_LABEL)]
            if agg_rows.empty:
                return default
            return float(agg_rows.iloc[0][col])

        h1_c = h1_df if h1_df is not None else pd.DataFrame()
        h2_c = h2_df if h2_df is not None else pd.DataFrame()
        h3_c = h3_corr_df if h3_corr_df is not None else pd.DataFrame()
        h4_c = h4_df if h4_df is not None else pd.DataFrame()

        h3_model = h3_c[h3_c["model"] == model] if not h3_c.empty and "model" in h3_c.columns else pd.DataFrame()

        H1_mean_abs_delta_correct = _agg(h1_c, "mean_abs_delta_correct")
        H1_mean_abs_delta_shuffled = _agg(h1_c, "mean_abs_delta_shuffled")
        H1_proportion_changed_correct = _agg(h1_c, "proportion_changed_correct")
        H1_proportion_changed_shuffled = _agg(h1_c, "proportion_changed_shuffled")

        H2_mean_alignment_correct = _agg(h2_c, "mean_alignment_correct")
        H2_mean_alignment_shuffled = _agg(h2_c, "mean_alignment_shuffled")
        H2_sign_agreement_correct = _agg(h2_c, "sign_agreement_correct")
        H2_sign_agreement_shuffled = _agg(h2_c, "sign_agreement_shuffled")

        def _h3val(col):
            if h3_model.empty or col not in h3_model.columns:
                return float("nan")
            return float(h3_model.iloc[0][col])

        H3_pearson_correct = _h3val("pearson_correct")
        H3_spearman_correct = _h3val("spearman_correct")
        H3_pearson_shuffled = _h3val("pearson_shuffled")
        H3_spearman_shuffled = _h3val("spearman_shuffled")

        H4_mean_alignment_difference = _agg(h4_c, "mean_alignment_difference")
        H4_ci_low = _agg(h4_c, "ci_low_difference")
        H4_ci_high = _agg(h4_c, "ci_high_difference")
        H4_p_value = _agg(h4_c, "paired_permutation_p")

        # Interpretation flag
        interpretation_flag = _assign_flag(
            H1_mean_abs_delta_correct,
            H4_ci_low,
            H4_mean_alignment_difference,
            epsilon,
        )

        rows.append(
            {
                "model": model,
                "n_patients": n_patients,
                "n_patient_items": n_patient_items,
                "n_cells": n_cells,
                "n_physician_shift_cells": n_shift,
                "H1_mean_abs_delta_correct": H1_mean_abs_delta_correct,
                "H1_mean_abs_delta_shuffled": H1_mean_abs_delta_shuffled,
                "H1_proportion_changed_correct": H1_proportion_changed_correct,
                "H1_proportion_changed_shuffled": H1_proportion_changed_shuffled,
                "H2_mean_alignment_correct": H2_mean_alignment_correct,
                "H2_mean_alignment_shuffled": H2_mean_alignment_shuffled,
                "H2_sign_agreement_correct": H2_sign_agreement_correct,
                "H2_sign_agreement_shuffled": H2_sign_agreement_shuffled,
                "H3_pearson_correct": H3_pearson_correct,
                "H3_spearman_correct": H3_spearman_correct,
                "H3_pearson_shuffled": H3_pearson_shuffled,
                "H3_spearman_shuffled": H3_spearman_shuffled,
                "H4_mean_alignment_difference": H4_mean_alignment_difference,
                "H4_ci_low": H4_ci_low,
                "H4_ci_high": H4_ci_high,
                "H4_p_value": H4_p_value,
                "failure_rate": failure_rate,
                "interpretation_flag": interpretation_flag,
            }
        )

    return pd.DataFrame(rows)


def _assign_flag(
    h1_mean_abs_delta_correct: float,
    h4_ci_low: float,
    h4_mean_diff: float,
    epsilon: float,
) -> str:
    """Assign interpretation flag based on H1 and H4 results."""
    def _is_nan(v: float) -> bool:
        try:
            return float(v) != float(v)  # NaN check
        except (TypeError, ValueError):
            return True

    if _is_nan(h1_mean_abs_delta_correct):
        return "failed_validation"
    if h1_mean_abs_delta_correct <= epsilon:
        return "no_context_sensitivity"
    if _is_nan(h4_ci_low) or _is_nan(h4_mean_diff):
        return "failed_validation"
    if h4_ci_low <= 0:
        return "nonspecific_context_inflation"
    if h4_mean_diff < 0.05:
        return "modest_patient_specific_alignment"
    return "strong_patient_specific_alignment"


def make_model_comparison_summary(
    hypothesis_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create model comparison summary sorted by H4 alignment difference.

    Parameters
    ----------
    hypothesis_summary:
        Output of ``make_hypothesis_summary``.

    Returns
    -------
    pd.DataFrame
        Models ranked by H4_mean_alignment_difference descending.
    """
    cols = [
        "model",
        "H1_mean_abs_delta_correct",
        "H1_mean_abs_delta_shuffled",
        "H2_mean_alignment_correct",
        "H2_mean_alignment_shuffled",
        "H3_pearson_correct",
        "H3_spearman_correct",
        "H4_mean_alignment_difference",
        "H4_ci_low",
        "H4_ci_high",
        "H4_p_value",
        "failure_rate",
        "interpretation_flag",
    ]
    available = [c for c in cols if c in hypothesis_summary.columns]
    df = hypothesis_summary[available].copy()

    sort_col = "H4_mean_alignment_difference"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
