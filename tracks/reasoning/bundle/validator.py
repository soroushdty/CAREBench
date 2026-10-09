"""
Input validation for the analysis bundle pipeline.

Runs 11 checks on the physician consensus and LLM score DataFrames before
analysis.  All failures are recorded; rows are never silently dropped.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


def validate_inputs(
    physician_df: pd.DataFrame,
    scores_long_df: pd.DataFrame,
    categories: List[str],
    epsilon: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate physician consensus and LLM score DataFrames.

    Runs 11 checks in sequence.  Results are collected into summary and
    failed-row DataFrames; no exception is raised on validation failures
    (the caller decides how to handle them).

    Parameters
    ----------
    physician_df:
        Long-format physician DataFrame from ``load_physician_consensus``.
        Required columns: patient_id, item_text, category,
        physician_survey_consensus, physician_interview_consensus.
    scores_long_df:
        Long-format LLM scores from ``load_llm_scores``.
        Required columns: model, patient_id, item_text, condition,
        {category columns}.
    categories:
        Canonical category names to check.
    epsilon:
        Epsilon threshold used for sensitivity checks.

    Returns
    -------
    (validation_summary, failed_rows)
        validation_summary: one row per check.
        failed_rows: rows that failed any check (with a 'failure_reason' column).
    """
    checks: list[dict] = []
    failed_rows: list[pd.DataFrame] = []

    def _record(
        check_id: int,
        description: str,
        status: str,
        count: int = 0,
        notes: str = "",
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "status": status,
                "count": count,
                "notes": notes,
            }
        )

    # ------------------------------------------------------------------
    # Check 1: All three conditions present
    # ------------------------------------------------------------------
    conditions_present = set(scores_long_df["condition"].unique())
    required_conditions = {"context_free", "correct_context", "shuffled_context"}
    missing_conds = required_conditions - conditions_present
    if missing_conds:
        _record(
            1,
            "All three LLM conditions present",
            "FAIL",
            len(missing_conds),
            f"Missing: {sorted(missing_conds)}",
        )
    else:
        _record(1, "All three LLM conditions present", "PASS", 3)

    # ------------------------------------------------------------------
    # Check 2: All required category columns exist in score data
    # ------------------------------------------------------------------
    missing_cats = [c for c in categories if c not in scores_long_df.columns]
    if missing_cats:
        _record(
            2,
            "All required category columns present in scores",
            "FAIL",
            len(missing_cats),
            f"Missing: {missing_cats}",
        )
    else:
        _record(
            2,
            "All required category columns present in scores",
            "PASS",
            len(categories),
        )

    # ------------------------------------------------------------------
    # Check 3: Category scores numeric and in [0, 1]
    # ------------------------------------------------------------------
    existing_cats = [c for c in categories if c in scores_long_df.columns]
    invalid_score_rows: list[pd.DataFrame] = []
    for cat in existing_cats:
        col = pd.to_numeric(scores_long_df[cat], errors="coerce")
        bad_mask = col.isna() | (col < 0) | (col > 1)
        if bad_mask.any():
            bad = scores_long_df[bad_mask].copy()
            bad["failure_reason"] = f"check3: invalid score in '{cat}'"
            invalid_score_rows.append(bad)
    if invalid_score_rows:
        combined = pd.concat(invalid_score_rows, ignore_index=True)
        n_bad = len(combined)
        failed_rows.append(combined)
        _record(
            3,
            "All category scores numeric and in [0, 1]",
            "FAIL",
            n_bad,
            f"{n_bad} rows with out-of-range or non-numeric scores",
        )
    else:
        _record(3, "All category scores numeric and in [0, 1]", "PASS")

    # ------------------------------------------------------------------
    # Check 4: patient×item keys matchable across all 3 LLM conditions
    # ------------------------------------------------------------------
    cond_keys: dict[str, set[tuple[str, str, str]]] = {}
    for cond in ["context_free", "correct_context", "shuffled_context"]:
        sub = scores_long_df[scores_long_df["condition"] == cond]
        cond_keys[cond] = set(
            zip(sub["model"].astype(str), sub["patient_id"].astype(str), sub["item_text"].astype(str))
        )
    all_keys = set.union(*cond_keys.values()) if cond_keys else set()
    cross_missing: list[dict] = []
    for key in all_keys:
        missing_in = [c for c, ks in cond_keys.items() if key not in ks]
        if missing_in:
            cross_missing.append(
                {
                    "model": key[0],
                    "patient_id": key[1],
                    "item_text": key[2],
                    "failure_reason": f"check4: missing in conditions {missing_in}",
                }
            )
    if cross_missing:
        bad = pd.DataFrame(cross_missing)
        failed_rows.append(bad)
        _record(
            4,
            "patient×item keys matchable across all 3 LLM conditions",
            "FAIL",
            len(cross_missing),
            f"{len(cross_missing)} model×patient×item combinations not in all conditions",
        )
    else:
        _record(
            4,
            "patient×item keys matchable across all 3 LLM conditions",
            "PASS",
            len(all_keys),
        )

    # ------------------------------------------------------------------
    # Check 5: patient×item keys matchable to dataset.xlsx
    # ------------------------------------------------------------------
    phys_keys = set(
        zip(physician_df["patient_id"].astype(str), physician_df["item_text"].astype(str))
    )
    score_pi_keys = set(
        zip(
            scores_long_df["patient_id"].astype(str),
            scores_long_df["item_text"].astype(str),
        )
    )
    unmatched_score = score_pi_keys - phys_keys
    unmatched_phys = phys_keys - score_pi_keys
    check5_fail = bool(unmatched_score or unmatched_phys)
    notes5 = ""
    if unmatched_score:
        sample = sorted(unmatched_score)[:5]
        notes5 += f"Score rows not in dataset: {sample}{'...' if len(unmatched_score) > 5 else ''}. "
    if unmatched_phys:
        sample = sorted(unmatched_phys)[:5]
        notes5 += f"Dataset rows not in scores: {sample}{'...' if len(unmatched_phys) > 5 else ''}."
    if check5_fail:
        if unmatched_score:
            bad_mask = (
                scores_long_df["patient_id"].astype(str)
                + "|||"
                + scores_long_df["item_text"].astype(str)
            ).isin({f"{p}|||{i}" for p, i in unmatched_score})
            bad = scores_long_df[bad_mask].copy()
            bad["failure_reason"] = "check5: not in dataset.xlsx"
            failed_rows.append(bad)
        _record(
            5,
            "patient×item keys matchable to dataset.xlsx",
            "FAIL",
            len(unmatched_score) + len(unmatched_phys),
            notes5.strip(),
        )
    else:
        _record(
            5,
            "patient×item keys matchable to dataset.xlsx",
            "PASS",
            len(phys_keys),
        )

    # ------------------------------------------------------------------
    # Check 6: No duplicate rows per model × patient × item × condition
    # ------------------------------------------------------------------
    dup_mask = scores_long_df.duplicated(
        subset=["model", "patient_id", "item_text", "condition"], keep=False
    )
    n_dups = dup_mask.sum()
    if n_dups:
        bad = scores_long_df[dup_mask].copy()
        bad["failure_reason"] = "check6: duplicate model×patient×item×condition"
        failed_rows.append(bad)
        _record(
            6,
            "No duplicate rows per model×patient×item×condition",
            "FAIL",
            n_dups,
            f"{n_dups} duplicate rows",
        )
    else:
        _record(6, "No duplicate rows per model×patient×item×condition", "PASS")

    # ------------------------------------------------------------------
    # Check 7: Each model has equal row counts across conditions
    # ------------------------------------------------------------------
    model_cond_counts = (
        scores_long_df.groupby(["model", "condition"])
        .size()
        .reset_index(name="n_rows")
    )
    unequal_models: list[str] = []
    for model, grp in model_cond_counts.groupby("model"):
        if grp["n_rows"].nunique() > 1:
            unequal_models.append(str(model))
    if unequal_models:
        _record(
            7,
            "Each model has equal row counts across conditions",
            "FAIL",
            len(unequal_models),
            f"Unequal counts for models: {unequal_models}",
        )
    else:
        _record(
            7,
            "Each model has equal row counts across conditions",
            "PASS",
            scores_long_df["model"].nunique(),
        )

    # ------------------------------------------------------------------
    # Check 8: Shuffled context patient ≠ true patient (if column exists)
    # ------------------------------------------------------------------
    if "context_patient_id" in scores_long_df.columns:
        shuffled_rows = scores_long_df[
            scores_long_df["condition"] == "shuffled_context"
        ]
        leakage_mask = (
            shuffled_rows["patient_id"].astype(str)
            == shuffled_rows["context_patient_id"].astype(str)
        )
        n_leakage = leakage_mask.sum()
        if n_leakage:
            bad = shuffled_rows[leakage_mask].copy()
            bad["failure_reason"] = "check8: shuffled context is same patient"
            failed_rows.append(bad)
            _record(
                8,
                "Shuffled context patient ≠ true patient",
                "FAIL",
                n_leakage,
                f"{n_leakage} rows where shuffled context is the same patient",
            )
        else:
            _record(
                8,
                "Shuffled context patient ≠ true patient",
                "PASS",
                len(shuffled_rows),
            )
    else:
        _record(
            8,
            "Shuffled context patient ≠ true patient",
            "SKIP",
            0,
            "context_patient_id column not present",
        )

    # ------------------------------------------------------------------
    # Check 9: At least 2 unique patients
    # ------------------------------------------------------------------
    n_patients = scores_long_df["patient_id"].nunique()
    if n_patients < 2:
        _record(
            9,
            "At least 2 unique patients present",
            "FAIL",
            n_patients,
            f"Only {n_patients} unique patient(s) found",
        )
    else:
        _record(9, "At least 2 unique patients present", "PASS", n_patients)

    # ------------------------------------------------------------------
    # Check 10: delta_llm_correct not identically zero for all rows
    # ------------------------------------------------------------------
    if existing_cats and "context_free" in conditions_present and "correct_context" in conditions_present:
        # Deduplicate before computing check 10 to avoid shape mismatches
        cf = (
            scores_long_df[scores_long_df["condition"] == "context_free"]
            .drop_duplicates(subset=["model", "patient_id", "item_text"])
            .set_index(["model", "patient_id", "item_text"])
        )
        cc = (
            scores_long_df[scores_long_df["condition"] == "correct_context"]
            .drop_duplicates(subset=["model", "patient_id", "item_text"])
            .set_index(["model", "patient_id", "item_text"])
        )
        common_idx = cf.index.intersection(cc.index)
        if len(common_idx) > 0:
            cf_vals = cf.loc[common_idx, existing_cats].to_numpy(dtype=float)
            cc_vals = cc.loc[common_idx, existing_cats].to_numpy(dtype=float)
            delta = cc_vals - cf_vals
            if np.all(np.abs(delta) < 1e-12):
                _record(
                    10,
                    "delta_llm_correct not identically zero for all rows",
                    "WARN",
                    0,
                    "All delta_llm_correct values are exactly zero — model may not be context-sensitive",
                )
            else:
                n_nonzero = int(np.sum(np.abs(delta) > epsilon))
                _record(
                    10,
                    "delta_llm_correct not identically zero for all rows",
                    "PASS",
                    n_nonzero,
                    f"{n_nonzero} cells with |delta_llm_correct| > epsilon",
                )
        else:
            _record(10, "delta_llm_correct not identically zero for all rows", "SKIP", 0, "No matching keys")
    else:
        _record(10, "delta_llm_correct not identically zero for all rows", "SKIP", 0, "Cannot check without both conditions")

    # ------------------------------------------------------------------
    # Check 11: Failed/missing rows tracked (summary check)
    # ------------------------------------------------------------------
    total_failed = sum(len(f) for f in failed_rows)
    if total_failed > 0:
        _record(
            11,
            "Failed and missing rows tracked (not silently dropped)",
            "INFO",
            total_failed,
            f"{total_failed} total rows with issues across all checks",
        )
    else:
        _record(
            11,
            "Failed and missing rows tracked (not silently dropped)",
            "PASS",
            0,
            "No failed rows detected",
        )

    validation_summary = pd.DataFrame(checks)

    if failed_rows:
        failed_df = pd.concat(failed_rows, ignore_index=True)
        # Deduplicate if a row appeared in multiple checks
        dup_cols = [c for c in ["model", "patient_id", "item_text", "condition"] if c in failed_df.columns]
        failed_df = failed_df.drop_duplicates(subset=dup_cols + ["failure_reason"])
    else:
        failed_df = pd.DataFrame(
            columns=["model", "patient_id", "item_text", "condition", "failure_reason"]
        )

    n_fails = (validation_summary["status"] == "FAIL").sum()
    n_warns = (validation_summary["status"] == "WARN").sum()
    logger.info(
        "Validation complete: %d FAIL, %d WARN, %d failed rows",
        n_fails,
        n_warns,
        len(failed_df),
    )
    return validation_summary, failed_df
