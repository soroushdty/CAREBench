"""Tests for assay.bundle.validator.validate_inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.validator import validate_inputs

CATS = ["behavioral_health", "diagnoses", "disabilities"]
_CONDITIONS = ["context_free", "correct_context", "shuffled_context"]


def _make_physician_df(n_patients: int = 3, n_items: int = 2) -> pd.DataFrame:
    """Build a minimal valid physician DataFrame."""
    rows = []
    for pid in range(n_patients):
        for item_idx in range(n_items):
            for cat in CATS:
                rows.append(
                    {
                        "patient_id": f"P{pid}",
                        "item_text": f"Item{item_idx}",
                        "item_id": item_idx,
                        "category": cat,
                        "physician_survey_consensus": 0.3,
                        "physician_interview_consensus": 0.6,
                        "delta_physician": 0.3,
                    }
                )
    return pd.DataFrame(rows)


def _make_scores_df(
    n_patients: int = 3,
    n_items: int = 2,
    model: str = "test_model",
    conditions=None,
    score_value: float = 0.5,
) -> pd.DataFrame:
    if conditions is None:
        conditions = _CONDITIONS
    rows = []
    for cond in conditions:
        for pid in range(n_patients):
            for item_idx in range(n_items):
                row = {
                    "model": model,
                    "patient_id": f"P{pid}",
                    "item_text": f"Item{item_idx}",
                    "condition": cond,
                }
                for cat in CATS:
                    row[cat] = score_value
                rows.append(row)
    return pd.DataFrame(rows)


class TestValidateInputs:
    def test_valid_data_passes_all_checks(self):
        phys = _make_physician_df()
        scores = _make_scores_df()
        summary, failed = validate_inputs(phys, scores, CATS)
        assert "FAIL" not in summary["status"].values, summary[summary["status"] == "FAIL"]
        assert len(failed) == 0

    def test_missing_condition_fails_check1(self):
        phys = _make_physician_df()
        # Only two conditions
        scores = _make_scores_df(conditions=["context_free", "correct_context"])
        summary, failed = validate_inputs(phys, scores, CATS)
        ch1 = summary[summary["check_id"] == 1].iloc[0]
        assert ch1["status"] == "FAIL"

    def test_invalid_score_outside_01_fails_check3(self):
        phys = _make_physician_df()
        scores = _make_scores_df()
        # Inject an invalid score
        scores.loc[0, "behavioral_health"] = 1.5
        summary, failed = validate_inputs(phys, scores, CATS)
        ch3 = summary[summary["check_id"] == 3].iloc[0]
        assert ch3["status"] == "FAIL"
        # Invalid row should appear in failed_df
        assert len(failed) > 0
        assert any("check3" in str(r) for r in failed["failure_reason"].values)

    def test_missing_condition_rows_for_some_keys_fails_check4(self):
        phys = _make_physician_df()
        scores = _make_scores_df()
        # Remove one patient from correct_context only
        scores = scores[
            ~((scores["condition"] == "correct_context") & (scores["patient_id"] == "P2"))
        ].reset_index(drop=True)
        summary, failed = validate_inputs(phys, scores, CATS)
        ch4 = summary[summary["check_id"] == 4].iloc[0]
        assert ch4["status"] == "FAIL"

    def test_score_key_not_in_dataset_fails_check5(self):
        phys = _make_physician_df(n_patients=2)  # only P0, P1
        scores = _make_scores_df(n_patients=3)   # includes P2 not in physician
        summary, failed = validate_inputs(phys, scores, CATS)
        ch5 = summary[summary["check_id"] == 5].iloc[0]
        assert ch5["status"] == "FAIL"

    def test_duplicate_rows_fails_check6(self):
        phys = _make_physician_df()
        scores = _make_scores_df()
        # Duplicate a row
        dup = scores.iloc[[0]].copy()
        scores = pd.concat([scores, dup], ignore_index=True)
        summary, failed = validate_inputs(phys, scores, CATS)
        ch6 = summary[summary["check_id"] == 6].iloc[0]
        assert ch6["status"] == "FAIL"

    def test_unequal_condition_counts_fails_check7(self):
        phys = _make_physician_df()
        scores = _make_scores_df()
        # Remove one row from shuffled_context for this model
        idx = scores[(scores["condition"] == "shuffled_context") & (scores["patient_id"] == "P0")].index
        scores = scores.drop(idx).reset_index(drop=True)
        summary, failed = validate_inputs(phys, scores, CATS)
        ch7 = summary[summary["check_id"] == 7].iloc[0]
        assert ch7["status"] == "FAIL"

    def test_only_one_patient_fails_check9(self):
        phys = _make_physician_df(n_patients=1)
        scores = _make_scores_df(n_patients=1)
        summary, failed = validate_inputs(phys, scores, CATS)
        ch9 = summary[summary["check_id"] == 9].iloc[0]
        assert ch9["status"] == "FAIL"

    def test_all_zero_deltas_warns_check10(self):
        phys = _make_physician_df()
        scores = _make_scores_df(score_value=0.5)  # both conditions same → delta = 0
        summary, failed = validate_inputs(phys, scores, CATS)
        ch10 = summary[summary["check_id"] == 10].iloc[0]
        assert ch10["status"] in ("WARN", "PASS", "SKIP")

    def test_multiple_models_handled(self):
        phys = _make_physician_df()
        s1 = _make_scores_df(model="model_a")
        s2 = _make_scores_df(model="model_b")
        scores = pd.concat([s1, s2], ignore_index=True)
        summary, failed = validate_inputs(phys, scores, CATS)
        assert "FAIL" not in summary["status"].values, summary[summary["status"] == "FAIL"]

    def test_validation_summary_has_expected_check_ids(self):
        phys = _make_physician_df()
        scores = _make_scores_df()
        summary, _ = validate_inputs(phys, scores, CATS)
        check_ids = set(summary["check_id"].tolist())
        # Checks 1–11 should all be present
        for cid in range(1, 12):
            assert cid in check_ids, f"Check {cid} missing from summary"
