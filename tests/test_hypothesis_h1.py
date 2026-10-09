"""Tests for tracks.reasoning.bundle.hypotheses.compute_h1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.hypotheses import compute_h1

CATS = ["behavioral_health", "diagnoses", "disabilities"]


def _make_cells(
    delta_correct: float = 0.2,
    delta_shuffled: float = 0.1,
    delta_physician: float = 0.3,
    n_patients: int = 3,
    n_items: int = 2,
    model: str = "test_model",
) -> pd.DataFrame:
    rows = []
    for pid in [f"P{i}" for i in range(n_patients)]:
        for item in [f"Item{j}" for j in range(n_items)]:
            for cat in CATS:
                llm_cf = 0.4
                llm_cc = llm_cf + delta_correct
                llm_sc = llm_cf + delta_shuffled
                rows.append(
                    {
                        "model": model,
                        "patient": pid,
                        "item_text": item,
                        "item_id": 0,
                        "category": cat,
                        "physician_survey_consensus": 0.3,
                        "physician_interview_consensus": 0.3 + delta_physician,
                        "delta_physician": delta_physician,
                        "llm_context_free": llm_cf,
                        "llm_correct_context": llm_cc,
                        "llm_shuffled_context": llm_sc,
                        "delta_llm_correct": delta_correct,
                        "delta_llm_shuffled": delta_shuffled,
                        "alignment_correct": np.sign(delta_physician) * delta_correct,
                        "alignment_shuffled": np.sign(delta_physician) * delta_shuffled,
                        "alignment_difference": (
                            np.sign(delta_physician) * delta_correct
                            - np.sign(delta_physician) * delta_shuffled
                        ),
                        "physician_shift_indicator": delta_physician != 0,
                        "llm_correct_changed_indicator": abs(delta_correct) > 0.01,
                        "llm_shuffled_changed_indicator": abs(delta_shuffled) > 0.01,
                    }
                )
    return pd.DataFrame(rows)


class TestComputeH1:
    def test_mean_abs_delta_correct_matches_input(self):
        cells = _make_cells(delta_correct=0.2, delta_shuffled=0.1)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert abs(agg["mean_abs_delta_correct"] - 0.2) < 1e-9

    def test_mean_abs_delta_shuffled_matches_input(self):
        cells = _make_cells(delta_correct=0.2, delta_shuffled=0.1)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert abs(agg["mean_abs_delta_shuffled"] - 0.1) < 1e-9

    def test_zero_delta_produces_zero_mean(self):
        cells = _make_cells(delta_correct=0.0, delta_shuffled=0.0)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert agg["mean_abs_delta_correct"] == pytest.approx(0.0)

    def test_proportion_changed_correct_with_zero_delta(self):
        cells = _make_cells(delta_correct=0.0, delta_shuffled=0.0)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert agg["proportion_changed_correct"] == pytest.approx(0.0)

    def test_proportion_changed_correct_above_epsilon(self):
        cells = _make_cells(delta_correct=0.2, delta_shuffled=0.1)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert agg["proportion_changed_correct"] == pytest.approx(1.0)

    def test_ci_present_and_ordered(self):
        cells = _make_cells(delta_correct=0.2)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=100, seed=42)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        lo = agg["ci_low_correct"]
        hi = agg["ci_high_correct"]
        assert not np.isnan(lo) and not np.isnan(hi)
        assert lo <= agg["mean_abs_delta_correct"] + 1e-9
        assert hi >= agg["mean_abs_delta_correct"] - 1e-9

    def test_per_category_rows_present(self):
        cells = _make_cells()
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        for cat in CATS:
            assert cat in h1["category"].values

    def test_aggregate_row_present(self):
        cells = _make_cells()
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        assert "aggregate" in h1["category"].values

    def test_model_column_correct(self):
        cells = _make_cells(model="my_model")
        h1 = compute_h1(cells, "my_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        assert (h1["model"] == "my_model").all()

    def test_correct_and_shuffled_differ(self):
        cells = _make_cells(delta_correct=0.3, delta_shuffled=0.05)
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert agg["mean_abs_delta_correct"] > agg["mean_abs_delta_shuffled"]

    def test_required_columns_present(self):
        cells = _make_cells()
        h1 = compute_h1(cells, "test_model", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        required = [
            "model", "category", "n_patients", "n_patient_items", "n_cells",
            "mean_abs_delta_correct", "ci_low_correct", "ci_high_correct",
            "mean_abs_delta_shuffled", "ci_low_shuffled", "ci_high_shuffled",
            "proportion_changed_correct", "proportion_changed_shuffled", "epsilon",
        ]
        for col in required:
            assert col in h1.columns, f"Missing: {col}"
