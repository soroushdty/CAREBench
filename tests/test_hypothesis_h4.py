"""Tests for assay.bundle.hypotheses.compute_h4."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.hypotheses import compute_h4

CATS = ["behavioral_health", "diagnoses", "disabilities"]


def _make_cells(
    delta_physician: float = 0.3,
    delta_correct: float = 0.2,
    delta_shuffled: float = 0.05,
    n_patients: int = 3,
    model: str = "m",
) -> pd.DataFrame:
    rows = []
    for pid in [f"P{i}" for i in range(n_patients)]:
        for item in ["Item0", "Item1"]:
            for cat in CATS:
                sign_dp = np.sign(delta_physician) if delta_physician != 0 else 0.0
                ac = sign_dp * delta_correct
                as_ = sign_dp * delta_shuffled
                rows.append(
                    {
                        "model": model,
                        "patient": pid,
                        "item_text": item,
                        "category": cat,
                        "delta_physician": delta_physician,
                        "delta_llm_correct": delta_correct,
                        "delta_llm_shuffled": delta_shuffled,
                        "alignment_correct": ac,
                        "alignment_shuffled": as_,
                        "alignment_difference": ac - as_,
                        "physician_shift_indicator": delta_physician != 0,
                    }
                )
    return pd.DataFrame(rows)


class TestComputeH4:
    def test_correct_aligns_better_than_shuffled(self):
        """When correct > shuffled alignment, mean_alignment_difference > 0."""
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2, delta_shuffled=0.05)
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        agg = h4[h4["category"] == "aggregate"].iloc[0]
        assert agg["mean_alignment_difference"] > 0

    def test_equal_correct_and_shuffled_gives_zero_diff(self):
        """When correct context = shuffled context, mean_alignment_difference ≈ 0."""
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2, delta_shuffled=0.2)
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        agg = h4[h4["category"] == "aggregate"].iloc[0]
        assert abs(agg["mean_alignment_difference"]) < 1e-9

    def test_no_physician_shifts_returns_nan(self):
        """When delta_physician=0, H4 returns NaN."""
        cells = _make_cells(delta_physician=0.0)
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        agg = h4[h4["category"] == "aggregate"].iloc[0]
        assert math.isnan(agg["mean_alignment_difference"])

    def test_ci_present_and_ordered(self):
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2, delta_shuffled=0.05)
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=100, n_permutations=100, seed=42)
        agg = h4[h4["category"] == "aggregate"].iloc[0]
        lo = agg["ci_low_difference"]
        hi = agg["ci_high_difference"]
        assert not math.isnan(lo) and not math.isnan(hi)
        # CI should bracket or equal the point estimate
        assert lo <= agg["mean_alignment_difference"] + 1e-9
        assert hi >= agg["mean_alignment_difference"] - 1e-9

    def test_p_value_in_range(self):
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2, delta_shuffled=0.05)
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        agg = h4[h4["category"] == "aggregate"].iloc[0]
        p = agg["paired_permutation_p"]
        assert 0.0 <= p <= 1.0

    def test_per_category_rows_present(self):
        cells = _make_cells()
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        for cat in CATS:
            assert cat in h4["category"].values

    def test_aggregate_row_present(self):
        cells = _make_cells()
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        assert "aggregate" in h4["category"].values

    def test_required_columns_present(self):
        cells = _make_cells()
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        required = [
            "model", "category", "n_patients", "n_shift_cells",
            "mean_alignment_correct", "mean_alignment_shuffled",
            "mean_alignment_difference", "ci_low_difference",
            "ci_high_difference", "paired_permutation_p",
        ]
        for col in required:
            assert col in h4.columns, f"Missing: {col}"

    def test_shuffled_worse_than_correct_scenario(self):
        """When shuffled aligns in the wrong direction, mean_diff is positive and large."""
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2, delta_shuffled=-0.2)
        h4 = compute_h4(cells, "m", CATS, n_bootstrap=50, n_permutations=100, seed=0)
        agg = h4[h4["category"] == "aggregate"].iloc[0]
        # alignment_correct = 0.2, alignment_shuffled = -0.2 → diff = 0.4
        assert agg["mean_alignment_difference"] == pytest.approx(0.4, abs=1e-9)
