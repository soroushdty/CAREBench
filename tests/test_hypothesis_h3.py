"""Tests for tracks.reasoning.bundle.hypotheses.compute_h3."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.hypotheses import compute_h3

CATS = [
    "behavioral_health", "diagnoses", "disabilities", "infectious_diseases",
    "genetics", "medications", "sexual_reproductive_health",
    "social_determinants_of_health", "violence", "other",
]


def _make_cells_with_category_pattern(
    phys_deltas: dict[str, float],
    correct_deltas: dict[str, float],
    shuffled_deltas: dict[str, float],
    n_patients: int = 3,
    n_items: int = 2,
    model: str = "m",
) -> pd.DataFrame:
    """Build cells_df where per-category mean deltas are set by input dicts."""
    rows = []
    for pid in [f"P{i}" for i in range(n_patients)]:
        for item in [f"I{j}" for j in range(n_items)]:
            for cat in CATS:
                dp = phys_deltas.get(cat, 0.0)
                dc = correct_deltas.get(cat, 0.0)
                ds = shuffled_deltas.get(cat, 0.0)
                rows.append(
                    {
                        "model": model,
                        "patient": pid,
                        "item_text": item,
                        "category": cat,
                        "delta_physician": dp,
                        "delta_llm_correct": dc,
                        "delta_llm_shuffled": ds,
                        "alignment_correct": np.sign(dp) * dc if dp != 0 else 0.0,
                        "alignment_shuffled": np.sign(dp) * ds if dp != 0 else 0.0,
                        "alignment_difference": (
                            np.sign(dp) * dc - np.sign(dp) * ds if dp != 0 else 0.0
                        ),
                        "physician_shift_indicator": dp != 0,
                    }
                )
    return pd.DataFrame(rows)


def _uniform_cat_deltas(val: float) -> dict[str, float]:
    return {cat: val for cat in CATS}


class TestComputeH3:
    def test_perfect_positive_correlation_correct(self):
        """When LLM correct deltas are proportional to physician deltas, Pearson ≈ 1."""
        phys_d = {cat: (i + 1) * 0.05 for i, cat in enumerate(CATS)}
        # LLM correct = physician × 2 → perfect correlation
        correct_d = {cat: v * 2 for cat, v in phys_d.items()}
        shuffled_d = _uniform_cat_deltas(0.0)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        _, corr = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        assert corr.iloc[0]["pearson_correct"] == pytest.approx(1.0, abs=1e-6)

    def test_correct_correlation_exceeds_shuffled_when_signal_present(self):
        """When correct context tracks physician shifts, pearson_correct > pearson_shuffled."""
        phys_d = {cat: (i + 1) * 0.04 for i, cat in enumerate(CATS)}
        correct_d = {cat: v * 1.5 for cat, v in phys_d.items()}
        # Shuffled has random / reversed pattern
        shuffled_d = {cat: -v * 0.5 for cat, v in phys_d.items()}
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        _, corr = compute_h3(cells, "m", CATS, n_permutations=100, seed=42)
        assert corr.iloc[0]["pearson_correct"] > corr.iloc[0]["pearson_shuffled"]

    def test_p_value_in_range(self):
        phys_d = {cat: (i + 1) * 0.05 for i, cat in enumerate(CATS)}
        correct_d = {cat: v * 2 for cat, v in phys_d.items()}
        shuffled_d = _uniform_cat_deltas(0.0)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        _, corr = compute_h3(cells, "m", CATS, n_permutations=200, seed=0)
        p = corr.iloc[0]["permutation_p_pearson_correct"]
        assert 0.0 <= p <= 1.0

    def test_n_categories_correct(self):
        phys_d = _uniform_cat_deltas(0.1)
        correct_d = _uniform_cat_deltas(0.15)
        shuffled_d = _uniform_cat_deltas(0.05)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        _, corr = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        assert corr.iloc[0]["n_categories"] == len(CATS)

    def test_class_level_effects_df_has_aggregate_row(self):
        phys_d = _uniform_cat_deltas(0.1)
        correct_d = _uniform_cat_deltas(0.15)
        shuffled_d = _uniform_cat_deltas(0.05)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        effects, _ = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        assert "aggregate" in effects["category"].values

    def test_class_level_effects_has_per_category_rows(self):
        phys_d = _uniform_cat_deltas(0.1)
        correct_d = _uniform_cat_deltas(0.15)
        shuffled_d = _uniform_cat_deltas(0.05)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        effects, _ = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        for cat in CATS:
            assert cat in effects["category"].values

    def test_required_correlation_columns_present(self):
        phys_d = {cat: (i + 1) * 0.05 for i, cat in enumerate(CATS)}
        correct_d = {cat: v for cat, v in phys_d.items()}
        shuffled_d = _uniform_cat_deltas(0.0)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        _, corr = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        required = [
            "model", "pearson_correct", "spearman_correct",
            "pearson_shuffled", "spearman_shuffled",
            "pearson_correct_minus_shuffled", "spearman_correct_minus_shuffled",
            "permutation_p_pearson_correct", "permutation_p_spearman_correct",
            "n_categories",
        ]
        for col in required:
            assert col in corr.columns, f"Missing: {col}"

    def test_required_effects_columns_present(self):
        phys_d = _uniform_cat_deltas(0.1)
        correct_d = _uniform_cat_deltas(0.15)
        shuffled_d = _uniform_cat_deltas(0.05)
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        effects, _ = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        required = [
            "model", "category", "mean_delta_physician", "mean_abs_delta_physician",
            "mean_delta_llm_correct", "mean_abs_delta_llm_correct",
            "mean_delta_llm_shuffled", "mean_abs_delta_llm_shuffled",
        ]
        for col in required:
            assert col in effects.columns, f"Missing: {col}"

    def test_pearson_correct_minus_shuffled_computation(self):
        phys_d = {cat: (i + 1) * 0.05 for i, cat in enumerate(CATS)}
        correct_d = {cat: v * 2 for cat, v in phys_d.items()}
        shuffled_d = {cat: -v for cat, v in phys_d.items()}
        cells = _make_cells_with_category_pattern(phys_d, correct_d, shuffled_d)
        _, corr = compute_h3(cells, "m", CATS, n_permutations=100, seed=0)
        r = corr.iloc[0]
        assert abs(r["pearson_correct_minus_shuffled"] - (r["pearson_correct"] - r["pearson_shuffled"])) < 1e-9
