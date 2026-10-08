"""Tests for assay.bundle.delta_builder.build_paired_cell_deltas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.delta_builder import build_paired_cell_deltas

CATS = ["behavioral_health", "diagnoses"]
_CONDITIONS = ["context_free", "correct_context", "shuffled_context"]


def _make_physician_df(delta: float = 0.2) -> pd.DataFrame:
    rows = []
    for pid in ["P0", "P1"]:
        for item in ["Item0", "Item1"]:
            for cat in CATS:
                rows.append(
                    {
                        "patient_id": pid,
                        "item_text": item,
                        "item_id": 0,
                        "category": cat,
                        "physician_survey_consensus": 0.3,
                        "physician_interview_consensus": 0.3 + delta,
                        "delta_physician": delta,
                    }
                )
    return pd.DataFrame(rows)


def _make_scores_df(
    cf: float = 0.4,
    cc: float = 0.6,
    sc: float = 0.5,
    model: str = "test_model",
) -> pd.DataFrame:
    rows = []
    for pid in ["P0", "P1"]:
        for item in ["Item0", "Item1"]:
            for cond, val in [("context_free", cf), ("correct_context", cc), ("shuffled_context", sc)]:
                row = {"model": model, "patient_id": pid, "item_text": item, "condition": cond}
                for cat in CATS:
                    row[cat] = val
                rows.append(row)
    return pd.DataFrame(rows)


class TestBuildPairedCellDeltas:
    def test_basic_delta_computation(self):
        """Known scores → exact delta values."""
        phys = _make_physician_df(delta=0.2)
        scores = _make_scores_df(cf=0.4, cc=0.7, sc=0.5)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)

        assert not cells.empty
        # delta_llm_correct = cc - cf = 0.7 - 0.4 = 0.3
        assert np.allclose(cells["delta_llm_correct"].unique(), [0.3], atol=1e-9)
        # delta_llm_shuffled = sc - cf = 0.5 - 0.4 = 0.1
        assert np.allclose(cells["delta_llm_shuffled"].unique(), [0.1], atol=1e-9)

    def test_alignment_correct_computation(self):
        """alignment_correct = sign(delta_physician) * delta_llm_correct."""
        phys = _make_physician_df(delta=0.2)   # physician shifts up
        scores = _make_scores_df(cf=0.4, cc=0.7, sc=0.5)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        # sign(0.2) = 1; delta_llm_correct = 0.3 → alignment = 0.3
        assert np.allclose(cells["alignment_correct"].unique(), [0.3], atol=1e-9)

    def test_alignment_difference(self):
        """alignment_difference = alignment_correct - alignment_shuffled."""
        phys = _make_physician_df(delta=0.2)
        scores = _make_scores_df(cf=0.4, cc=0.7, sc=0.5)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        # alignment_correct=0.3, alignment_shuffled=0.1 → diff=0.2
        assert np.allclose(cells["alignment_difference"].unique(), [0.2], atol=1e-9)

    def test_indicator_columns(self):
        """Boolean indicator columns match expected values."""
        phys = _make_physician_df(delta=0.2)
        scores = _make_scores_df(cf=0.4, cc=0.7, sc=0.5)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)

        # physician_shift_indicator: delta_physician=0.2 ≠ 0 → True
        assert cells["physician_shift_indicator"].all()

        # llm_correct_changed_indicator: |0.3| > 0.01 → True
        assert cells["llm_correct_changed_indicator"].all()

        # llm_shuffled_changed_indicator: |0.1| > 0.01 → True
        assert cells["llm_shuffled_changed_indicator"].all()

    def test_no_physician_shift(self):
        """When delta_physician=0, physician_shift_indicator is False."""
        phys = _make_physician_df(delta=0.0)
        scores = _make_scores_df(cf=0.4, cc=0.7, sc=0.5)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        assert not cells["physician_shift_indicator"].any()

    def test_multi_model_produces_separate_rows(self):
        """Two models produce 2× the rows."""
        phys = _make_physician_df()
        s1 = _make_scores_df(model="model_a")
        s2 = _make_scores_df(model="model_b")
        scores = pd.concat([s1, s2], ignore_index=True)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        models = cells["model"].unique()
        assert set(models) == {"model_a", "model_b"}
        assert len(cells[cells["model"] == "model_a"]) == len(cells[cells["model"] == "model_b"])

    def test_required_output_columns_present(self):
        """All required output columns are present."""
        phys = _make_physician_df()
        scores = _make_scores_df()
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        required = [
            "model", "patient", "item_id", "item_text", "category",
            "physician_survey_consensus", "physician_interview_consensus",
            "delta_physician", "llm_context_free", "llm_correct_context",
            "llm_shuffled_context", "delta_llm_correct", "delta_llm_shuffled",
            "alignment_correct", "alignment_shuffled", "alignment_difference",
            "physician_shift_indicator", "llm_correct_changed_indicator",
            "llm_shuffled_changed_indicator",
        ]
        for col in required:
            assert col in cells.columns, f"Missing column: {col}"

    def test_one_row_per_model_patient_item_category(self):
        """Output has exactly n_patients × n_items × n_categories rows per model."""
        phys = _make_physician_df()  # 2 patients × 2 items × 2 cats = 8 rows
        scores = _make_scores_df()
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        assert len(cells) == 2 * 2 * len(CATS)  # one model

    def test_epsilon_threshold_for_indicators(self):
        """Small delta below epsilon → llm_correct_changed_indicator is False."""
        phys = _make_physician_df()
        # cf=0.5, cc=0.505 → delta=0.005, which is < epsilon=0.01
        scores = _make_scores_df(cf=0.5, cc=0.505, sc=0.5)
        cells = build_paired_cell_deltas(phys, scores, CATS, epsilon=0.01)
        assert not cells["llm_correct_changed_indicator"].any()
