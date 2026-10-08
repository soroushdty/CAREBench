"""Tests for LOPO-CV split generator and macro Brier Score."""

import numpy as np
import pandas as pd
import pytest

from tracks.representation.training.shared.lopo_cv import lopo_splits
from tracks.representation.training.shared.soft_label_utils import macro_brier_score
from shared.utils.text_utils import normalize_for_matching


# ============================================================================
# Tests for lopo_splits
# ============================================================================

class TestLopoSplits:
    """Tests for the lopo_splits function."""

    def test_returns_one_fold_per_patient(self):
        """Number of folds equals number of unique patients."""
        patient_ids = np.array([1, 1, 2, 2, 3, 3])
        splits = lopo_splits(patient_ids)
        assert len(splits) == 3

    def test_val_indices_are_held_out_patient_rows(self):
        """Val indices must contain exactly the rows of the held-out patient."""
        patient_ids = np.array([1, 1, 2, 2, 3, 3])
        splits = lopo_splits(patient_ids)
        unique_patients = np.unique(patient_ids)

        for i, (train_idx, val_idx) in enumerate(splits):
            expected_patient = unique_patients[i]
            assert np.all(patient_ids[val_idx] == expected_patient)

    def test_train_indices_contain_no_held_out_patient(self):
        """Train indices must NOT contain any row from the held-out patient."""
        patient_ids = np.array([0, 0, 1, 1, 2, 2, 2])
        splits = lopo_splits(patient_ids)
        unique_patients = np.unique(patient_ids)

        for i, (train_idx, val_idx) in enumerate(splits):
            held_out = unique_patients[i]
            assert not np.any(patient_ids[train_idx] == held_out)

    def test_all_indices_covered_exactly_once(self):
        """Every row index appears in exactly one val fold."""
        patient_ids = np.array([10, 10, 20, 20, 30])
        splits = lopo_splits(patient_ids)

        all_val = np.concatenate([val for _, val in splits])
        assert sorted(all_val.tolist()) == list(range(len(patient_ids)))

    def test_train_and_val_are_disjoint_per_fold(self):
        """Train and val index sets must be disjoint within each fold."""
        patient_ids = np.array([1, 2, 3, 1, 2, 3])
        splits = lopo_splits(patient_ids)

        for train_idx, val_idx in splits:
            assert len(set(train_idx).intersection(set(val_idx))) == 0

    def test_raises_with_single_patient(self):
        """Fewer than 2 patients should raise ValueError."""
        patient_ids = np.array([1, 1, 1])
        with pytest.raises(ValueError, match="at least 2 unique patients"):
            lopo_splits(patient_ids)

    def test_raises_with_2d_input(self):
        """2-D patient_ids should raise ValueError."""
        patient_ids = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1-D"):
            lopo_splits(patient_ids)

    def test_accepts_non_contiguous_patient_ids(self):
        """Patient IDs do not need to be contiguous integers."""
        patient_ids = np.array([5, 5, 100, 100, 200])
        splits = lopo_splits(patient_ids)
        assert len(splits) == 3

    def test_12_patients_yields_12_folds(self):
        """Regression: 12 patients should give 12 LOPO folds."""
        n_patients = 12
        items_per_patient = 10
        patient_ids = np.repeat(np.arange(n_patients), items_per_patient)
        splits = lopo_splits(patient_ids)
        assert len(splits) == n_patients


# ============================================================================
# Tests for macro_brier_score
# ============================================================================

class TestMacroBrierScore:
    """Tests for the macro_brier_score function."""

    def test_perfect_predictions_give_zero(self):
        """Perfect probability predictions should yield Brier Score = 0."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert macro_brier_score(y_true, probs) == pytest.approx(0.0)

    def test_worst_predictions_give_one(self):
        """Completely wrong probability predictions should yield Brier Score = 1."""
        y_true = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        probs = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        assert macro_brier_score(y_true, probs) == pytest.approx(1.0)

    def test_uniform_probs_give_expected_score(self):
        """0.5 predictions for all-positive labels should yield 0.25."""
        y_true = np.array([[1.0], [1.0]], dtype=np.float32)
        probs = np.array([[0.5], [0.5]], dtype=np.float32)
        assert macro_brier_score(y_true, probs) == pytest.approx(0.25)

    def test_returns_float(self):
        y_true = np.array([[1.0, 0.0]], dtype=np.float32)
        probs = np.array([[0.8, 0.2]], dtype=np.float32)
        result = macro_brier_score(y_true, probs)
        assert isinstance(result, float)

    def test_macro_averages_over_classes(self):
        """Brier Score is the mean of per-class scores."""
        y_true = np.array([[1.0, 0.0]], dtype=np.float32)
        probs = np.array([[0.0, 0.0]], dtype=np.float32)
        # class_0: (1 - 0)^2 = 1.0; class_1: (0 - 0)^2 = 0.0 → mean = 0.5
        assert macro_brier_score(y_true, probs) == pytest.approx(0.5)


# ============================================================================
# Tests for pos-weight formula
# ============================================================================

class TestPosWeightFormula:
    """Tests for the N / (2 * pos) positive-class weight formula."""

    def test_pos_weight_formula(self):
        """Verify N/(2*pos) formula matches expected values."""
        N = 100
        pos = np.array([10.0, 50.0, 1.0])
        expected = np.array([5.0, 1.0, 50.0])
        result = N / (2.0 * np.maximum(pos, 1.0))
        np.testing.assert_allclose(result, expected)

    def test_pos_weight_clipped_to_cap(self):
        """Values above weight_cap should be clipped."""
        N = 100
        pos = np.array([1.0])  # N / (2 * 1) = 50
        cap = 20.0
        result = np.clip(N / (2.0 * np.maximum(pos, 1.0)), 1.0, cap)
        np.testing.assert_allclose(result, np.array([20.0]))

    def test_pos_weight_minimum_is_one(self):
        """Clipped values should always be at least 1.0."""
        N = 2
        pos = np.array([2.0])  # N / (2 * 2) = 0.5, clipped to 1.0
        result = np.clip(N / (2.0 * np.maximum(pos, 1.0)), 1.0, 100.0)
        np.testing.assert_allclose(result, np.array([1.0]))


# ============================================================================
# Tests for fold-local stratum assignment
# ============================================================================

def _compute_fold_strata(item_strings_train, patient_ids_train):
    """Replicate the fold-local stratum logic from train_ensemble_pipeline."""
    splits = lopo_splits(patient_ids_train)
    oof_strata = np.full(len(patient_ids_train), None, dtype=object)
    for train_ix, val_ix in splits:
        train_norms = set(
            pd.Series(item_strings_train[train_ix]).map(normalize_for_matching)
        )
        fold_stratum = np.where(
            pd.Series(item_strings_train[val_ix])
              .map(normalize_for_matching)
              .isin(train_norms),
            "repeated",
            "unique",
        )
        oof_strata[val_ix] = fold_stratum
    return oof_strata


class TestFoldLocalStratum:
    """Verify that per-fold stratum correctly reflects each fold's training vocabulary."""

    def test_shared_items_are_repeated(self):
        """Items present in every patient's training data should be 'repeated'."""
        # 3 patients; all have the item "CBC"
        patient_ids = np.array([0, 0, 1, 1, 2, 2])
        items = np.array(["CBC", "CBC", "CBC", "CBC", "CBC", "CBC"])
        strata = _compute_fold_strata(items, patient_ids)
        # When patient 0 is held out, "CBC" is still in {1,2} training data → repeated
        assert all(s == "repeated" for s in strata)

    def test_unique_item_for_held_out_patient_is_unique(self):
        """An item that only appears for the held-out patient should be 'unique'."""
        # Patient 0 has "RARE"; patients 1 and 2 only have "CBC"
        patient_ids = np.array([0, 0, 1, 1, 2, 2])
        items = np.array(["RARE", "RARE", "CBC", "CBC", "CBC", "CBC"])
        strata = _compute_fold_strata(items, patient_ids)
        # When patient 0 is held out: "RARE" not in {1,2} vocab → unique
        assert strata[0] == "unique"
        assert strata[1] == "unique"
        # When patients 1 or 2 are held out: "CBC" is still present → repeated
        # (patients 1's rows are indices 2,3; patients 2's rows are 4,5)
        assert strata[2] == "repeated"
        assert strata[3] == "repeated"
        assert strata[4] == "repeated"
        assert strata[5] == "repeated"

    def test_globally_repeated_but_fold_unique(self):
        """An item shared only between two patients is 'unique' when the other is held out."""
        # Patient 0 and 1 both have "SHARED"; patient 2 has only "OTHER".
        # Globally "SHARED" appears in training data → global stratum = repeated.
        # But when patient 1 is held out, "SHARED" still comes from patient 0 → repeated.
        # When patient 0 is held out, "SHARED" still comes from patient 1 → repeated.
        # Patient 2's item "OTHER" never appears in patient 0 or 1 → unique every fold.
        patient_ids = np.array([0, 1, 2])
        items = np.array(["SHARED", "SHARED", "OTHER"])
        strata = _compute_fold_strata(items, patient_ids)
        # patient 0 held out: training items = {SHARED (from p1)} → SHARED repeated
        assert strata[0] == "repeated"
        # patient 1 held out: training items = {SHARED (from p0)} → SHARED repeated
        assert strata[1] == "repeated"
        # patient 2 held out: training items = {SHARED} → OTHER unique
        assert strata[2] == "unique"

    def test_all_rows_receive_stratum(self):
        """Every row should be assigned a stratum (no None left)."""
        patient_ids = np.array([1, 1, 2, 2, 3])
        items = np.array(["A", "B", "A", "C", "D"])
        strata = _compute_fold_strata(items, patient_ids)
        assert all(s is not None for s in strata)
        assert all(s in ("repeated", "unique") for s in strata)
