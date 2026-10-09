"""Tests for shared.cv: patient-grouped cross-validation splits."""

from __future__ import annotations

import numpy as np
import pytest

from shared.cv import (
    CVScheme,
    held_out_patients,
    lopo_splits,
    patient_splits,
    patient_to_fold,
    resolve_cv,
)


def _pids(n_patients=10, items_per=3):
    # Interleave patients so held-out rows are not contiguous.
    return np.tile(np.arange(100, 100 + n_patients), items_per)


def _kfold(n_splits, seed=0):
    return CVScheme(scheme="group_kfold", n_splits=n_splits, seed=seed)


# ---------------------------------------------------------------------------
# Both schemes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cv", [CVScheme(), _kfold(3), _kfold(10)])
class TestPatientGroupedInvariants:

    def test_val_rows_cover_every_row_once(self, cv):
        pids = _pids()
        all_val = np.concatenate([val for _, val in patient_splits(pids, cv)])
        assert sorted(all_val.tolist()) == list(range(len(pids)))

    def test_no_patient_on_both_sides(self, cv):
        pids = _pids()
        for train_ix, val_ix in patient_splits(pids, cv):
            assert not set(pids[train_ix]) & set(pids[val_ix])
            assert len(train_ix) + len(val_ix) == len(pids)

    def test_val_holds_all_rows_of_its_patients(self, cv):
        pids = _pids()
        for _, val_ix in patient_splits(pids, cv):
            for p in np.unique(pids[val_ix]):
                assert set(np.where(pids == p)[0]) <= set(val_ix)

    def test_indices_are_sorted(self, cv):
        for train_ix, val_ix in patient_splits(_pids(), cv):
            assert np.all(np.diff(train_ix) > 0)
            assert np.all(np.diff(val_ix) > 0)

    def test_patient_to_fold_matches_splits(self, cv):
        pids = _pids()
        splits = patient_splits(pids, cv)
        mapping = patient_to_fold(pids, splits)
        assert set(mapping) == set(np.unique(pids).tolist())
        for patient, fold_idx in mapping.items():
            assert patient in pids[splits[fold_idx][1]]
            assert isinstance(patient, int)


# ---------------------------------------------------------------------------
# LOPO
# ---------------------------------------------------------------------------

class TestLopo:

    def test_default_is_lopo(self):
        pids = _pids()
        for (tr_a, va_a), (tr_b, va_b) in zip(patient_splits(pids), lopo_splits(pids)):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(va_a, va_b)

    def test_one_fold_per_patient_in_sorted_order(self):
        pids = np.array([7, 3, 7, 5, 3])
        held = held_out_patients(pids, lopo_splits(pids))
        assert [h.tolist() for h in held] == [[3], [5], [7]]

    def test_matches_previous_implementation(self):
        """Same indices as the leave-one-patient-out generator this replaced."""
        pids = _pids(n_patients=6, items_per=4)
        for patient, (train_ix, val_ix) in zip(np.unique(pids), lopo_splits(pids)):
            np.testing.assert_array_equal(val_ix, np.where(pids == patient)[0])
            np.testing.assert_array_equal(train_ix, np.where(pids != patient)[0])


# ---------------------------------------------------------------------------
# Grouped k-fold
# ---------------------------------------------------------------------------

class TestGroupKFold:

    def test_fold_count_and_balanced_patient_counts(self):
        held = held_out_patients(_pids(n_patients=10), patient_splits(_pids(10), _kfold(3)))
        assert len(held) == 3
        assert sorted(len(h) for h in held) == [3, 3, 4]

    def test_same_seed_same_splits(self):
        pids = _pids()
        a = held_out_patients(pids, patient_splits(pids, _kfold(3, seed=1)))
        b = held_out_patients(pids, patient_splits(pids, _kfold(3, seed=1)))
        assert [x.tolist() for x in a] == [x.tolist() for x in b]

    def test_different_seed_changes_assignment(self):
        pids = _pids(n_patients=20)
        a = held_out_patients(pids, patient_splits(pids, _kfold(4, seed=1)))
        b = held_out_patients(pids, patient_splits(pids, _kfold(4, seed=2)))
        assert [x.tolist() for x in a] != [x.tolist() for x in b]

    def test_assignment_ignores_row_order(self):
        """Fold membership depends on the patient set and seed, not on row order."""
        pids = _pids()
        shuffled = np.random.default_rng(5).permutation(pids)
        a = held_out_patients(pids, patient_splits(pids, _kfold(3)))
        b = held_out_patients(shuffled, patient_splits(shuffled, _kfold(3)))
        assert [x.tolist() for x in a] == [x.tolist() for x in b]

    def test_n_splits_above_patient_count_gives_one_patient_per_fold(self):
        pids = _pids(n_patients=4)
        held = held_out_patients(pids, patient_splits(pids, _kfold(10)))
        assert len(held) == 4
        assert all(len(h) == 1 for h in held)

    def test_string_patient_ids(self):
        pids = np.array(["a", "b", "c", "d", "a", "b"])
        splits = patient_splits(pids, _kfold(2))
        assert len(splits) == 2
        assert set(patient_to_fold(pids, splits)) == {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_single_patient_raises(self):
        with pytest.raises(ValueError, match="at least 2 unique patients"):
            patient_splits(np.array([1, 1, 1]), _kfold(2))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            patient_splits(np.array([[1, 2], [3, 4]]))

    @pytest.mark.parametrize("kwargs, match", [
        ({"scheme": "kfold"}, "must be one of"),
        ({"scheme": "lopo", "n_splits": 5}, "takes no n_splits or seed"),
        ({"scheme": "lopo", "seed": 1}, "takes no n_splits or seed"),
        ({"scheme": "group_kfold", "seed": 1}, "integer n_splits"),
        ({"scheme": "group_kfold", "n_splits": 1, "seed": 1}, "at least 2"),
        ({"scheme": "group_kfold", "n_splits": 3.0, "seed": 1}, "integer n_splits"),
        ({"scheme": "group_kfold", "n_splits": 3}, "integer seed"),
    ])
    def test_invalid_scheme(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CVScheme(**kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestResolveCv:

    def test_missing_block_is_lopo(self):
        assert resolve_cv({}, "outer") == CVScheme()
        assert resolve_cv({"cv": None}, "inner") == CVScheme()
        assert resolve_cv({"cv": {"outer": None}}, "outer") == CVScheme()

    def test_explicit_lopo_with_nulls(self):
        cfg = {"cv": {"outer": {"scheme": "lopo", "n_splits": None, "seed": None}}}
        assert resolve_cv(cfg, "outer") == CVScheme()

    def test_loops_are_independent(self):
        cfg = {"cv": {"outer": {"scheme": "group_kfold", "n_splits": 5, "seed": 3}}}
        assert resolve_cv(cfg, "outer") == _kfold(5, seed=3)
        assert resolve_cv(cfg, "inner") == CVScheme()

    def test_seed_falls_back_to_global_seed(self):
        cfg = {"global_seed": 42, "cv": {"inner": {"scheme": "group_kfold", "n_splits": 4}}}
        assert resolve_cv(cfg, "inner") == _kfold(4, seed=42)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key"):
            resolve_cv({"cv": {"outer": {"scheme": "lopo", "folds": 3}}}, "outer")

    def test_unknown_loop_raises(self):
        with pytest.raises(ValueError, match="Unknown cv loop"):
            resolve_cv({"cv": {"middle": {}}}, "outer")
        with pytest.raises(ValueError, match="cv loop must be one of"):
            resolve_cv({}, "middle")

    def test_shipped_training_config_is_lopo(self):
        """configs/training_config.yaml keeps the default LOPO scheme for both loops."""
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "configs" / "training_config.yaml")
            .read_text(encoding="utf-8")
        )
        assert resolve_cv(cfg, "outer") == CVScheme()
        assert resolve_cv(cfg, "inner") == CVScheme()
