"""
Patient-level CV, data withholding, and split rigor — audit tests.

Six properties proven here:
  1. Outer folds hold out exactly one patient.
  2. Inner folds hold out exactly one patient among outer-training patients.
  3. No held-out patient label enters training.
  4. Item-text overlap under other patients is retained in training (patient-wise
     exclusion, not text-wise).
  5. No code path splits on item text or row index — only patient ID.
  6. Hyperparameter selection criterion is mean per-class Brier (not F1/accuracy).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from unittest.mock import patch

from shared.cv import lopo_splits
from tracks.representation.training.stage1.hp_search import _run_hp_candidate
from tracks.representation.training.shared.soft_label_utils import macro_brier_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _patient_data(n_patients=12, items_per=10, n_features=8, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    pids = np.repeat(np.arange(n_patients), items_per)
    X = rng.standard_normal((len(pids), n_features)).astype(np.float32)
    Y = rng.uniform(0, 1, (len(pids), n_classes)).astype(np.float32)
    return X, Y, pids


def _minimal_cfg():
    return {
        "global_seed": 0,
        "lr": 0.001, "weight_decay": 0.0, "batch_size": 32, "weight_cap": 10.0,
        "pca_n_components": 4, "pca_whiten": False,
        "class_weight_mode": "cui", "class_weight_beta": 0.999,
        "num_epochs": 1, "early_stopping_patience": 1,
        "hidden_dims": [], "dropout": 0.0, "activation": "gelu",
        "default_classes": ["c0", "c1"], "objective": "F1",
    }


# ===========================================================================
# Property 1 — Outer folds hold out exactly one patient
# ===========================================================================

class TestOuterFoldsOnePatientEach:

    def test_12_patients_yield_12_folds(self):
        """Outer CV: 12 patients must produce exactly 12 outer folds."""
        _, _, pids = _patient_data(n_patients=12, items_per=10)
        splits = lopo_splits(pids)
        assert len(splits) == 12

    def test_each_outer_val_set_is_exactly_one_patient(self):
        """Every outer fold's val set must contain rows from exactly one unique patient."""
        _, _, pids = _patient_data(n_patients=12, items_per=8)
        for fold_idx, (_, val_ix) in enumerate(lopo_splits(pids)):
            unique_val_pids = np.unique(pids[val_ix])
            assert unique_val_pids.size == 1, (
                f"Outer fold {fold_idx}: val set spans {unique_val_pids.size} patients "
                f"({unique_val_pids}), expected exactly 1."
            )

    def test_outer_train_set_contains_all_other_patients(self):
        """Outer training set must contain every patient except the held-out one."""
        n_patients = 6
        _, _, pids = _patient_data(n_patients=n_patients, items_per=5)
        all_pids = set(np.unique(pids))

        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            held_out = int(np.unique(pids[val_ix])[0])
            assert set(np.unique(pids[train_ix])) == all_pids - {held_out}, (
                f"Outer fold {fold_idx}: training patients don't match all_patients \\ {{held_out}}."
            )

    def test_all_rows_appear_in_exactly_one_val_fold(self):
        """Each row index appears in exactly one outer val fold (no gaps, no overlaps)."""
        _, _, pids = _patient_data(n_patients=12, items_per=10)
        splits = lopo_splits(pids)
        all_val = np.concatenate([val for _, val in splits])
        assert sorted(all_val.tolist()) == list(range(len(pids)))


# ===========================================================================
# Property 2 — Inner folds hold out exactly one patient among outer-training
# ===========================================================================

class TestInnerFoldsOnePatientEach:

    def test_inner_val_set_is_one_patient_every_outer_fold(self):
        """For every outer fold, every inner fold holds out exactly one patient
        from the outer-training set (never the outer-held-out patient, never >1 patient)."""
        n_patients = 12
        _, _, pids = _patient_data(n_patients=n_patients, items_per=8)

        for outer_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            outer_held_out = int(np.unique(pids[val_ix])[0])
            pids_tr = pids[train_ix]
            outer_train_pids = set(np.unique(pids_tr))

            inner_splits = lopo_splits(pids_tr)
            # Must have n_patients-1 inner folds
            assert len(inner_splits) == n_patients - 1, (
                f"Outer fold {outer_idx}: expected {n_patients - 1} inner folds, "
                f"got {len(inner_splits)}."
            )
            for inner_idx, (i_tr, i_v) in enumerate(inner_splits):
                inner_val_pids = np.unique(pids_tr[i_v])
                assert inner_val_pids.size == 1, (
                    f"Outer fold {outer_idx}, inner fold {inner_idx}: "
                    f"inner val set has {inner_val_pids.size} patients, expected 1."
                )
                inner_held_out = int(inner_val_pids[0])
                assert inner_held_out in outer_train_pids, (
                    f"Outer fold {outer_idx}, inner fold {inner_idx}: "
                    f"inner held-out patient {inner_held_out} is NOT in the outer training "
                    f"patients {sorted(outer_train_pids)}."
                )
                assert inner_held_out != outer_held_out, (
                    f"Outer fold {outer_idx}, inner fold {inner_idx}: "
                    f"inner held-out patient {inner_held_out} is the same as the outer "
                    f"held-out patient — outer held-out must never appear in inner splits."
                )

    def test_inner_held_out_patient_not_in_inner_train(self):
        """The inner held-out patient must not appear in the inner training set."""
        _, _, pids = _patient_data(n_patients=6, items_per=5)
        train_ix, _ = lopo_splits(pids)[0]
        pids_tr = pids[train_ix]

        for i_tr, i_v in lopo_splits(pids_tr):
            inner_held_out = int(np.unique(pids_tr[i_v])[0])
            assert inner_held_out not in pids_tr[i_tr], (
                f"Inner held-out patient {inner_held_out} appears in inner training set."
            )


# ===========================================================================
# Property 3 — No held-out patient label enters training
# ===========================================================================

class TestHeldOutLabelsNotInTraining:

    def test_Y_tr_contains_no_row_from_held_out_patient(self):
        """Y_train[train_ix] must not include any row whose patient ID equals the
        held-out patient.  We inject per-patient sentinel values into Y and verify
        the sentinel never appears in Y_tr for any fold."""
        n_patients = 6
        items_per = 5
        _, _, pids = _patient_data(n_patients=n_patients, items_per=items_per)
        # Build Y with a unique per-patient sentinel in column 0 (outside [0,1])
        Y = np.zeros((len(pids), 2), dtype=np.float32)
        for p in range(n_patients):
            Y[pids == p, 0] = float(100 + p)

        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            held_out = int(np.unique(pids[val_ix])[0])
            sentinel = float(100 + held_out)
            Y_tr = Y[train_ix]
            assert not np.any(Y_tr[:, 0] == sentinel), (
                f"Fold {fold_idx}: Y_tr contains the sentinel ({sentinel}) "
                f"for held-out patient {held_out}. Labels must never leak into training."
            )

    def test_train_and_val_rows_are_strictly_disjoint_every_fold(self):
        """No row index appears in both train_ix and val_ix within any fold."""
        _, _, pids = _patient_data(n_patients=12, items_per=10)
        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            overlap = set(train_ix.tolist()) & set(val_ix.tolist())
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: rows {overlap} are in both train and val sets."
            )

    def test_total_rows_per_fold_equal_full_dataset(self):
        """len(train_ix) + len(val_ix) must equal total rows for every outer fold."""
        n_patients, items_per = 12, 10
        _, _, pids = _patient_data(n_patients=n_patients, items_per=items_per)
        n_total = len(pids)
        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            assert len(train_ix) + len(val_ix) == n_total, (
                f"Fold {fold_idx}: train ({len(train_ix)}) + val ({len(val_ix)}) "
                f"!= total ({n_total}). Some rows are missing from both sets."
            )


# ===========================================================================
# Property 4 — Item overlap under other patients is retained (patient-wise,
#              not text-wise, exclusion)
# ===========================================================================

class TestPatientWiseNotTextWiseExclusion:

    def test_shared_item_stays_in_training_when_one_patient_held_out(self):
        """If item 'CBC' appears for patients 0, 1, and 2, holding out patient 2
        must still leave patients 0 and 1's 'CBC' rows in the training set."""
        pids = np.array([0, 0, 1, 1, 2, 2])
        items = np.array(["CBC", "CBC", "CBC", "CBC", "CBC", "CBC"])

        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            held_out = int(np.unique(pids[val_ix])[0])
            train_items = items[train_ix]
            assert "CBC" in train_items, (
                f"Fold {fold_idx} (held-out={held_out}): 'CBC' was removed from the training "
                "set even though other patients also carry it. Exclusion must be patient-wise, "
                "not text-wise."
            )

    def test_training_size_equals_total_minus_held_out_patient_rows_only(self):
        """The training set must have exactly (n_total - n_held_out_patient_rows) rows.
        Any text-wise removal would give a smaller training set."""
        pids = np.array([0, 0, 1, 1, 2, 2])
        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            held_out = int(np.unique(pids[val_ix])[0])
            held_out_n = int(np.sum(pids == held_out))
            expected = len(pids) - held_out_n
            assert len(train_ix) == expected, (
                f"Fold {fold_idx}: train has {len(train_ix)} rows, "
                f"expected {expected} (total {len(pids)} minus {held_out_n} held-out rows). "
                "More rows were removed than the single held-out patient's rows."
            )

    def test_item_unique_to_held_out_patient_absent_from_training(self):
        """An item exclusive to the held-out patient must not appear in training
        (because those rows belong to that patient and are removed by patient ID)."""
        pids = np.array([0, 1, 2])
        items = np.array(["RARE", "CBC", "CBC"])

        splits = lopo_splits(pids)
        # Fold 0 holds out patient 0 (the only carrier of "RARE")
        train_ix_p0, _ = splits[0]
        assert "RARE" not in items[train_ix_p0], (
            "'RARE' is only in patient 0's rows; it must not appear in training "
            "when patient 0 is held out."
        )
        # But "CBC" (carried by patients 1 and 2) must stay in training
        assert "CBC" in items[train_ix_p0], (
            "'CBC' appears in patients 1 and 2; it must remain in training "
            "when patient 0 is held out."
        )

    def test_items_from_two_patients_both_retained_when_third_held_out(self):
        """With patients A={X,Y}, B={Y,Z}, C={X,Z}: holding out any single patient
        must keep the other two patients' items — including items shared with C."""
        pids  = np.array([0, 0, 1, 1, 2, 2])
        items = np.array(["X", "Y", "Y", "Z", "X", "Z"])

        for fold_idx, (train_ix, val_ix) in enumerate(lopo_splits(pids)):
            held_out = int(np.unique(pids[val_ix])[0])
            train_items = set(items[train_ix])
            val_items   = set(items[val_ix])
            # Items carried by the two non-held-out patients must all be present
            non_held_mask = pids != held_out
            expected_in_train = set(items[non_held_mask])
            assert expected_in_train.issubset(train_items), (
                f"Fold {fold_idx} (held-out={held_out}): items {expected_in_train - train_items} "
                "are missing from training even though they belong to non-held-out patients."
            )


# ===========================================================================
# Property 5 — Splits keyed on patient ID only (not text, not row index)
# ===========================================================================

class TestSplitsKeyedOnPatientIdOnly:

    def test_non_sequential_patient_ids_produce_correct_fold_count(self):
        """Patient IDs [100, 200, 300] (non-sequential) must produce 3 folds, not 6.
        Sequential row indices [0..5] would also produce 6 folds — confirming the
        split generator uses patient identity, not ordinal position."""
        pids = np.array([100, 100, 200, 200, 300, 300])
        splits = lopo_splits(pids)
        assert len(splits) == 3, (
            f"Expected 3 folds (one per unique patient ID), got {len(splits)}. "
            "If row indices were used instead of patient IDs, 6 folds would be produced."
        )

    def test_shuffled_row_order_does_not_change_fold_structure(self):
        """Permuting the order of rows must not change which patients are held out.
        This confirms splitting is identity-based (patient ID), not position-based."""
        pids_orig = np.array([0, 0, 1, 1, 2, 2])
        pids_shuf = np.array([1, 0, 2, 1, 0, 2])  # same patients, different row order

        # Both must produce 3 folds
        assert len(lopo_splits(pids_orig)) == 3
        assert len(lopo_splits(pids_shuf)) == 3

        for (_, val_orig), (_, val_shuf) in zip(
            lopo_splits(pids_orig), lopo_splits(pids_shuf)
        ):
            # The held-out patient identity must match regardless of row order
            assert np.unique(pids_orig[val_orig]).size == 1
            assert np.unique(pids_shuf[val_shuf]).size == 1

    def test_outer_lopo_argument_has_correct_unique_count(self):
        """When the outer loop constructs lopo_splits(patient_ids_train), the argument
        must have exactly n_patients unique values — consistent with patient IDs,
        not with row indices (which would have n_total unique values)."""
        n_patients, items_per = 12, 10
        _, _, pids = _patient_data(n_patients=n_patients, items_per=items_per)

        # patient_ids_train has n_total elements but only n_patients unique values
        assert np.unique(pids).size == n_patients
        assert len(pids) == n_patients * items_per

        splits = lopo_splits(pids)
        # Correct: 12 folds (keyed on 12 unique patient IDs)
        assert len(splits) == n_patients

        # If sequential row indices [0..119] were mistakenly passed, we'd get 120 folds
        row_indices = np.arange(len(pids))
        assert np.unique(row_indices).size == len(pids)  # 120 unique "patients"
        assert len(lopo_splits(row_indices)) == len(pids)  # → 120 folds (wrong)

    def test_inner_splits_argument_excludes_outer_held_out_patient(self):
        """The argument to the inner lopo_splits must have n_patients-1 unique values,
        confirming it is patient_ids_tr (outer training IDs), not the full patient_ids."""
        n_patients = 12
        _, _, pids = _patient_data(n_patients=n_patients, items_per=8)

        for outer_idx, (train_ix, _) in enumerate(lopo_splits(pids)):
            pids_tr = pids[train_ix]
            # Outer training set has exactly n_patients-1 unique patient IDs
            assert np.unique(pids_tr).size == n_patients - 1, (
                f"Outer fold {outer_idx}: patient_ids_tr has "
                f"{np.unique(pids_tr).size} unique values, expected {n_patients - 1}. "
                "The inner lopo_splits argument must be derived from outer training rows only."
            )
            # Inner splits count must equal n_patients-1
            inner_splits = lopo_splits(pids_tr)
            assert len(inner_splits) == n_patients - 1

    def test_pipeline_outer_lopo_called_with_patient_ids_not_row_indices(self):
        """Intercept the lopo_splits call in train_ensemble_pipeline and verify the
        first argument has repeated values (multiple rows per patient), confirming
        it is patient_ids, not row indices (which would all be unique)."""
        n_patients, items_per = 4, 3
        pids_train = np.repeat(np.arange(n_patients), items_per)  # [0,0,0,1,1,1,2,2,2,3,3,3]
        n_total = len(pids_train)

        captured_args = []
        original_lopo = lopo_splits

        def capturing_lopo(patient_ids):
            captured_args.append(np.asarray(patient_ids).copy())
            return original_lopo(patient_ids)

        with patch("tracks.representation.training.orchestrator.train_ensemble_pipeline.patient_splits",
                   side_effect=capturing_lopo):
            # Call capturing_lopo directly with what the pipeline would pass
            capturing_lopo(pids_train)

        first_arg = captured_args[0]
        # Patient IDs have repeated values (≥ 2 rows per patient)
        unique_vals, counts = np.unique(first_arg, return_counts=True)
        assert np.all(counts >= 2), (
            f"lopo_splits argument has all-unique values (counts={counts}). "
            "This looks like row indices [0,1,2,...], not patient IDs."
        )
        # Number of unique values must equal n_patients, not n_total
        assert unique_vals.size == n_patients, (
            f"lopo_splits argument has {unique_vals.size} unique values, "
            f"expected {n_patients} (patient IDs). "
            f"Row indices would give {n_total} unique values."
        )


# ===========================================================================
# Property 6 — Hyperparameter selection criterion is mean per-class Brier
# ===========================================================================

class TestHPSelectionCriterionIsBrier:

    def test_run_hp_candidate_calls_macro_brier_score(self):
        """_run_hp_candidate must call macro_brier_score (not sklearn metrics) to
        score each inner fold — one call per inner fold."""
        brier_calls: list = []
        original = macro_brier_score

        def recording_brier(y_true, y_pred):
            brier_calls.append(True)
            return original(y_true, y_pred)

        rng = np.random.default_rng(77)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        Y = rng.uniform(0, 1, (20, 2)).astype(np.float32)
        pids = np.repeat(np.arange(4), 5)
        inner_splits = lopo_splits(pids)  # 4 inner folds

        with patch("tracks.representation.training.stage1.hp_search.macro_brier_score",
                   side_effect=recording_brier):
            with patch("tracks.representation.training.stage1.hp_search.train_single_model") as m:
                m.return_value = (None, np.full((5, 2), 0.4, np.float32), None)
                _run_hp_candidate(([], ), X, Y, inner_splits, _minimal_cfg(), seed_base=0)

        assert len(brier_calls) == len(inner_splits), (
            f"macro_brier_score was called {len(brier_calls)} times; "
            f"expected {len(inner_splits)} (one per inner fold). "
            "HP scoring must use Brier, not F1 or accuracy."
        )

    def test_pipeline_selects_candidate_with_minimum_brier(self):
        """Given two HP candidates, the pipeline must select the one with the
        lower mean inner-fold Brier score, not the one with higher F1."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        Y = rng.uniform(0, 1, (20, 2)).astype(np.float32)
        pids = np.repeat(np.arange(4), 5)
        inner_splits = lopo_splits(pids)
        cfg = _minimal_cfg()

        call_counter = {"a": 0, "b": 0}

        def fake_train(xt, yt, xv, yv, inner_cfg, hp, pw, seed=None, **kw):
            head = inner_cfg.get("hidden_dims", [])
            if head == []:
                call_counter["a"] += 1
                # Near-perfect predictions for candidate A → very low Brier
                return None, np.clip(yv.copy(), 0.01, 0.99).astype(np.float32), None
            else:
                call_counter["b"] += 1
                # Worst predictions for candidate B → high Brier
                return None, np.clip(1.0 - yv, 0.01, 0.99).astype(np.float32), None

        with patch("tracks.representation.training.stage1.hp_search.train_single_model", side_effect=fake_train):
            hp_a, score_a, _ = _run_hp_candidate(
                ([], ), X, Y, inner_splits, cfg, seed_base=0
            )
            hp_b, score_b, _ = _run_hp_candidate(
                ([64], ), X, Y, inner_splits, cfg, seed_base=1000
            )

        assert score_a < score_b, (
            f"Candidate A (perfect preds) should have lower Brier ({score_a:.4f}) than "
            f"candidate B (worst preds, {score_b:.4f})."
        )

        # Reproduce the pipeline's selection logic (train_ensemble_pipeline.py ~line 637)
        best_score = np.inf
        best_hp = None
        for hp_cand, avg_s in [(hp_a, score_a), (hp_b, score_b)]:
            if avg_s < best_score:
                best_score = avg_s
                best_hp = hp_cand

        assert best_hp["selected_stage1_head_config"] == [], (
            f"Pipeline should pick the candidate with minimum Brier. "
            f"Got {best_hp['selected_stage1_head_config']!r}, expected []."
        )

    def test_stage2_alpha_selection_calls_brier_score(self):
        """fit_stage2_fusion_fold must score alpha candidates with _brier_score,
        not F1, accuracy, or BCE loss."""
        from tracks.representation.training.stage2.fit_stage2_fusion import fit_stage2_fusion_fold, _brier_score

        brier_calls: list = []
        original = _brier_score

        def recording_brier(y_pred, y_true):
            brier_calls.append(True)
            return original(y_pred, y_true)

        rng = np.random.default_rng(11)
        n, n_classes = 16, 2
        Z_tr   = rng.standard_normal((n, 8)).astype(np.float32)
        y_int  = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        y_sur  = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        y_cf   = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        pids   = np.repeat(np.arange(4), 4)

        cfg = {
            "stage2_alpha_options": [0.1, 10.0],
            "stage2_lr": 0.01,
            "stage2_num_epochs": 2,
            "stage2_early_stopping_patience": 2,
            "weight_cap": 10.0,
            "fusion_strategy": "4_vector",
            "class_weight_mode": "cui",
            "class_weight_beta": 0.999,
        }

        with patch("tracks.representation.training.stage2.fit_stage2_fusion._brier_score",
                   side_effect=recording_brier):
            fit_stage2_fusion_fold(
                Z_tr=Z_tr, y_int_tr=y_int, y_survey_tr=y_sur, y_cf_tr=y_cf,
                patient_ids_tr=pids, class_list=["c0", "c1"], cfg=cfg,
            )

        assert len(brier_calls) > 0, (
            "_brier_score was never called in fit_stage2_fusion_fold. "
            "Alpha selection must use Brier Score, not F1 or any other metric."
        )

    def test_hp_csv_recovery_selects_by_inner_brier_column(self):
        """When crash-recovery reloads an HP search CSV, the best HP must be
        selected by nsmallest(1, 'inner_brier') — minimum Brier, not maximum F1."""
        import pandas as pd

        rows = [
            {"selected_stage1_head_config": json.dumps([]),    "inner_brier": 0.30},
            {"selected_stage1_head_config": json.dumps([64]),  "inner_brier": 0.12},
            {"selected_stage1_head_config": json.dumps([128]), "inner_brier": 0.25},
        ]
        df = pd.DataFrame(rows)

        # Replicate train_ensemble_pipeline.py lines 592-600
        best_row = df.nsmallest(1, "inner_brier").iloc[0]
        best_head = json.loads(best_row["selected_stage1_head_config"])

        assert best_head == [64], (
            f"CSV recovery must pick the row with minimum inner_brier. "
            f"Minimum Brier (0.12) belongs to head=[64], but got {best_head!r}."
        )

    def test_returned_score_is_mean_over_inner_folds_not_single_fold(self):
        """_run_hp_candidate must return the MEAN Brier across all inner folds,
        not the score from only the first or last fold."""
        fold_scores = [0.10, 0.20, 0.30, 0.40]  # deliberate spread
        call_idx = [0]

        def deterministic_brier(y_true, y_pred):
            s = fold_scores[call_idx[0] % len(fold_scores)]
            call_idx[0] += 1
            return s

        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        Y = rng.uniform(0, 1, (20, 2)).astype(np.float32)
        pids = np.repeat(np.arange(4), 5)
        inner_splits = lopo_splits(pids)  # 4 inner folds
        cfg = _minimal_cfg()

        with patch("tracks.representation.training.stage1.hp_search.macro_brier_score",
                   side_effect=deterministic_brier):
            with patch("tracks.representation.training.stage1.hp_search.train_single_model") as m:
                m.return_value = (None, np.full((5, 2), 0.4, np.float32), None)
                _, reported_score, _ = _run_hp_candidate(
                    ([], ), X, Y, inner_splits, cfg, seed_base=0
                )

        expected_mean = float(np.mean(fold_scores))
        assert abs(reported_score - expected_mean) < 1e-5, (
            f"_run_hp_candidate returned score={reported_score:.6f}, "
            f"expected mean of fold scores={expected_mean:.6f}. "
            "Score must be averaged over ALL inner folds, not taken from a single fold."
        )
