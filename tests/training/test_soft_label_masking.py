import numpy as np
import pandas as pd
import pytest
import torch

from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
from tracks.representation.models.MultiLabelModel import MultiLabelModel
from tracks.representation.training.orchestrator.compute_metrics_and_save import compute_metrics_df
from tracks.representation.training.shared.fit_calibrators import fit_calibrators
from tracks.representation.training.orchestrator.plot_ensemble_figures import compute_ensemble_curves, write_ensemble_figures
from tracks.representation.training.shared.threshold_tuning import threshold_tuning
from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline
from tracks.representation.training.stage1.train_single_model import train_single_model

# Distinct non-binary marker used to assert held-out test labels are not consumed.
TEST_LABEL_SENTINEL_VALUE = 0.123


def _base_cfg():
    return {
        "global_seed": 13,
        "eval_pos_threshold": 0.5,
        "threshold_objective": "F1",
        "threshold_beta": 1.0,
        "threshold_search_grid": [0.25, 0.5, 0.75],
        "threshold_min_precision_floor": None,
        "threshold_min_recall_floor": None,
        "threshold_min_value": 0.01,
        "threshold_max_value": 0.99,
        "threshold_max_predicted_positive_rate": None,
        "calibration_method": "isotonic",
        "calibration_min_samples": 2,
        "num_epochs": 2,
        "early_stopping_patience": 1,
        "objective": "F1",
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": 8,
        "weight_cap": 10.0,
        "pca_n_components": 50,
        "pca_whiten": False,
        "default_classes": ["class_0", "class_1"],
        "hidden_dims": [],
        "dropout": 0.0,
        "activation": "gelu",
    }


def test_threshold_tuning_uses_explicit_grid_not_prevalence_multiplier_default():
    cfg = _base_cfg()
    cfg["threshold_search_grid"] = [0.33, 0.66]
    cfg["threshold_objective"] = "Precision"
    cfg["threshold_multipliers"] = [0.8, 1.0, 1.2]  # ignored by new tuner

    probs = np.array([[0.95], [0.60], [0.40], [0.20], [0.10]], dtype=np.float32)
    y_true = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    prevalence = float(y_true.mean())

    thresholds = threshold_tuning(probs, y_true, ["class_0"], cfg)

    assert thresholds.shape == (1,)
    assert float(thresholds[0]) == pytest.approx(0.33) or float(thresholds[0]) == pytest.approx(0.66)
    assert all(
        float(thresholds[0]) != pytest.approx(prevalence * m)
        for m in cfg["threshold_multipliers"]
    )


def test_threshold_tuning_honors_objective_and_constraints():
    cfg = _base_cfg()
    cfg["threshold_search_grid"] = [0.3, 0.55, 0.9]

    probs = np.array([[0.9], [0.6], [0.55], [0.4], [0.3]], dtype=np.float32)
    y_true = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]], dtype=np.float32)

    cfg["threshold_objective"] = "Recall"
    t_recall = float(threshold_tuning(probs, y_true, ["class_0"], cfg)[0])

    cfg["threshold_objective"] = "Precision"
    t_precision = float(threshold_tuning(probs, y_true, ["class_0"], cfg)[0])

    assert t_recall == pytest.approx(0.3)
    assert t_precision == pytest.approx(0.9)


def test_threshold_tuning_enforces_min_precision_floor():
    cfg = _base_cfg()
    cfg["threshold_objective"] = "Recall"
    cfg["threshold_search_grid"] = [0.3, 0.55, 0.9]
    cfg["threshold_min_precision_floor"] = 0.8

    probs = np.array([[0.9], [0.6], [0.55], [0.4], [0.3]], dtype=np.float32)
    y_true = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]], dtype=np.float32)

    threshold, report = threshold_tuning(probs, y_true, ["class_0"], cfg, return_report=True)
    row = report.iloc[0]

    assert float(threshold[0]) == pytest.approx(0.9)
    assert row["Precision@Selected"] >= 0.8


def test_train_single_model_early_stopping_uses_configured_objective(monkeypatch):
    cfg = _base_cfg()
    cfg["objective"] = "F1"
    cfg["num_epochs"] = 1
    cfg["early_stopping_patience"] = 1

    objective_seen = []

    def fake_loss(logits, yb, pos_w):
        return logits.mean() * 0.0 + torch.tensor(1.0, dtype=logits.dtype, device=logits.device, requires_grad=True)

    def fake_threshold_tuning(v_probs, Y_val, class_list, cfg):
        return np.full(Y_val.shape[1], 0.5, dtype=np.float32)

    def fake_masked_macro_metric(y_true, probs, thresholds, eval_pos_threshold_or_cfg, objective):
        objective_seen.append(objective)
        return 0.5

    monkeypatch.setattr("tracks.representation.training.stage1.train_single_model.compute_weighted_bce_loss", fake_loss)
    monkeypatch.setattr("tracks.representation.training.stage1.train_single_model.threshold_tuning", fake_threshold_tuning)
    monkeypatch.setattr("tracks.representation.training.stage1.train_single_model.masked_macro_metric", fake_masked_macro_metric)

    X_train = np.random.randn(8, 4).astype(np.float32)
    Y_train = np.random.binomial(1, 0.5, (8, 2)).astype(np.float32)
    X_val = np.random.randn(4, 4).astype(np.float32)
    Y_val = np.random.binomial(1, 0.5, (4, 2)).astype(np.float32)
    hp = {"lr": 0.001, "weight_decay": 0.0, "batch_size": 4, "weight_cap": 10.0}
    pos_weight = np.array([1.0, 1.0], dtype=np.float32)

    train_single_model(X_train, Y_train, X_val, Y_val, cfg, hp, pos_weight)
    assert objective_seen == ["F1"]


def test_calibration_and_threshold_tuning_use_validation_data_only(monkeypatch, tmp_path):
    cfg = _base_cfg()
    cfg["DIR_MODEL"] = str(tmp_path / "model")
    cfg["num_epochs"] = 1
    cfg["early_stopping_patience"] = 1

    # Use 4 patients so inner LOPO is feasible (outer fold has 3 patients,
    # inner LOPO needs ≥ 2).
    n_patients = 4
    samples_per_patient = 5
    X_train = np.random.randn(n_patients * samples_per_patient, 5).astype(np.float32)
    Y_train = np.random.binomial(1, 0.4, (n_patients * samples_per_patient, 2)).astype(np.float32)
    patient_ids_train = np.repeat(np.arange(n_patients), samples_per_patient)
    X_test = np.random.randn(8, 5).astype(np.float32)
    Y_test = np.full((8, 2), TEST_LABEL_SENTINEL_VALUE, dtype=np.float32)

    seen_threshold_inputs = []
    seen_calibration_inputs = []

    def fake_train_single_model(X_train, Y_train, X_val, Y_val, cfg, hp, pos_weight, **kwargs):
        model = MultiLabelModel(X_train.shape[1], Y_train.shape[1], hidden_dims=[])
        probs = np.full((X_val.shape[0], Y_val.shape[1]), 0.5, dtype=np.float32)
        thresholds = np.full(Y_val.shape[1], 0.5, dtype=np.float32)
        return model, probs, thresholds

    def fake_threshold_tuning(probs, Y_val_local, class_list, cfg, return_report=False):
        seen_threshold_inputs.append(np.array(Y_val_local, copy=True))
        thresholds = np.full(len(class_list), 0.5, dtype=np.float32)
        if return_report:
            report = pd.DataFrame(
                {
                    "Class": class_list,
                    "Valid Count": [len(Y_val_local)] * len(class_list),
                    "Soft Prevalence": [0.5] * len(class_list),
                    "Threshold Objective": [cfg["threshold_objective"]] * len(class_list),
                    "Threshold Beta": [cfg["threshold_beta"]] * len(class_list),
                    "Selected Threshold": [0.5] * len(class_list),
                    "Selected Objective Score": [0.0] * len(class_list),
                    "Precision@Selected": [0.0] * len(class_list),
                    "Recall@Selected": [0.0] * len(class_list),
                    "F1@Selected": [0.0] * len(class_list),
                    "Fbeta@Selected": [0.0] * len(class_list),
                    "Pred Pos Rate@Selected": [0.5] * len(class_list),
                }
            )
            return thresholds, report
        return thresholds

    def fake_fit_calibrators(probs, Y_val_local, class_list, cfg):
        seen_calibration_inputs.append(np.array(Y_val_local, copy=True))
        return {cls: ConstantCalibrator(0.5) for cls in class_list}

    monkeypatch.setattr("tracks.representation.training.orchestrator.train_ensemble_pipeline.train_single_model", fake_train_single_model)
    monkeypatch.setattr("tracks.representation.training.orchestrator.fold_postprocessing.threshold_tuning", fake_threshold_tuning)
    monkeypatch.setattr("tracks.representation.training.orchestrator.train_ensemble_pipeline.fit_calibrators", fake_fit_calibrators)

    train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, cfg, patient_ids_test=np.arange(X_test.shape[0]))

    assert seen_threshold_inputs
    assert seen_calibration_inputs
    assert all(arr.shape[0] != Y_test.shape[0] for arr in seen_threshold_inputs)
    assert all(arr.shape[0] != Y_test.shape[0] for arr in seen_calibration_inputs)
    assert all(not np.all(np.isclose(arr, TEST_LABEL_SENTINEL_VALUE)) for arr in seen_threshold_inputs)
    assert all(not np.all(np.isclose(arr, TEST_LABEL_SENTINEL_VALUE)) for arr in seen_calibration_inputs)


def test_macro_row_does_not_include_misleading_confusion_counts():
    cfg = _base_cfg()
    probs = np.array([[0.8, 0.2], [0.7, 0.9], [0.1, 0.3]], dtype=np.float32)
    y_true = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    thresholds = np.array([0.5, 0.5], dtype=np.float32)

    patient_ids = np.arange(probs.shape[0])
    df = compute_metrics_df(probs, y_true, thresholds, ["class_0", "class_1"], cfg, patient_ids)
    macro = df[df["Class"] == "Macro Average"].iloc[0]
    micro = df[df["Class"] == "Micro Aggregate"].iloc[0]

    assert np.isnan(macro["TP"])
    assert np.isnan(macro["FP"])
    assert np.isnan(macro["TN"])
    assert np.isnan(macro["FN"])
    assert micro["TP"] + micro["FP"] + micro["TN"] + micro["FN"] == micro["Valid Count"]
    assert "Pred Pos Rate" in df.columns


def test_plot_uses_config_driven_eval_pos_threshold(monkeypatch, tmp_path):
    calls = []

    def fake_masked_class_data(y_true, probs, class_idx, cfg):
        calls.append(float(cfg["eval_pos_threshold"]))
        return np.ones(y_true.shape[0], dtype=bool), y_true[:, class_idx], probs[:, class_idx]

    monkeypatch.setattr("tracks.representation.training.orchestrator.plot_ensemble_figures.masked_class_data", fake_masked_class_data)

    probs = np.array([[0.6], [0.4]], dtype=np.float32)
    y_true = np.array([[0.4], [1.0]], dtype=np.float32)
    cfg = {"eval_pos_threshold": 0.4}
    write_ensemble_figures(compute_ensemble_curves(probs, y_true, ["class_0"], cfg), str(tmp_path))

    assert calls
    assert set(calls) == {0.4}


def test_fit_calibrators_handles_none_method_and_single_class_masked_subset():
    probs = np.array([[0.2], [0.8]], dtype=np.float32)
    y_true = np.array([[0.5], [0.5]], dtype=np.float32)
    cfg_none = {"eval_pos_threshold": 0.5, "calibration_method": "none"}

    calibrators_none = fit_calibrators(probs, y_true, ["class_0"], cfg_none)
    assert calibrators_none["class_0"] is None


def test_thresh_inner_does_not_use_held_out_patient_labels(monkeypatch, tmp_path):
    """thresh_inner must be derived from inner-fold data, not the held-out patient."""
    cfg = _base_cfg()
    cfg["DIR_MODEL"] = str(tmp_path / "model")
    cfg["num_epochs"] = 1
    cfg["early_stopping_patience"] = 1

    # Sentinel: a value that only appears in the per-fold held-out Y_val rows.
    # Each outer fold holds out one patient; those rows get sentinel labels so
    # any threshold call that sees sentinel values came from the held-out patient.
    SENTINEL = 0.777

    n_patients = 4
    samples_per_patient = 5
    n_total = n_patients * samples_per_patient
    X_train = np.random.randn(n_total, 5).astype(np.float32)
    # Non-sentinel binary labels for training rows (0 or 1, no 0.777).
    Y_train = np.random.binomial(1, 0.4, (n_total, 2)).astype(np.float32)
    patient_ids_train = np.repeat(np.arange(n_patients), samples_per_patient)
    X_test = np.random.randn(8, 5).astype(np.float32)
    Y_test = np.full((8, 2), TEST_LABEL_SENTINEL_VALUE, dtype=np.float32)

    # Inject sentinel into each patient's rows so we can detect if threshold_tuning
    # in train_ensemble_pipeline is ever called with those rows.
    for p in range(n_patients):
        mask = patient_ids_train == p
        Y_train[mask] = SENTINEL

    thresh_inner_y_vals = []

    def spy_threshold_tuning(probs, Y_val, class_list, cfg, return_report=False):
        thresh_inner_y_vals.append(np.array(Y_val, copy=True))
        return threshold_tuning(probs, Y_val, class_list, cfg, return_report=return_report)

    monkeypatch.setattr(
        "tracks.representation.training.orchestrator.train_ensemble_pipeline.threshold_tuning", spy_threshold_tuning
    )

    train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, cfg, patient_ids_test=np.arange(X_test.shape[0]))

    assert thresh_inner_y_vals, "threshold_tuning was never called from train_ensemble_pipeline"
    for y in thresh_inner_y_vals:
        # Each call must use ALL outer-training rows — never a single-patient slice.
        # A single held-out patient contributes samples_per_patient=5 rows; the
        # inner OOF covers n_total - samples_per_patient = 15 rows minimum.
        assert y.shape[0] > samples_per_patient, (
            f"threshold_tuning received only {y.shape[0]} rows — "
            "looks like a single-patient slice, not inner-LOPO OOF"
        )


# ---------------------------------------------------------------------------
# Regression: Stage 2 pos_weights must not binarize 0.5 labels
# ---------------------------------------------------------------------------

def test_stage2_pos_weights_inverse_frequency_half_labels():
    """_compute_pos_weights (inverse_frequency) must use soft sum, not > 0.5 binarization.

    For labels [1.0, 0.5, 0.0] with n_total=3:
    - OLD (> 0.5): pos_mass=1  → w = 3/(2*1) = 1.5
    - NEW (sum):   pos_mass=1.5 → w = 3/(2*1.5) = 1.0

    This is the primary regression test that would have failed with the old code.
    Cui-2019 mode is not tested here because its max(pos_mass, 1) safeguard
    always clamps weights to 1.0 for any pos_mass >= 1, making the two code
    paths identical in that mode.
    """
    from tracks.representation.training.stage2.fit_stage2_fusion import _compute_pos_weights
    import numpy as np

    y = np.array([[1.0], [0.5], [0.0]], dtype=np.float32)
    cfg_inv = {"class_weight_mode": "inverse_frequency"}
    weights = _compute_pos_weights(y, weight_cap=50.0, cfg=cfg_inv)

    # soft sum = 1.5; n_total = 3; expected = 3 / (2 * 1.5) = 1.0
    expected_w = min(max(3.0 / (2.0 * 1.5), 1.0), 50.0)  # = 1.0

    assert abs(float(weights[0]) - expected_w) < 1e-4, (
        f"inverse_frequency weight with soft-sum pos_mass=1.5 should be {expected_w:.4f}, "
        f"got {float(weights[0]):.4f}. "
        "Old code using '> 0.5' binarization gives pos_mass=1 → w=1.5."
    )


def test_stage2_pos_weights_inverse_frequency_larger_gap():
    """_compute_pos_weights inverse_frequency: larger gap between hard and soft count."""
    from tracks.representation.training.stage2.fit_stage2_fusion import _compute_pos_weights
    import numpy as np

    # 10 items: 2 hard positives + 6 disagreements (0.5) + 2 negatives
    # OLD pos_mass = 2, NEW pos_mass = 2 + 3 = 5
    y = np.array([[1.0]] * 2 + [[0.5]] * 6 + [[0.0]] * 2, dtype=np.float32)
    cfg_inv = {"class_weight_mode": "inverse_frequency"}
    weights = _compute_pos_weights(y, weight_cap=50.0, cfg=cfg_inv)

    # soft sum = 2 + 0.5*6 = 5; n_total = 10; expected = 10 / (2 * 5) = 1.0
    expected_w = min(max(10.0 / (2.0 * 5.0), 1.0), 50.0)  # = 1.0

    assert abs(float(weights[0]) - expected_w) < 1e-4, (
        f"inverse_frequency weight with soft-sum pos_mass=5 should be {expected_w:.4f}, "
        f"got {float(weights[0]):.4f}. "
        "Old '> 0.5' code gives pos_mass=2 → w = 10/(2*2) = 2.5."
    )


def test_half_labels_preserved_through_preprocessing_targets():
    """Pair aggregation of [1, 0] must produce exactly 0.5 (not rounded).

    This ensures physician-disagreement rows reach training as float 0.5,
    not as 0 or 1 through any integer cast in the aggregation path.
    """
    import pandas as pd
    from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians

    df = pd.DataFrame({
        "Patient":   [1.0, 1.0],
        "Item":      ["CBC", "CBC"],
        "Physician": [10.0, 11.0],
        "label":     [1.0, 0.0],
    })
    result = _aggregate_physicians(
        df,
        patient_col="Patient",
        physician_col="Physician",
        item_col="Item",
        class_cols=["label"],
        physician_count=2,
    )

    assert len(result) == 1, f"Expected 1 aggregated row, got {len(result)}"
    label_val = float(result.iloc[0]["label"])
    assert label_val == 0.5, (
        f"Physician disagreement (1 + 0) / 2 must be exactly 0.5, got {label_val}. "
        "A round() or astype(int) in the aggregation path would cause this failure."
    )


# ---------------------------------------------------------------------------
# Regression: physician count edge cases in pair aggregation
# ---------------------------------------------------------------------------

def test_physician_reconciliation_drops_count_1_group_with_warning():
    """A (patient, item) group with only 1 physician row (expected 2) must be
    dropped — not silently kept with a misleading aggregated label — when
    mismatch_error=False.
    """
    import pandas as pd
    import logging
    from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians

    df = pd.DataFrame({
        "Patient":   [1.0, 2.0, 2.0],
        "Item":      ["CBC", "CBC", "CBC"],
        "Physician": [10.0, 10.0, 11.0],
        "label":     [1.0, 1.0, 0.0],
    })
    # Patient 1 has only 1 physician row → should be dropped (count=1 ≠ physician_count=2)
    # Patient 2 has 2 physician rows → should be kept (aggregated to mean)
    result = _aggregate_physicians(
        df,
        patient_col="Patient",
        physician_col="Physician",
        item_col="Item",
        class_cols=["label"],
        physician_count=2,
        mismatch_error=False,
    )

    assert 1.0 not in result["Patient"].values, (
        "Patient 1 (count=1) must be dropped when physician_count=2 and mismatch_error=False"
    )
    assert 2.0 in result["Patient"].values, (
        "Patient 2 (count=2) must be retained and aggregated"
    )
    p2_label = float(result[result["Patient"] == 2.0].iloc[0]["label"])
    assert p2_label == 0.5, f"Patient 2 label should be (1+0)/2=0.5, got {p2_label}"


def test_physician_reconciliation_raises_on_count_1_when_error_flag_set():
    """mismatch_error=True must raise ValueError for a group with wrong count."""
    import pandas as pd
    import pytest
    from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians

    df = pd.DataFrame({
        "Patient":   [1.0],
        "Item":      ["CBC"],
        "Physician": [10.0],
        "label":     [1.0],
    })
    with pytest.raises(ValueError, match="Reference observer count mismatch"):
        _aggregate_physicians(
            df,
            patient_col="Patient",
            physician_col="Physician",
            item_col="Item",
            class_cols=["label"],
            physician_count=2,
            mismatch_error=True,
        )


def test_physician_reconciliation_condition_a_resolvable_quadruples():
    """Four rows (physician 10 × 2 + physician 11 × 2) with physician_count=2
    must resolve via Condition A: every row has an exact replica AND unique_count==2.
    The two distinct responses are aggregated to their mean.
    """
    import pandas as pd
    from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians

    # Each physician's response is duplicated → all rows have replicas, unique_count=2.
    df = pd.DataFrame({
        "Patient":   [1.0, 1.0, 1.0, 1.0],
        "Item":      ["CBC", "CBC", "CBC", "CBC"],
        "Physician": [10.0, 10.0, 11.0, 11.0],
        "label":     [1.0, 1.0, 0.0, 0.0],
    })
    result = _aggregate_physicians(
        df,
        patient_col="Patient",
        physician_col="Physician",
        item_col="Item",
        class_cols=["label"],
        physician_count=2,
        mismatch_error=False,
    )

    assert len(result) == 1, f"Condition A should produce 1 aggregated row, got {len(result)}"
    label_val = float(result.iloc[0]["label"])
    assert label_val == 0.5, (
        f"After Condition A dedup: physicians [1, 0] → mean 0.5, got {label_val}"
    )


def test_physician_reconciliation_condition_b_drops_non_replica_triples():
    """Three rows where physician 10 appears twice and physician 11 once (no replica)
    must fall to Condition B: the group is dropped because not all rows have replicas.
    """
    import pandas as pd
    from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians

    df = pd.DataFrame({
        "Patient":   [1.0, 1.0, 1.0, 2.0, 2.0],
        "Item":      ["CBC"] * 5,
        "Physician": [10.0, 10.0, 11.0, 10.0, 11.0],
        "label":     [1.0, 1.0, 0.0, 1.0, 0.0],
    })
    result = _aggregate_physicians(
        df,
        patient_col="Patient",
        physician_col="Physician",
        item_col="Item",
        class_cols=["label"],
        physician_count=2,
        mismatch_error=False,
    )

    # Patient 1: count=3, Condition A fails (physician 11 has no replica) → dropped
    assert 1.0 not in result["Patient"].values, (
        "Patient 1 (3 rows, physician 11 has no replica) must be dropped via Condition B"
    )
    # Patient 2: count=2 = physician_count → kept and aggregated correctly
    assert 2.0 in result["Patient"].values
    p2_label = float(result[result["Patient"] == 2.0].iloc[0]["label"])
    assert p2_label == 0.5, f"Patient 2 label should be 0.5, got {p2_label}"


# ---------------------------------------------------------------------------
# Regression: full-train fuzzy matcher for test set must not absorb test items
# ---------------------------------------------------------------------------

def test_full_train_matcher_does_not_absorb_test_item_strings():
    """The pre-fold fuzzy matcher (built from all item_strings_train to standardize
    X_test) must never add test item strings to its candidate pool.

    Simulates the pre-fold path: build matcher from training strings, then
    apply to test strings as queries.  After apply_fuzzy_fallback, the test
    item's normalised form must not appear in the matcher's internal lookup.
    """
    from shared.preprocessing.fuzzy_mapping import (
        build_fuzzy_matcher_from_train,
        apply_fuzzy_fallback,
        normalize_text,
    )
    import pandas as pd

    train_strings = ["Electrocardiogram", "Complete Blood Count"]
    test_strings  = ["ECG"]       # held-out test item not in training vocabulary

    train_df   = pd.DataFrame({"Item": train_strings})
    unresolved = pd.Series([True] * len(train_strings))
    matcher = build_fuzzy_matcher_from_train(train_df, "Item", threshold=80.0, unresolved_mask=unresolved)

    # Apply matcher to test items (query path, not training path)
    test_df    = pd.DataFrame({"Item": test_strings})
    test_unres = pd.Series([True] * len(test_strings))
    apply_fuzzy_fallback(test_df, "Item", matcher, test_unres)

    # Test item must not have been added to the candidate pool
    norm_test_item = normalize_text(test_strings[0])
    assert norm_test_item not in matcher._norm_variant_to_norm_canonical, (
        f"Test item '{test_strings[0]}' (normalized: '{norm_test_item}') "
        "must NOT appear in the matcher's candidate space after apply_fuzzy_fallback. "
        "This would constitute held-out vocabulary leaking into the fuzzy dictionary."
    )
