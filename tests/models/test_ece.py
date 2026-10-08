import numpy as np
import pytest

from tracks.representation.training.shared.soft_label_utils import (
    _soft_ece, soft_ece_with_ci,
    _soft_brier_score,
    masked_class_data,
)
from tracks.representation.training.orchestrator.compute_metrics_and_save import compute_metrics_df


class TestSoftEce:
    def test_perfect_calibration_returns_zero(self):
        # When predicted probs equal true labels in each bin, ECE = 0.
        y = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9], dtype=np.float32)
        p = y.copy()
        assert _soft_ece(y, p) == pytest.approx(0.0, abs=1e-6)

    def test_inverted_predictions_large_ece(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, 200).astype(np.float32)
        # Invert probabilities so miscalibration is maximised.
        p = (1.0 - y).astype(np.float32)
        assert _soft_ece(y, p) > 0.3

    def test_empty_array_returns_nan(self):
        assert np.isnan(_soft_ece(np.array([]), np.array([])))

    def test_constant_probs_returns_nan(self):
        y = np.array([0.2, 0.8], dtype=np.float32)
        p = np.full(2, 0.5, dtype=np.float32)
        assert np.isnan(_soft_ece(y, p))

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 1, 100).astype(np.float32)
        p = rng.uniform(0, 1, 100).astype(np.float32)
        result = _soft_ece(y, p)
        assert 0.0 <= result <= 1.0

    def test_soft_labels_accepted(self):
        y = np.array([0.3, 0.7, 0.5], dtype=np.float32)
        p = np.array([0.3, 0.7, 0.5], dtype=np.float32)
        result = _soft_ece(y, p)
        assert np.isfinite(result) or np.isnan(result)  # no crash; constant check may trigger

    def test_single_observation(self):
        y = np.array([1.0], dtype=np.float32)
        p = np.array([0.8], dtype=np.float32)
        # One sample, one bin: ECE = |0.8 - 1.0| * (1/1) = 0.2
        assert _soft_ece(y, p) == pytest.approx(0.2, abs=1e-6)

    def test_empty_bins_skipped_not_nan(self):
        # Sparse data → many empty bins; result should still be finite.
        y = np.array([0.01, 0.99], dtype=np.float32)
        p = np.array([0.02, 0.98], dtype=np.float32)
        result = _soft_ece(y, p)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_float64_input_coerced(self):
        y = np.array([0.2, 0.8], dtype=np.float64)
        p = np.array([0.3, 0.7], dtype=np.float64)
        result = _soft_ece(y, p)
        assert isinstance(result, float)

    def test_n_bins_parameter_accepted(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0, 1, 100).astype(np.float32)
        p = rng.uniform(0, 1, 100).astype(np.float32)
        r5  = _soft_ece(y, p, n_bins=5)
        r30 = _soft_ece(y, p, n_bins=30)
        # Both should be finite; no crash.
        assert np.isfinite(r5)
        assert np.isfinite(r30)


class TestSoftEceWithCi:
    def _valid_arrays(self, n=200, seed=3):
        rng = np.random.default_rng(seed)
        y = rng.uniform(0, 1, n).astype(np.float32)
        p = rng.uniform(0, 1, n).astype(np.float32)
        return y, p

    def test_returns_three_tuple(self):
        y, p = self._valid_arrays()
        ids = np.arange(len(y))
        result = soft_ece_with_ci(y, p, ids, n_resamples=50, rng=np.random.default_rng(0))
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ci_ordered(self):
        y, p = self._valid_arrays()
        ids = np.arange(len(y))
        _, ci_lo, ci_hi = soft_ece_with_ci(y, p, ids, n_resamples=200, rng=np.random.default_rng(0))
        assert ci_lo <= ci_hi

    def test_deterministic_with_seeded_rng(self):
        y, p = self._valid_arrays()
        ids = np.arange(len(y))
        r1 = soft_ece_with_ci(y, p, ids, n_resamples=100, rng=np.random.default_rng(42))
        r2 = soft_ece_with_ci(y, p, ids, n_resamples=100, rng=np.random.default_rng(42))
        assert r1 == r2

    def test_empty_returns_all_nan(self):
        result = soft_ece_with_ci(np.array([]), np.array([]), np.array([], dtype=int))
        assert all(np.isnan(v) for v in result)

    def test_single_patient_ci_is_nan(self):
        y, p = self._valid_arrays(n=50)
        ids = np.zeros(50, dtype=int)  # all rows belong to one patient
        _, ci_lo, ci_hi = soft_ece_with_ci(y, p, ids, rng=np.random.default_rng(0))
        assert np.isnan(ci_lo)
        assert np.isnan(ci_hi)

    def test_single_patient_still_returns_point_estimate(self):
        y, p = self._valid_arrays(n=50)
        ids = np.zeros(50, dtype=int)
        ece_pt, _, _ = soft_ece_with_ci(y, p, ids, rng=np.random.default_rng(0))
        assert np.isfinite(ece_pt)

    def test_point_estimate_matches_soft_ece_directly(self):
        y, p = self._valid_arrays()
        ids = np.arange(len(y))
        ece_pt, _, _ = soft_ece_with_ci(y, p, ids, n_resamples=50, rng=np.random.default_rng(0))
        assert ece_pt == pytest.approx(_soft_ece(y, p), abs=1e-7)

    def test_ci_width_nonzero(self):
        y, p = self._valid_arrays(n=300)
        ids = np.arange(len(y))
        _, ci_lo, ci_hi = soft_ece_with_ci(y, p, ids, n_resamples=500, rng=np.random.default_rng(5))
        assert ci_hi - ci_lo > 0.0

    def test_all_results_finite_on_valid_input(self):
        y, p = self._valid_arrays()
        ids = np.arange(len(y))
        result = soft_ece_with_ci(y, p, ids, n_resamples=100, rng=np.random.default_rng(7))
        assert all(np.isfinite(v) for v in result)

    def test_raises_when_patient_ids_none(self):
        y, p = self._valid_arrays()
        with pytest.raises(ValueError, match="patient_ids is required"):
            soft_ece_with_ci(y, p, None)


class TestEceInMetricsDf:
    def _make_cfg(self):
        return {
            "eval_pos_threshold": 0.5,
            "ece_n_bins": 15,
            "ece_n_resamples": 100,  # fast for tests
        }

    def _make_inputs(self, n=60, n_classes=2, seed=0):
        rng = np.random.default_rng(seed)
        probs  = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        y_true = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        thresholds = np.full(n_classes, 0.5, dtype=np.float32)
        class_list = [f"class_{i}" for i in range(n_classes)]
        patient_ids = np.arange(n)
        return probs, y_true, thresholds, class_list, patient_ids

    def test_ece_columns_present(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        assert "ECE" in df.columns
        assert "ECE CI Lower" in df.columns
        assert "ECE CI Upper" in df.columns

    def test_ece_per_class_rows_finite(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]
        assert per_class["ECE"].notna().all()
        assert per_class["ECE CI Lower"].notna().all()
        assert per_class["ECE CI Upper"].notna().all()

    def test_macro_average_ece_is_mean_of_per_class(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]["ECE"]
        macro_ece = df[df["Class"] == "Macro Average"].iloc[0]["ECE"]
        assert macro_ece == pytest.approx(float(per_class.mean()), abs=1e-4)

    def test_micro_aggregate_ece_is_nan(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        micro = df[df["Class"] == "Micro Aggregate"].iloc[0]
        assert np.isnan(micro["ECE"])
        assert np.isnan(micro["ECE CI Lower"])
        assert np.isnan(micro["ECE CI Upper"])

    def test_ece_rounded_to_4_decimals(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        for col in ("ECE", "ECE CI Lower", "ECE CI Upper"):
            finite_vals = df[col].dropna()
            assert (finite_vals.round(4) == finite_vals).all()

    def test_cfg_ece_keys_override_defaults(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        cfg = self._make_cfg()
        cfg["ece_n_bins"] = 5
        cfg["ece_n_resamples"] = 50
        df = compute_metrics_df(probs, y_true, thresholds, class_list, cfg, patient_ids)
        assert "ECE" in df.columns

    def test_ci_bounds_ordered_per_class(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs(n=60)
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]
        assert (per_class["ECE CI Lower"] <= per_class["ECE CI Upper"]).all()


class TestSoftBrierScore:
    def test_perfect_predictions_return_zero(self):
        y = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
        p = y.copy()
        assert _soft_brier_score(y, p) == pytest.approx(0.0, abs=1e-6)

    def test_empty_array_returns_nan(self):
        assert np.isnan(_soft_brier_score(np.array([]), np.array([])))

    def test_result_nonnegative(self):
        rng = np.random.default_rng(10)
        y = rng.uniform(0, 1, 100).astype(np.float32)
        p = rng.uniform(0, 1, 100).astype(np.float32)
        assert _soft_brier_score(y, p) >= 0.0

    def test_soft_labels_accepted(self):
        y = np.array([0.3, 0.6, 0.2, 0.9], dtype=np.float32)
        p = np.array([0.4, 0.5, 0.3, 0.8], dtype=np.float32)
        assert np.isfinite(_soft_brier_score(y, p))


class TestBrierScoreInMetricsDf:
    def _make_cfg(self):
        return {
            "eval_pos_threshold": 0.5,
            "ece_n_bins": 15,
            "ece_n_resamples": 100,  # fast for tests
        }

    def _make_inputs(self, n=60, n_classes=2, seed=0):
        rng = np.random.default_rng(seed)
        probs  = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        y_true = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        thresholds = np.full(n_classes, 0.5, dtype=np.float32)
        class_list = [f"class_{i}" for i in range(n_classes)]
        patient_ids = np.arange(n)
        return probs, y_true, thresholds, class_list, patient_ids

    def test_brier_columns_present(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        assert "Brier Score" in df.columns
        assert "Brier CI Lower" in df.columns
        assert "Brier CI Upper" in df.columns

    def test_brier_per_class_rows_finite(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]
        assert per_class["Brier Score"].notna().all()
        assert per_class["Brier CI Lower"].notna().all()
        assert per_class["Brier CI Upper"].notna().all()

    def test_macro_average_brier_is_mean_of_per_class(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]["Brier Score"]
        macro_brier = df[df["Class"] == "Macro Average"].iloc[0]["Brier Score"]
        assert macro_brier == pytest.approx(float(per_class.mean()), abs=1e-4)

    def test_micro_aggregate_brier_is_nan(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        micro = df[df["Class"] == "Micro Aggregate"].iloc[0]
        assert np.isnan(micro["Brier Score"])
        assert np.isnan(micro["Brier CI Lower"])
        assert np.isnan(micro["Brier CI Upper"])

    def test_brier_rounded_to_4_decimals(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs()
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        for col in ("Brier Score", "Brier CI Lower", "Brier CI Upper"):
            finite_vals = df[col].dropna()
            assert (finite_vals.round(4) == finite_vals).all()

    def test_ci_bounds_ordered_per_class(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs(n=60)
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]
        assert (per_class["Brier CI Lower"] <= per_class["Brier CI Upper"]).all()

    def test_perfect_predictions_brier_near_zero(self):
        rng = np.random.default_rng(99)
        y_true = rng.uniform(0.1, 0.9, (60, 2)).astype(np.float32)
        probs  = y_true.copy()
        thresholds = np.full(2, 0.5, dtype=np.float32)
        patient_ids = np.arange(60)
        df = compute_metrics_df(probs, y_true, thresholds, ["c0", "c1"], self._make_cfg(), patient_ids)
        per_class = df[~df["Class"].isin(["Macro Average", "Micro Aggregate"])]
        assert all(v == pytest.approx(0.0, abs=1e-4) for v in per_class["Brier Score"])

    def test_brier_point_matches_direct_calculation(self):
        probs, y_true, thresholds, class_list, patient_ids = self._make_inputs(n=60)
        df = compute_metrics_df(probs, y_true, thresholds, class_list, self._make_cfg(), patient_ids)
        y_c = y_true[:, 0].astype(np.float32)
        p_c = probs[:, 0].astype(np.float32)
        expected = float(np.mean((y_c - p_c) ** 2))
        actual = float(df[df["Class"] == class_list[0]].iloc[0]["Brier Score"])
        assert actual == pytest.approx(expected, abs=1e-4)


class TestMaskedClassData:
    """Tests for masked_class_data — the soft-label mid-band exclusion filter."""

    def _make_inputs(self):
        # y column 0: [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        # y column 1: ignored (class_idx=0 targets col 0)
        y = np.array([
            [0.0, 0.9],
            [0.2, 0.8],
            [0.4, 0.7],
            [0.5, 0.5],
            [0.6, 0.3],
            [0.8, 0.1],
            [1.0, 0.0],
        ], dtype=np.float32)
        p = np.arange(14, dtype=np.float32).reshape(7, 2) / 14.0
        return y, p

    def test_default_threshold_includes_all(self):
        y, p = self._make_inputs()
        mask, y_col, p_col = masked_class_data(y, p, 0, 0.5)
        assert mask.all()
        assert len(y_col) == 7
        assert len(p_col) == 7

    def test_float_threshold_excludes_midband(self):
        y, p = self._make_inputs()
        # threshold=0.7 keeps y>=0.7 or y<=0.3; excludes 0.3 < y < 0.7
        mask, y_col, p_col = masked_class_data(y, p, 0, 0.7)
        expected_y = np.array([0.0, 0.2, 0.8, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(y_col, expected_y)
        assert p_col.shape == (4,)

    def test_dict_threshold_same_as_float(self):
        y, p = self._make_inputs()
        _, y_float, _ = masked_class_data(y, p, 0, 0.7)
        _, y_dict, _ = masked_class_data(y, p, 0, {"eval_pos_threshold": 0.7})
        np.testing.assert_array_equal(y_float, y_dict)

    def test_dict_missing_key_falls_back_to_half(self):
        y, p = self._make_inputs()
        mask_default, _, _ = masked_class_data(y, p, 0, 0.5)
        mask_missing, _, _ = masked_class_data(y, p, 0, {})
        np.testing.assert_array_equal(mask_default, mask_missing)

    def test_class_idx_selects_correct_column(self):
        y, p = self._make_inputs()
        # class_idx=1 targets y[:,1] = [0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0]
        # threshold=0.6: keep y>=0.6 | y<=0.4 → [0.9, 0.8, 0.7, 0.3, 0.1, 0.0]
        mask, y_col, p_col = masked_class_data(y, p, 1, 0.6)
        np.testing.assert_array_equal(y_col, np.array([0.9, 0.8, 0.7, 0.3, 0.1, 0.0], dtype=np.float32))
        # p_col must come from p[:,1], not p[:,0]
        np.testing.assert_array_equal(p_col, p[mask, 1])

    def test_returned_mask_indexes_correctly(self):
        y, p = self._make_inputs()
        mask, y_col, p_col = masked_class_data(y, p, 0, 0.7)
        np.testing.assert_array_equal(y_col, y[mask, 0])
        np.testing.assert_array_equal(p_col, p[mask, 0])

    def test_threshold_one_keeps_only_extreme_labels(self):
        # threshold=1.0: keep y>=1.0 | y<=0.0 — only hard labels
        y, p = self._make_inputs()
        mask, y_col, _ = masked_class_data(y, p, 0, 1.0)
        assert set(y_col.tolist()) <= {0.0, 1.0}

    def test_symmetric_band_exclusion(self):
        # threshold=0.6: mask keeps y>=0.6 or y<=0.4, so y=0.5 (the only value
        # strictly inside the exclusion band) must be absent from the output.
        y, p = self._make_inputs()
        mask, y_col, _ = masked_class_data(y, p, 0, 0.6)
        assert np.float32(0.5) not in y_col
