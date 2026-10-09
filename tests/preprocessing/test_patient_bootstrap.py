"""Tests for patient-level cluster bootstrap CIs in soft_ece_with_ci and soft_brier_with_ci."""

import numpy as np
import pytest

from tracks.representation.training.shared.soft_label_utils import soft_ece_with_ci, soft_brier_with_ci


class TestClusterBootstrapEce:
    def test_cluster_bootstrap_resamples_patient_blocks(self):
        """When all rows of a patient have identical values, every bootstrap
        replicate that includes that patient preserves those values."""
        rng = np.random.default_rng(0)
        # 3 patients, 4 rows each — patient values are distinct and constant per patient
        n_per_patient = 4
        y_pt = np.array([0.1, 0.5, 0.9])
        p_pt = np.array([0.15, 0.45, 0.85])
        patient_ids = np.repeat(np.arange(3), n_per_patient)
        y = np.repeat(y_pt, n_per_patient).astype(np.float32)
        p = np.repeat(p_pt, n_per_patient).astype(np.float32)

        # Patch _soft_ece to capture the idx arrays passed to it
        from tracks.representation.training.shared import soft_label_utils
        captured_ys = []
        original_ece = soft_label_utils._soft_ece

        def capturing_ece(y_arr, p_arr, n_bins=15):
            captured_ys.append(y_arr.copy())
            return original_ece(y_arr, p_arr, n_bins)

        soft_label_utils._soft_ece = capturing_ece
        try:
            soft_ece_with_ci(y, p, patient_ids, n_resamples=10, rng=rng)
        finally:
            soft_label_utils._soft_ece = original_ece

        # Skip the first call (point estimate). Each bootstrap resample should
        # contain only values that appear as whole patient blocks.
        # Use float32 values to avoid float64-vs-float32 precision mismatches.
        valid_values = set(np.unique(y).tolist())
        for boot_y in captured_ys[1:]:
            assert all(v in valid_values for v in np.unique(boot_y).tolist())

    def test_duplicate_patient_doubles_block_length(self):
        """When a patient appears twice in a resample its block length doubles."""
        # 2 patients, 3 rows each
        y = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float32)
        p = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8], dtype=np.float32)
        patient_ids = np.array([0, 0, 0, 1, 1, 1])

        from tracks.representation.training.shared import soft_label_utils
        sample_sizes = []
        original_ece = soft_label_utils._soft_ece

        def size_capturing_ece(y_arr, p_arr, n_bins=15):
            sample_sizes.append(len(y_arr))
            return original_ece(y_arr, p_arr, n_bins)

        soft_label_utils._soft_ece = size_capturing_ece
        try:
            soft_ece_with_ci(y, p, patient_ids, n_resamples=50, rng=np.random.default_rng(1))
        finally:
            soft_label_utils._soft_ece = original_ece

        # Each bootstrap resample samples 2 patients with replacement from {0,1}.
        # Possible resample sizes: 3+3=6 (one of each) or 3+3=6 (same patient twice).
        # With 2 patients of 3 rows each, the resample always has exactly 6 rows.
        for sz in sample_sizes[1:]:  # skip point estimate
            assert sz == 6

    def test_soft_ece_raises_when_patient_ids_none(self):
        y = np.array([0.2, 0.8, 0.3], dtype=np.float32)
        p = np.array([0.3, 0.7, 0.4], dtype=np.float32)
        with pytest.raises(ValueError, match="patient_ids is required"):
            soft_ece_with_ci(y, p, None)

    def test_soft_brier_raises_when_patient_ids_none(self):
        y = np.array([0.2, 0.8, 0.3], dtype=np.float32)
        p = np.array([0.3, 0.7, 0.4], dtype=np.float32)
        with pytest.raises(ValueError, match="patient_ids is required"):
            soft_brier_with_ci(y, p, None)

    def test_single_unique_patient_ece_returns_point_and_nan_ci(self):
        y = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        p = np.array([0.2, 0.4, 0.8, 0.35], dtype=np.float32)
        ids = np.zeros(4, dtype=int)  # all same patient
        ece_pt, ci_lo, ci_hi = soft_ece_with_ci(y, p, ids, rng=np.random.default_rng(0))
        assert np.isfinite(ece_pt)
        assert np.isnan(ci_lo)
        assert np.isnan(ci_hi)

    def test_single_unique_patient_brier_returns_point_and_nan_ci(self):
        y = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        p = np.array([0.2, 0.4, 0.8, 0.35], dtype=np.float32)
        ids = np.zeros(4, dtype=int)
        brier_pt, ci_lo, ci_hi = soft_brier_with_ci(y, p, ids, rng=np.random.default_rng(0))
        assert np.isfinite(brier_pt)
        assert np.isnan(ci_lo)
        assert np.isnan(ci_hi)
