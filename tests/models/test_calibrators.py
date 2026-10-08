"""Tests for tracks/representation/training/shared/fit_calibrators and apply_calibrators."""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from tracks.representation.training.shared.fit_calibrators import fit_calibrators
from tracks.representation.training.shared.apply_calibrators import apply_calibrators
from tracks.representation.models.ConstantCalibrator import ConstantCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_data(n=40, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.random((n, n_classes)).astype(np.float32)
    Y = rng.integers(0, 2, size=(n, n_classes)).astype(np.float32)
    return probs, Y


def _cfg(method="isotonic", min_samples=5):
    return {"calibration_method": method, "calibration_min_samples": min_samples}


# ---------------------------------------------------------------------------
# fit_calibrators
# ---------------------------------------------------------------------------

class TestFitCalibrators:
    def test_normal_case_returns_isotonic(self):
        probs, Y = _make_binary_data(n=50, n_classes=2)
        calibrators = fit_calibrators(probs, Y, ["cls_a", "cls_b"], _cfg())
        assert isinstance(calibrators["cls_a"], IsotonicRegression)
        assert isinstance(calibrators["cls_b"], IsotonicRegression)

    def test_method_none_returns_none_per_class(self):
        probs, Y = _make_binary_data(n=30, n_classes=2)
        calibrators = fit_calibrators(probs, Y, ["x", "y"], _cfg(method="none"))
        assert calibrators["x"] is None
        assert calibrators["y"] is None

    def test_fallback_empty_validation_set(self):
        # Zero samples → ConstantCalibrator(0.5)
        probs = np.empty((0, 1), dtype=np.float32)
        Y = np.empty((0, 1), dtype=np.float32)
        calibrators = fit_calibrators(probs, Y, ["cls"], _cfg())
        cal = calibrators["cls"]
        assert isinstance(cal, ConstantCalibrator)
        assert cal.prob == 0.5

    def test_fallback_single_class_labels(self):
        # All-positive validation labels → ConstantCalibrator(mean=1.0)
        n = 20
        probs = np.full((n, 1), 0.8, dtype=np.float32)
        Y = np.ones((n, 1), dtype=np.float32)
        calibrators = fit_calibrators(probs, Y, ["cls"], _cfg())
        cal = calibrators["cls"]
        assert isinstance(cal, ConstantCalibrator)
        assert cal.prob == pytest.approx(1.0)

    def test_fallback_all_negative_labels(self):
        # All-negative validation labels → ConstantCalibrator(mean=0.0)
        n = 20
        probs = np.full((n, 1), 0.2, dtype=np.float32)
        Y = np.zeros((n, 1), dtype=np.float32)
        calibrators = fit_calibrators(probs, Y, ["cls"], _cfg())
        cal = calibrators["cls"]
        assert isinstance(cal, ConstantCalibrator)
        assert cal.prob == pytest.approx(0.0)

    def test_fallback_too_few_samples(self):
        # 3 samples, min_samples=10 → ConstantCalibrator
        n = 3
        probs = np.array([[0.2], [0.8], [0.5]], dtype=np.float32)
        Y = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
        calibrators = fit_calibrators(probs, Y, ["cls"], _cfg(min_samples=10))
        assert isinstance(calibrators["cls"], ConstantCalibrator)

    def test_unsupported_method_raises(self):
        probs, Y = _make_binary_data(n=30, n_classes=1)
        with pytest.raises(ValueError, match="Unsupported calibration_method"):
            fit_calibrators(probs, Y, ["cls"], {"calibration_method": "sigmoid"})

    def test_returns_dict_with_correct_keys(self):
        probs, Y = _make_binary_data(n=30, n_classes=3)
        class_list = ["a", "b", "c"]
        calibrators = fit_calibrators(probs, Y, class_list, _cfg())
        assert set(calibrators.keys()) == set(class_list)

    def test_calibrator_fitted_on_validation_data(self):
        # IsotonicRegression should be callable after fitting
        probs, Y = _make_binary_data(n=40, n_classes=1)
        calibrators = fit_calibrators(probs, Y, ["cls"], _cfg())
        cal = calibrators["cls"]
        # Should be able to predict without error
        result = cal.predict(probs[:, 0])
        assert result.shape == (40,)


# ---------------------------------------------------------------------------
# apply_calibrators
# ---------------------------------------------------------------------------

class TestApplyCalibrators:
    def test_output_shape_matches_input(self):
        probs = np.random.rand(10, 3).astype(np.float32)
        calibrators = {c: None for c in ["a", "b", "c"]}
        out = apply_calibrators(calibrators, probs, ["a", "b", "c"])
        assert out.shape == probs.shape

    def test_none_calibrator_is_passthrough(self):
        probs = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
        calibrators = {"x": None, "y": None}
        out = apply_calibrators(calibrators, probs, ["x", "y"])
        np.testing.assert_allclose(out, probs)

    def test_constant_calibrator_via_predict_proba(self):
        # ConstantCalibrator has predict_proba → dispatches to predict_proba path
        n = 5
        probs = np.random.rand(n, 1).astype(np.float32)
        cal = ConstantCalibrator(0.3)
        calibrators = {"cls": cal}
        out = apply_calibrators(calibrators, probs, ["cls"])
        np.testing.assert_allclose(out[:, 0], 0.3, rtol=1e-6)

    def test_isotonic_calibrator_via_predict(self):
        # IsotonicRegression has no predict_proba → dispatches to predict path
        n = 20
        rng = np.random.default_rng(7)
        raw = rng.random((n, 1)).astype(np.float32)
        y = (raw[:, 0] > 0.5).astype(np.float32)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw[:, 0], y)
        calibrators = {"cls": iso}
        out = apply_calibrators(calibrators, raw, ["cls"])
        assert out.shape == (n, 1)
        # IsotonicRegression output is already in [0,1]; clip shouldn't change it
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_output_clipped_to_unit_interval(self):
        # Build a calibrator that returns values outside [0, 1]
        class OutOfRangeCal:
            def predict(self, X):
                return np.full(X.shape[0], 1.5)

        probs = np.ones((4, 1), dtype=np.float32)
        calibrators = {"cls": OutOfRangeCal()}
        out = apply_calibrators(calibrators, probs, ["cls"])
        assert (out <= 1.0).all()

    def test_multiple_classes_each_calibrated_independently(self):
        n = 10
        probs = np.column_stack([
            np.full(n, 0.2),
            np.full(n, 0.8),
        ]).astype(np.float32)
        calibrators = {
            "low": ConstantCalibrator(0.1),
            "high": ConstantCalibrator(0.9),
        }
        out = apply_calibrators(calibrators, probs, ["low", "high"])
        np.testing.assert_allclose(out[:, 0], 0.1, rtol=1e-6)
        np.testing.assert_allclose(out[:, 1], 0.9, rtol=1e-6)
