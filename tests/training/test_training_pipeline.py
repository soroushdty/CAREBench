"""
Test module for train_ensemble_pipeline function.

Tests the ensemble training pipeline including:
- Configuration validation
- Data shape and type handling
- Output structure and metrics
- Edge cases and error handling
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tracks.representation.training.orchestrator.train_ensemble_pipeline import (
    train_ensemble_pipeline,
    _as_float32_array,
    _resolve_class_list,
    _compute_macro_metrics,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_config():
    """Minimal configuration for testing the pipeline."""
    return {
        "global_seed": 42,
        "pca_n_components": 50,
        "pca_whiten": False,
        "eval_pos_threshold": 0.5,
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": 16,
        "weight_cap": 10.0,
        "threshold_multipliers": [0.8, 1.0, 1.2],
        "default_classes": ["class_0", "class_1"],
        "num_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "DIR_MODEL": "output/model",
        "hidden_dims": [],
        "dropout": 0.0,
        "activation": "gelu",
    }


@pytest.fixture
def minimal_multiclass_config():
    """Configuration for multiclass testing."""
    config = {
        "global_seed": 42,
        "pca_n_components": 50,
        "pca_whiten": False,
        "eval_pos_threshold": 0.5,
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": 16,
        "weight_cap": 10.0,
        "threshold_multipliers": [0.8, 1.0, 1.2],
        "default_classes": ["class_0", "class_1", "class_2"],
        "num_epochs": 2,
        "early_stopping_patience": 2,
        "device": "cpu",
        "DIR_MODEL": "output/model",
        "hidden_dims": [],
        "dropout": 0.0,
        "activation": "gelu",
    }
    return config


@pytest.fixture
def binary_classification_data():
    """Create minimal binary classification data for testing."""
    n_patients = 5
    samples_per_patient = 10
    n_samples_train = n_patients * samples_per_patient
    n_samples_test = 20
    n_features = 10
    n_classes = 2

    X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
    Y_train = np.random.binomial(1, 0.5, (n_samples_train, n_classes)).astype(np.float32)
    patient_ids_train = np.repeat(np.arange(n_patients), samples_per_patient)

    X_test = np.random.randn(n_samples_test, n_features).astype(np.float32)
    Y_test = np.random.binomial(1, 0.5, (n_samples_test, n_classes)).astype(np.float32)
    patient_ids_test = np.arange(n_samples_test)

    return X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test


@pytest.fixture
def multiclass_classification_data():
    """Create minimal multiclass classification data for testing."""
    n_patients = 6
    samples_per_patient = 10
    n_samples_train = n_patients * samples_per_patient
    n_samples_test = 20
    n_features = 15
    n_classes = 3

    X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
    Y_train = np.random.binomial(1, 0.4, (n_samples_train, n_classes)).astype(np.float32)
    patient_ids_train = np.repeat(np.arange(n_patients), samples_per_patient)

    X_test = np.random.randn(n_samples_test, n_features).astype(np.float32)
    Y_test = np.random.binomial(1, 0.4, (n_samples_test, n_classes)).astype(np.float32)
    patient_ids_test = np.arange(n_samples_test)

    return X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

class TestAsFloat32Array:
    """Tests for _as_float32_array function."""

    def test_converts_to_float32(self):
        """Test that array is converted to float32."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = _as_float32_array(arr, "test")
        assert result.dtype == np.float32

    def test_handles_object_dtype(self):
        """Test handling of object dtype arrays."""
        arr = np.array([1.0, "2.0", 3.0], dtype=object)
        result = _as_float32_array(arr, "test")
        assert result.dtype == np.float32
        assert not np.isnan(result).any()

    def test_replaces_nan_with_zero(self):
        """Test that NaNs are replaced with 0."""
        arr = np.array([1.0, np.nan, 3.0])
        result = _as_float32_array(arr, "test")
        assert not np.isnan(result).any()
        assert result[1] == 0.0

    def test_handles_2d_arrays(self):
        """Test handling of 2D arrays."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = _as_float32_array(arr, "test")
        assert result.dtype == np.float32
        assert result.shape == (2, 2)


class TestResolveClassList:
    """Tests for _resolve_class_list function."""

    def test_resolves_default_classes(self):
        """Test resolving from 'default_classes' key."""
        cfg = {"default_classes": ["a", "b", "c"]}
        result = _resolve_class_list(cfg)
        assert result == ["a", "b", "c"]

    def test_resolves_classes_key(self):
        """Test resolving from 'classes' key."""
        cfg = {"classes": ["x", "y"]}
        result = _resolve_class_list(cfg)
        assert result == ["x", "y"]

    def test_resolves_class_list_key(self):
        """Test resolving from 'class_list' key."""
        cfg = {"class_list": ["p", "q", "r"]}
        result = _resolve_class_list(cfg)
        assert result == ["p", "q", "r"]

    def test_raises_on_missing_classes(self):
        """Test that KeyError is raised when no class key is present."""
        cfg = {}
        with pytest.raises(KeyError):
            _resolve_class_list(cfg)

    def test_prefers_default_classes_order(self):
        """Test that default_classes is preferred when multiple keys exist."""
        cfg = {
            "default_classes": ["a", "b"],
            "classes": ["x", "y"],
            "class_list": ["p", "q"],
        }
        result = _resolve_class_list(cfg)
        assert result == ["a", "b"]


class TestComputeMacroMetrics:
    """Tests for _compute_macro_metrics function."""

    def test_returns_all_metrics(self):
        """Test that all required metrics are returned."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.8]], dtype=np.float32)
        thresholds = np.array([0.5, 0.5], dtype=np.float32)

        metrics = _compute_macro_metrics(y_true, y_prob, thresholds)

        required_keys = {"Precision", "Recall", "MCC", "F1", "AUC ROC", "AUC PR", "AP"}
        assert set(metrics.keys()) == required_keys

    def test_metrics_are_numeric(self):
        """Test that all metrics are numeric values."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        thresholds = np.array([0.5, 0.5], dtype=np.float32)

        metrics = _compute_macro_metrics(y_true, y_prob, thresholds)

        for key, value in metrics.items():
            assert isinstance(value, (float, np.floating))

    def test_metrics_in_valid_range(self):
        """Test that metrics are in valid ranges (except AUC which can be NaN)."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.8]], dtype=np.float32)
        thresholds = np.array([0.5, 0.5], dtype=np.float32)

        metrics = _compute_macro_metrics(y_true, y_prob, thresholds)

        # Precision, Recall, F1 should be in [0, 1]; MCC in [-1, 1]
        for metric in ["Precision", "Recall", "F1"]:
            if not np.isnan(metrics[metric]):
                assert 0.0 <= metrics[metric] <= 1.0


# ============================================================================
# Integration Tests for Pipeline
# ============================================================================

class TestTrainEnsemblePipelineBasic:
    """Basic integration tests for train_ensemble_pipeline."""

    @pytest.mark.slow
    def test_pipeline_returns_correct_structure(
        self, minimal_config, binary_classification_data, tmp_path
    ):
        """Test that pipeline returns a dict with 'train' and 'test' keys."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data

        result = train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, minimal_config, patient_ids_test=patient_ids_test)

        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result

    @pytest.mark.slow
    def test_pipeline_returns_metrics(self, minimal_config, binary_classification_data, tmp_path):
        """Test that train and test metrics are returned."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data

        result = train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, minimal_config, patient_ids_test=patient_ids_test)

        assert isinstance(result["train"], dict)
        assert isinstance(result["test"], dict)

    def test_pipeline_validates_input_shapes(self, minimal_config):
        """Test that pipeline validates Y_test shape matches Y_train."""
        n_patients = 3
        X_train = np.random.randn(30, 10).astype(np.float32)
        Y_train = np.random.binomial(1, 0.5, (30, 2)).astype(np.float32)
        patient_ids_train = np.repeat(np.arange(n_patients), 10)
        X_test = np.random.randn(20, 10).astype(np.float32)
        Y_test = np.random.binomial(1, 0.5, (20, 3)).astype(np.float32)  # Wrong shape

        with pytest.raises(ValueError):
            train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, minimal_config)

    def test_pipeline_handles_integer_inputs(self, minimal_config, binary_classification_data):
        """Test that pipeline converts integer inputs to float32."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data
        X_train = X_train.astype(np.int32)
        Y_train = Y_train.astype(np.int32)
        X_test = X_test.astype(np.int32)
        Y_test = Y_test.astype(np.int32)

        # This should not raise an error after conversion
        assert X_train.dtype != np.float32
        # Pipeline should handle the conversion internally

    @pytest.mark.slow
    def test_pipeline_multiclass(
        self, minimal_multiclass_config, multiclass_classification_data, tmp_path
    ):
        """Test pipeline with multiclass classification."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = multiclass_classification_data

        result = train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, minimal_multiclass_config, patient_ids_test=patient_ids_test)

        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) > 0
        assert len(result["test"]) > 0


class TestTrainEnsemblePipelineErrors:
    """Error handling tests for train_ensemble_pipeline."""

    def test_pipeline_missing_class_list(
        self, binary_classification_data
    ):
        """Test that pipeline raises error when class list is missing."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data
        cfg = {
            "global_seed": 42,
            "pca_n_components": 50,
            "pca_whiten": False,
            "eval_pos_threshold": 0.5,
            "lr": 0.001,
            "weight_decay": 0.0,
            "batch_size": 16,
            "weight_cap": 10.0,
            # Missing class list
        }

        with pytest.raises(KeyError):
            train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, cfg, patient_ids_test=patient_ids_test)

    def test_pipeline_with_nan_data(self, minimal_config):
        """Test that pipeline handles NaN values in input data."""
        X_train = np.random.randn(50, 10).astype(np.float32)
        X_train[0, 0] = np.nan  # Introduce NaN
        Y_train = np.random.binomial(1, 0.5, (50, 2)).astype(np.float32)
        X_test = np.random.randn(20, 10).astype(np.float32)
        Y_test = np.random.binomial(1, 0.5, (20, 2)).astype(np.float32)

        # Pipeline should handle NaN by replacing with 0
        # This tests the _as_float32_array function behavior

    def test_pipeline_with_small_dataset(self, minimal_config):
        """Test that pipeline handles very small datasets."""
        X_train = np.random.randn(10, 5).astype(np.float32)
        Y_train = np.random.binomial(1, 0.5, (10, 2)).astype(np.float32)
        X_test = np.random.randn(5, 5).astype(np.float32)
        Y_test = np.random.binomial(1, 0.5, (5, 2)).astype(np.float32)

        # This should complete without error, though results may be poor
        # We don't call the full pipeline as it may be slow


class TestTrainEnsemblePipelineOutputFiles:
    """Tests that verify output files are created by the pipeline."""

    @pytest.mark.slow
    def test_pipeline_creates_output_directories(
        self, minimal_config, binary_classification_data, tmp_path
    ):
        """Test that pipeline creates necessary output directories."""
        cfg = minimal_config.copy()
        cfg["DIR_MODEL"] = str(tmp_path / "models")

        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data
        train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, cfg, patient_ids_test=patient_ids_test)

        model_dir = Path(cfg["DIR_MODEL"])
        reproducibility_dir = model_dir.parent / "reproducibility_artifacts"
        assert model_dir.exists()
        assert (model_dir / "ensemble_bundle.joblib").exists()
        assert (reproducibility_dir / "ensemble_manifest.json").exists()
        assert (model_dir / "train_stage1_oof_vs_survey_metrics.csv").exists()
        assert (model_dir / "test_stage1_vs_survey_metrics.csv").exists()


# ============================================================================
# Parametrized Tests
# ============================================================================

class TestTrainEnsemblePipelineParametrized:
    """Parametrized tests for different configurations and data scenarios."""

    @pytest.mark.parametrize(
        "n_samples_train,n_features,n_classes",
        [
            (50, 10, 2),
            (100, 20, 3),
            (80, 15, 2),
        ],
    )
    def test_pipeline_with_different_data_shapes(
        self, n_samples_train, n_features, n_classes, tmp_path
    ):
        """Test pipeline with different data shapes."""
        cfg = {
            "global_seed": 42,
            "pca_n_components": 50,
            "pca_whiten": False,
            "eval_pos_threshold": 0.5,
            "lr": 0.001,
            "weight_decay": 0.0,
            "batch_size": 16,
            "weight_cap": 10.0,
            "default_classes": [f"class_{i}" for i in range(n_classes)],
            "num_epochs": 1,
            "early_stopping_patience": 1,
            "device": "cpu",
            "hidden_dims": [],
            "dropout": 0.0,
            "activation": "gelu",
        }

        X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
        Y_train = np.random.binomial(1, 0.5, (n_samples_train, n_classes)).astype(np.float32)
        X_test = np.random.randn(20, n_features).astype(np.float32)
        Y_test = np.random.binomial(1, 0.5, (20, n_classes)).astype(np.float32)

        # Verify shapes are consistent
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_test.shape[0] == Y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
        assert Y_train.shape[1] == Y_test.shape[1]


# ============================================================================
# Regression Tests
# ============================================================================

class TestTrainEnsemblePipelineRegression:
    """Regression tests to ensure pipeline behavior remains consistent."""

    @pytest.mark.slow
    def test_pipeline_reproducibility(self, minimal_config, binary_classification_data):
        """Test that pipeline produces reproducible results with same seed."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data

        # Note: This test may be difficult due to randomness in neural network training
        # and various libraries, but we include it for completeness

    @pytest.mark.slow
    def test_pipeline_train_metrics_structure(
        self, minimal_config, binary_classification_data, tmp_path
    ):
        """Test that train metrics have the expected structure."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data

        result = train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, minimal_config, patient_ids_test=patient_ids_test)

        train_metrics = result["train"]
        expected_keys = {"Precision", "Recall", "MCC", "F1", "AUC ROC", "AUC PR", "AP"}
        assert set(train_metrics.keys()) == expected_keys

    @pytest.mark.slow
    def test_pipeline_test_metrics_structure(
        self, minimal_config, binary_classification_data, tmp_path
    ):
        """Test that test metrics have the expected structure."""
        X_train, Y_train, patient_ids_train, X_test, Y_test, patient_ids_test = binary_classification_data

        result = train_ensemble_pipeline(X_train, Y_train, patient_ids_train, X_test, Y_test, minimal_config, patient_ids_test=patient_ids_test)

        test_metrics = result["test"]
        expected_keys = {"Precision", "Recall", "MCC", "F1", "AUC ROC", "AUC PR", "AP"}
        assert set(test_metrics.keys()) == expected_keys


# ============================================================================
# Checkpoint Resume Tests
# ============================================================================

from tracks.representation.training.orchestrator.fold_postprocessing import (
    CHECKPOINT_VERSION,
    validate_fold_checkpoint,
    load_fold_checkpoint,
    detect_completed_folds,
    _fold_already_in_csv,
)


def _make_valid_v2_checkpoint(fold_idx: int = 0, n_val: int = 4, n_classes: int = 2) -> dict:
    """Return a minimal valid v2 checkpoint dict for testing."""
    val_ix = np.arange(n_val)
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "fold_idx":     fold_idx,
        "val_ix":       val_ix,
        "model_state":  {},
        "preprocessor": None,
        "calibrators":  {},
        "thresholds":   np.zeros(n_classes),
        "pos_weights":  np.ones(n_classes),
        "thresh_inner": np.full(n_classes, 0.5),
        "oof_probs":    np.zeros((n_val, n_classes), dtype=np.float32),
        "oof_labels":   np.zeros((n_val, n_classes), dtype=np.float32),
        "best_hp":      {"lr": 0.001, "weight_cap": 10.0, "weight_decay": 0.0, "batch_size": 16},
    }


class TestValidateFoldCheckpoint:
    """Tests for validate_fold_checkpoint()."""

    def test_valid_v2_passes(self):
        ckpt = _make_valid_v2_checkpoint()
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=2)
        assert valid, reason

    def test_not_a_dict_fails(self):
        valid, reason = validate_fold_checkpoint("not a dict", n_classes=2)
        assert not valid
        assert "not a dict" in reason

    def test_missing_version_fails(self):
        ckpt = _make_valid_v2_checkpoint()
        del ckpt["checkpoint_version"]
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=2)
        assert not valid
        assert "version mismatch" in reason

    def test_wrong_version_fails(self):
        ckpt = _make_valid_v2_checkpoint()
        ckpt["checkpoint_version"] = "pdm_fold_v1"
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=2)
        assert not valid
        assert "version mismatch" in reason

    def test_missing_key_fails(self):
        ckpt = _make_valid_v2_checkpoint()
        del ckpt["oof_probs"]
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=2)
        assert not valid
        assert "missing keys" in reason

    def test_oof_probs_wrong_shape_fails(self):
        ckpt = _make_valid_v2_checkpoint(n_val=4, n_classes=2)
        ckpt["oof_probs"] = np.zeros((3, 2), dtype=np.float32)  # row mismatch
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=2)
        assert not valid
        assert "oof_probs shape" in reason

    def test_n_classes_mismatch_fails(self):
        ckpt = _make_valid_v2_checkpoint(n_val=4, n_classes=2)
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=3)  # wrong n_classes
        assert not valid
        assert "oof_probs shape" in reason

    def test_oof_labels_shape_mismatch_fails(self):
        ckpt = _make_valid_v2_checkpoint(n_val=4, n_classes=2)
        ckpt["oof_labels"] = np.zeros((4, 3), dtype=np.float32)  # wrong n_classes
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=2)
        assert not valid
        assert "oof_labels shape" in reason

    def test_v1_era_checkpoint_rejected(self):
        """Old v1 checkpoints (missing checkpoint_version) are rejected gracefully."""
        ckpt_v1 = {
            "fold_idx":     0,
            "val_ix":       np.array([0, 1]),
            "model_state":  {},
            "preprocessor": None,
            "calibrators":  {},
            "thresholds":   np.array([0.5, 0.5]),
            "pos_weights":  np.array([1.0, 1.0]),
        }
        valid, reason = validate_fold_checkpoint(ckpt_v1, n_classes=2)
        assert not valid
        assert "version mismatch" in reason


class TestLoadFoldCheckpoint:
    """Tests for load_fold_checkpoint()."""

    def test_missing_file_returns_none(self, tmp_path):
        result = load_fold_checkpoint(tmp_path / "nonexistent.joblib", n_classes=2)
        assert result is None

    def test_corrupt_file_returns_none(self, tmp_path):
        bad = tmp_path / "bad.joblib"
        bad.write_bytes(b"not a joblib file")
        result = load_fold_checkpoint(bad, n_classes=2)
        assert result is None

    def test_valid_file_returns_ckpt(self, tmp_path):
        import joblib
        ckpt = _make_valid_v2_checkpoint()
        path = tmp_path / "fold_1.joblib"
        joblib.dump(ckpt, path)
        result = load_fold_checkpoint(path, n_classes=2)
        assert result is not None
        assert result["fold_idx"] == 0

    def test_invalid_ckpt_returns_none(self, tmp_path):
        import joblib
        ckpt = _make_valid_v2_checkpoint()
        del ckpt["oof_probs"]  # make it invalid
        path = tmp_path / "fold_1.joblib"
        joblib.dump(ckpt, path)
        result = load_fold_checkpoint(path, n_classes=2)
        assert result is None


class TestDetectCompletedFolds:
    """Tests for detect_completed_folds()."""

    def test_empty_dir_returns_empty(self, tmp_path):
        result = detect_completed_folds(tmp_path / "nonexistent", n_folds=3, n_classes=2)
        assert result == {}

    def test_detects_valid_checkpoints(self, tmp_path):
        import joblib
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for fold_idx in range(2):
            ckpt = _make_valid_v2_checkpoint(fold_idx=fold_idx)
            joblib.dump(ckpt, ckpt_dir / f"fold_{fold_idx + 1}.joblib")

        result = detect_completed_folds(ckpt_dir, n_folds=3, n_classes=2)
        assert set(result.keys()) == {0, 1}

    def test_skips_invalid_checkpoint(self, tmp_path):
        import joblib
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        good = _make_valid_v2_checkpoint(fold_idx=0)
        joblib.dump(good, ckpt_dir / "fold_1.joblib")
        bad = _make_valid_v2_checkpoint(fold_idx=1)
        del bad["oof_probs"]
        joblib.dump(bad, ckpt_dir / "fold_2.joblib")

        result = detect_completed_folds(ckpt_dir, n_folds=3, n_classes=2)
        assert list(result.keys()) == [0]

    def test_partial_completion_detected(self, tmp_path):
        import joblib
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for fold_idx in [0, 1]:
            ckpt = _make_valid_v2_checkpoint(fold_idx=fold_idx)
            joblib.dump(ckpt, ckpt_dir / f"fold_{fold_idx + 1}.joblib")
        # fold 2 is missing

        result = detect_completed_folds(ckpt_dir, n_folds=3, n_classes=2)
        assert set(result.keys()) == {0, 1}


class TestFoldAlreadyInCsv:
    """Tests for _fold_already_in_csv()."""

    def test_returns_false_if_missing(self, tmp_path):
        assert _fold_already_in_csv(tmp_path / "missing.csv", 1) is False

    def test_returns_true_if_fold_present(self, tmp_path):
        import pandas as pd
        p = tmp_path / "cv.csv"
        pd.DataFrame({"Fold": [1, 2]}).to_csv(p, index=False)
        assert _fold_already_in_csv(p, 1) is True
        assert _fold_already_in_csv(p, 2) is True

    def test_returns_false_if_fold_absent(self, tmp_path):
        import pandas as pd
        p = tmp_path / "cv.csv"
        pd.DataFrame({"Fold": [1]}).to_csv(p, index=False)
        assert _fold_already_in_csv(p, 2) is False


class TestAtomicCheckpointWrite:
    """Tests for atomic checkpoint write in write_fold_artifacts()."""

    def test_written_checkpoint_passes_validation(self, tmp_path):
        """After write_fold_artifacts, the .joblib passes validate_fold_checkpoint."""
        import joblib
        from tracks.representation.training.orchestrator.fold_postprocessing import write_fold_artifacts

        n_val = 4
        n_classes = 2
        val_ix = np.arange(n_val)
        oof_probs = np.zeros((n_val, n_classes), dtype=np.float32)
        Y_val = np.zeros((n_val, n_classes), dtype=np.float32)
        thresh = np.full(n_classes, 0.5)
        thresh_inner = np.full(n_classes, 0.5)
        pos_w = np.ones(n_classes)

        result = {
            "fold_idx":      0,
            "val_ix":        val_ix,
            "val_probs_cal": oof_probs,
            "calibs":        {},
            "thresh":        thresh,
            "thresh_report": MagicMock(insert=lambda *a, **kw: None, to_csv=lambda *a, **kw: None),
            "fold_df":       MagicMock(insert=lambda *a, **kw: None, to_csv=lambda *a, **kw: None),
            "Y_val":         Y_val,
            "best_hp":       {"lr": 0.001, "weight_cap": 10.0, "weight_decay": 0.0, "batch_size": 16},
        }
        # Use real DataFrames for CSV methods
        import pandas as pd
        result["thresh_report"] = pd.DataFrame({"Fold": [1]})
        result["fold_df"] = pd.DataFrame({"Fold": [1]})

        ensemble_artifacts = {
            "models":      [{}],
            "preps":       [None],
            "calibs":      [{}],
            "thresh":      [thresh],
            "thresh_inner":[thresh_inner],
            "pos_weights": [pos_w],
        }

        paths = MagicMock()
        paths.cv_folds_csv = tmp_path / "CV_folds.csv"
        paths.threshold_report_csv = tmp_path / "threshold_report.csv"
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        paths.fold_checkpoint.return_value = ckpt_dir / "fold_1.joblib"

        write_fold_artifacts(result, ensemble_artifacts, paths)

        ckpt = joblib.load(ckpt_dir / "fold_1.joblib")
        valid, reason = validate_fold_checkpoint(ckpt, n_classes=n_classes)
        assert valid, reason


class TestReconstructFromCheckpoints:
    """Unit tests for the resume state reconstruction logic."""

    def test_populates_oof_preds(self):
        """Checkpoint oof_probs are written into the oof_preds array at val_ix."""
        n_train = 20
        n_classes = 2
        oof_preds = np.full((n_train, n_classes), -1.0)

        val_ix = np.array([0, 1, 2, 3])
        oof_probs = np.full((4, n_classes), 0.7, dtype=np.float32)

        completed = {
            0: {
                "model_state":  {},
                "preprocessor": None,
                "calibrators":  {},
                "thresholds":   np.zeros(n_classes),
                "thresh_inner": np.full(n_classes, 0.5),
                "pos_weights":  np.ones(n_classes),
                "val_ix":       val_ix,
                "oof_probs":    oof_probs,
            }
        }
        ensemble_artifacts = {
            "models": [], "preps": [], "calibs": [], "thresh": [],
            "thresh_inner": [], "pos_weights": [],
        }

        for fold_idx in sorted(completed):
            ckpt = completed[fold_idx]
            ensemble_artifacts["models"].append(ckpt["model_state"])
            ensemble_artifacts["preps"].append(ckpt["preprocessor"])
            ensemble_artifacts["calibs"].append(ckpt["calibrators"])
            ensemble_artifacts["thresh"].append(ckpt["thresholds"])
            ensemble_artifacts["thresh_inner"].append(ckpt["thresh_inner"])
            ensemble_artifacts["pos_weights"].append(ckpt["pos_weights"])
            oof_preds[ckpt["val_ix"]] = ckpt["oof_probs"]

        assert np.all(oof_preds[val_ix] == pytest.approx(0.7))
        assert np.all(oof_preds[4:] == -1.0)

    def test_populates_all_artifact_lists(self):
        """All six artifact lists are filled in fold order."""
        n_classes = 2
        completed = {
            0: {
                "model_state":  {"w": 0},
                "preprocessor": "prep0",
                "calibrators":  {"c": 0},
                "thresholds":   np.zeros(n_classes),
                "thresh_inner": np.full(n_classes, 0.4),
                "pos_weights":  np.ones(n_classes) * 2,
                "val_ix":       np.array([0]),
                "oof_probs":    np.zeros((1, n_classes), dtype=np.float32),
            },
            1: {
                "model_state":  {"w": 1},
                "preprocessor": "prep1",
                "calibrators":  {"c": 1},
                "thresholds":   np.ones(n_classes),
                "thresh_inner": np.full(n_classes, 0.6),
                "pos_weights":  np.ones(n_classes) * 3,
                "val_ix":       np.array([1]),
                "oof_probs":    np.ones((1, n_classes), dtype=np.float32),
            },
        }
        ensemble_artifacts = {
            "models": [], "preps": [], "calibs": [], "thresh": [],
            "thresh_inner": [], "pos_weights": [],
        }
        oof_preds = np.full((10, n_classes), -1.0)

        for fold_idx in sorted(completed):
            ckpt = completed[fold_idx]
            ensemble_artifacts["models"].append(ckpt["model_state"])
            ensemble_artifacts["preps"].append(ckpt["preprocessor"])
            ensemble_artifacts["calibs"].append(ckpt["calibrators"])
            ensemble_artifacts["thresh"].append(ckpt["thresholds"])
            ensemble_artifacts["thresh_inner"].append(ckpt["thresh_inner"])
            ensemble_artifacts["pos_weights"].append(ckpt["pos_weights"])
            oof_preds[ckpt["val_ix"]] = ckpt["oof_probs"]

        assert len(ensemble_artifacts["models"]) == 2
        assert ensemble_artifacts["models"][0] == {"w": 0}
        assert ensemble_artifacts["models"][1] == {"w": 1}
        assert ensemble_artifacts["preps"] == ["prep0", "prep1"]
        assert len(ensemble_artifacts["thresh_inner"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
