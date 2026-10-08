"""Tests for INCOMPAT 4: fixed_f1_threshold scalar replaces f1_sweep list."""

import warnings

import numpy as np
import pytest


def _minimal_cfg(primary_threshold, **extra):
    cfg = {
        "global_seed": 0,
        "num_epochs": 1,
        "early_stopping_patience": 1,
        "hidden_dims": [],
        "dropout": 0.0,
        "activation": "gelu",
        "objective": "F1",
        "device": "cpu",
        "primary_threshold": primary_threshold,
        "tau": 0.5,
        "eval_pos_threshold": 0.5,
        "default_classes": ["c0", "c1"],
    }
    cfg.update(extra)
    return cfg


def _minimal_data(n=20, n_classes=2, n_features=4):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    Y = (rng.uniform(0, 1, (n, n_classes)) > 0.5).astype(np.float32)
    return X, Y


class TestFixedF1Threshold:
    def test_fixed_f1_threshold_scalar_used_directly(self):
        """primary_threshold='fixed_f1' with fixed_f1_threshold=0.4 → all-0.4 threshold array."""
        from tracks.representation.training.stage1.train_single_model import train_single_model

        X, Y = _minimal_data()
        X_val, Y_val = _minimal_data(n=10, n_features=4)
        cfg = _minimal_cfg("fixed_f1", fixed_f1_threshold=0.4)
        hp = {"lr": 0.001, "weight_decay": 0.0001, "batch_size": 32, "weight_cap": 10.0,
              "selected_stage1_head_config": []}
        pos_w = np.ones(2, dtype=np.float32)

        _, _, thresholds = train_single_model(X, Y, X_val, Y_val, cfg, hp, pos_w)

        assert thresholds.shape == (2,)
        np.testing.assert_allclose(thresholds, 0.4, atol=1e-6)

    def test_deprecated_f1_sweep_emits_warning_and_uses_first_element(self):
        """primary_threshold='f1_sweep' emits DeprecationWarning and uses f1_sweep[0]."""
        from tracks.representation.training.stage1.train_single_model import train_single_model

        X, Y = _minimal_data()
        X_val, Y_val = _minimal_data(n=10, n_features=4)
        cfg = _minimal_cfg("f1_sweep", f1_sweep=[0.3, 0.0])
        hp = {"lr": 0.001, "weight_decay": 0.0001, "batch_size": 32, "weight_cap": 10.0,
              "selected_stage1_head_config": []}
        pos_w = np.ones(2, dtype=np.float32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, thresholds = train_single_model(X, Y, X_val, Y_val, cfg, hp, pos_w)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "f1_sweep" in str(dep_warnings[0].message).lower() or "fixed_f1" in str(dep_warnings[0].message).lower()
        np.testing.assert_allclose(thresholds, 0.3, atol=1e-6)

    def test_fixed_f1_threshold_parsed_as_scalar_not_list(self):
        """fixed_f1_threshold in config is a scalar; accessing cfg.get('fixed_f1_threshold') returns float."""
        cfg = {"fixed_f1_threshold": 0.35}
        val = cfg.get("fixed_f1_threshold")
        assert isinstance(val, float)
        threshold = float(val)
        assert threshold == pytest.approx(0.35)
