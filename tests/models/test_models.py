"""Tests for models/ConstantCalibrator, MultiLabelModel, and Preprocessor."""
import pickle

import numpy as np
import pytest
import torch

from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
from tracks.representation.models.MultiLabelModel import MultiLabelModel
from tracks.representation.models.Preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# ConstantCalibrator
# ---------------------------------------------------------------------------

class TestConstantCalibrator:
    def test_predict_shape(self):
        cal = ConstantCalibrator(0.3)
        X = np.zeros((5, 1))
        out = cal.predict(X)
        assert out.shape == (5,)

    def test_predict_all_equal_prob(self):
        cal = ConstantCalibrator(0.7)
        X = np.zeros((10, 1))
        out = cal.predict(X)
        np.testing.assert_array_equal(out, np.full(10, 0.7))

    def test_predict_proba_has_two_columns(self):
        # predict_proba always returns a 2-column array (col0=neg, col1=pos).
        # The current implementation returns (1, 2) regardless of input length
        # and relies on numpy broadcasting in apply_calibrators.
        cal = ConstantCalibrator(0.4)
        X = np.zeros((8, 1))
        out = cal.predict_proba(X)
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_predict_proba_sums_to_one(self):
        cal = ConstantCalibrator(0.4)
        X = np.zeros((8, 1))
        out = cal.predict_proba(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(out.shape[0]))

    def test_predict_proba_col1_is_prob(self):
        prob = 0.6
        cal = ConstantCalibrator(prob)
        X = np.zeros((3, 1))
        out = cal.predict_proba(X)
        np.testing.assert_allclose(out[:, 1], prob)

    def test_predict_proba_col0_is_complement(self):
        prob = 0.6
        cal = ConstantCalibrator(prob)
        X = np.zeros((3, 1))
        out = cal.predict_proba(X)
        np.testing.assert_allclose(out[:, 0], 1 - prob)

    def test_prob_zero(self):
        cal = ConstantCalibrator(0.0)
        X = np.zeros((4, 1))
        assert (cal.predict(X) == 0.0).all()
        assert (cal.predict_proba(X)[:, 1] == 0.0).all()

    def test_prob_one(self):
        cal = ConstantCalibrator(1.0)
        X = np.zeros((4, 1))
        assert (cal.predict(X) == 1.0).all()
        assert (cal.predict_proba(X)[:, 0] == 0.0).all()


# ---------------------------------------------------------------------------
# MultiLabelModel
# ---------------------------------------------------------------------------

class TestMultiLabelModel:
    def test_logistic_regression_output_shape(self):
        model = MultiLabelModel(n_features=10, n_classes=3)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 3)

    def test_logistic_regression_state_dict_keys(self):
        model = MultiLabelModel(n_features=10, n_classes=3)
        keys = list(model.state_dict().keys())
        assert any("linear" in k for k in keys)
        assert not any("net" in k for k in keys)

    def test_mlp_output_shape(self):
        model = MultiLabelModel(n_features=10, n_classes=3, hidden_dims=[64])
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 3)

    def test_mlp_state_dict_keys(self):
        model = MultiLabelModel(n_features=10, n_classes=3, hidden_dims=[64])
        keys = list(model.state_dict().keys())
        assert any("net" in k for k in keys)
        assert not any("linear" in k for k in keys)

    def test_two_hidden_layer_mlp(self):
        model = MultiLabelModel(n_features=8, n_classes=2, hidden_dims=[32, 16])
        x = torch.randn(4, 8)
        out = model(x)
        assert out.shape == (4, 2)

    def test_relu_activation(self):
        model = MultiLabelModel(n_features=8, n_classes=2, hidden_dims=[16], activation="relu")
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 2)

    def test_gradients_flow_logistic(self):
        model = MultiLabelModel(n_features=6, n_classes=2)
        x = torch.randn(4, 6)
        loss = model(x).sum()
        loss.backward()
        assert model.linear.weight.grad is not None

    def test_gradients_flow_mlp(self):
        model = MultiLabelModel(n_features=6, n_classes=2, hidden_dims=[16])
        x = torch.randn(4, 6)
        loss = model(x).sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_single_class(self):
        model = MultiLabelModel(n_features=4, n_classes=1)
        x = torch.randn(3, 4)
        out = model(x)
        assert out.shape == (3, 1)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def _make_data(n=50, p=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float32)


class TestPreprocessor:
    def test_fit_transform_output_shape(self):
        X = _make_data(50, 20)
        pre = Preprocessor(n_components=10)
        pre.fit(X)
        out = pre.transform(X)
        assert out.shape == (50, 10)

    def test_output_dtype_is_float32(self):
        X = _make_data(50, 20)
        pre = Preprocessor(n_components=10)
        pre.fit(X)
        out = pre.transform(X)
        assert out.dtype == np.float32

    def test_n_components_capped_at_min_n_p(self):
        # 5 samples, 10 features — components capped at 5
        X = _make_data(5, 10)
        pre = Preprocessor(n_components=100)
        pre.fit(X)
        out = pre.transform(X)
        assert out.shape[1] <= min(5, 10)

    def test_fit_raises_on_nan(self):
        X = _make_data(20, 10)
        X[0, 0] = float("nan")
        pre = Preprocessor(n_components=5)
        with pytest.raises(ValueError, match="NaN"):
            pre.fit(X)

    def test_fit_raises_on_zero_variance(self):
        X = np.ones((20, 10), dtype=np.float32)
        pre = Preprocessor(n_components=5)
        with pytest.raises(ValueError, match="zero variance"):
            pre.fit(X)

    def test_transform_before_fit_raises(self):
        X = _make_data(20, 10)
        pre = Preprocessor(n_components=5)
        with pytest.raises(RuntimeError, match="fit"):
            pre.transform(X)

    def test_pickle_roundtrip_produces_same_output(self):
        X = _make_data(30, 15)
        pre = Preprocessor(n_components=8, whiten=False)
        pre.fit(X)
        out_before = pre.transform(X)

        serialized = pickle.dumps(pre)
        pre2 = pickle.loads(serialized)
        out_after = pre2.transform(X)

        np.testing.assert_allclose(out_before, out_after, rtol=1e-4, atol=1e-6)

    def test_whitening_true_differs_from_false(self):
        X = _make_data(40, 15)
        pre_w = Preprocessor(n_components=8, whiten=True)
        pre_nw = Preprocessor(n_components=8, whiten=False)
        pre_w.fit(X)
        pre_nw.fit(X)
        out_w = pre_w.transform(X)
        out_nw = pre_nw.transform(X)
        # Whitened and non-whitened outputs should differ
        assert not np.allclose(out_w, out_nw, atol=1e-3)

    def test_transform_new_data(self):
        X_train = _make_data(50, 20)
        X_test = _make_data(10, 20, seed=99)
        pre = Preprocessor(n_components=8)
        pre.fit(X_train)
        out = pre.transform(X_test)
        assert out.shape == (10, 8)
        assert out.dtype == np.float32

    def test_whitening_produces_unit_variance_on_train_data(self):
        # After whitening with scale = S/sqrt(N) (biased std), the biased
        # per-component variance of the training projection should equal 1.0.
        X = _make_data(50, 20)
        pre = Preprocessor(n_components=8, whiten=True)
        pre.fit(X)
        out = pre.transform(X)
        component_vars = np.var(out, axis=0, ddof=0)
        np.testing.assert_allclose(component_vars, np.ones(8), rtol=1e-4, atol=1e-5)
