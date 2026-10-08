"""Tests for shared/statistical/entropy.py."""
import numpy as np
import pytest
from shared.statistical.entropy import (
    binary_entropy,
    physician_entropy_change,
    model_entropy_change,
    entropy_pearson_r,
)


# ---------------------------------------------------------------------------
# binary_entropy
# ---------------------------------------------------------------------------

def test_entropy_zero_is_zero():
    assert binary_entropy(np.array([0.0])) == pytest.approx(0.0)


def test_entropy_one_is_zero():
    assert binary_entropy(np.array([1.0])) == pytest.approx(0.0)


def test_entropy_half_is_one_bit():
    assert binary_entropy(np.array([0.5])) == pytest.approx(1.0, abs=1e-10)


def test_entropy_shape_preserved():
    p = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]])
    h = binary_entropy(p)
    assert h.shape == p.shape


def test_entropy_symmetry():
    p = np.array([0.3, 0.7])
    np.testing.assert_allclose(binary_entropy(p[0:1]), binary_entropy(p[1:2]), atol=1e-10)


def test_entropy_clips_out_of_range():
    # Values slightly outside [0, 1] should be clipped without error
    p = np.array([-0.01, 1.01, 0.5])
    h = binary_entropy(p)
    assert not np.any(np.isnan(h))
    assert h[0] == pytest.approx(0.0)
    assert h[1] == pytest.approx(0.0)


def test_entropy_monotone_increasing_to_half():
    p = np.linspace(0.0, 0.5, 20)
    h = binary_entropy(p)
    assert np.all(np.diff(h) >= -1e-12)  # non-decreasing


# ---------------------------------------------------------------------------
# physician_entropy_change
# ---------------------------------------------------------------------------

def test_physician_entropy_change_shape():
    y_s = np.zeros((10, 3), dtype=np.float32)
    y_i = np.ones((10, 3), dtype=np.float32) * 0.5
    delta = physician_entropy_change(y_s, y_i)
    assert delta.shape == (10, 3)


def test_physician_entropy_change_values():
    # Survey: full agreement (0 or 1) → H=0; Interview: 0.5 → H=1
    y_s = np.zeros((5, 1), dtype=np.float32)    # H(y_s) = 0
    y_i = np.ones((5, 1), dtype=np.float32) * 0.5  # H(y_i) = 1
    delta = physician_entropy_change(y_s, y_i)
    np.testing.assert_allclose(delta, 1.0, atol=1e-6)


def test_physician_entropy_change_negative_when_resolves():
    # Survey: 0.5 (disagreement); Interview: 0.0 (agreement) → negative change
    y_s = np.ones((5, 1), dtype=np.float32) * 0.5
    y_i = np.zeros((5, 1), dtype=np.float32)
    delta = physician_entropy_change(y_s, y_i)
    assert np.all(delta < 0.0)


def test_physician_entropy_change_zero_when_unchanged():
    y_s = np.ones((5, 2), dtype=np.float32) * 0.5
    y_i = np.ones((5, 2), dtype=np.float32) * 0.5
    delta = physician_entropy_change(y_s, y_i)
    np.testing.assert_allclose(delta, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# model_entropy_change
# ---------------------------------------------------------------------------

def test_model_entropy_change_shape():
    y_cf = np.random.default_rng(0).uniform(0, 1, (8, 3))
    y_ca = np.random.default_rng(1).uniform(0, 1, (8, 3))
    delta = model_entropy_change(y_cf, y_ca)
    assert delta.shape == (8, 3)


def test_model_entropy_change_zero_when_same():
    y = np.random.default_rng(2).uniform(0, 1, (8, 3))
    delta = model_entropy_change(y, y)
    np.testing.assert_allclose(delta, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# entropy_pearson_r
# ---------------------------------------------------------------------------

def test_entropy_pearson_r_returns_dict():
    n, c = 20, 3
    classes = ["A", "B", "C"]
    rng = np.random.default_rng(42)
    phys_d = rng.uniform(-1, 1, (n, c))
    model_d = rng.uniform(-1, 1, (n, c))
    result = entropy_pearson_r(phys_d, model_d, ["A", "B"], classes)
    assert set(result.keys()) >= {"pearson_r", "p_value_descriptive_only", "descriptive_only", "n_observations"}


def test_entropy_pearson_r_perfect_correlation():
    n, c = 20, 2
    classes = ["A", "B"]
    rng = np.random.default_rng(7)
    data = rng.uniform(-1, 1, (n, c))
    result = entropy_pearson_r(data, data, classes, classes)
    assert abs(result["pearson_r"] - 1.0) < 1e-6


def test_entropy_pearson_r_no_eligible_classes():
    n, c = 20, 3
    classes = ["A", "B", "C"]
    rng = np.random.default_rng(8)
    data = rng.uniform(-1, 1, (n, c))
    result = entropy_pearson_r(data, data, [], classes)
    assert np.isnan(result["pearson_r"])
    assert result["n_observations"] == 0


def test_entropy_pearson_r_observation_count():
    n, c = 10, 2
    classes = ["A", "B"]
    rng = np.random.default_rng(9)
    data = rng.uniform(-1, 1, (n, c))
    # Eligible: both classes → 10*2 = 20 observations
    result = entropy_pearson_r(data, data, classes, classes)
    assert result["n_observations"] == n * c
