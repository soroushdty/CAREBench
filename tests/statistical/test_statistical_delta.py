"""Tests for shared/statistical/delta.py."""
import numpy as np
from shared.statistical.delta import (
    compute_physician_deltas,
    compute_model_deltas,
    sign_agreement_mask,
    confirmatory_eligible_classes,
    nonzero_counts_per_class,
)

N, C = 20, 3


def _survey():
    return np.array([[0.0, 0.5, 1.0, 0.0, 0.5] * 4], dtype=np.float32).T.reshape(N, 1).repeat(C, axis=1)


def test_physician_delta_values():
    y_s = np.zeros((5, 2), dtype=np.float32)
    y_s[2, 1] = 1.0  # row 2, col 1 → dp[2,1] = 0.0 - 1.0 = -1.0
    y_i = np.array([[0.0, 0.5], [0.5, 1.0], [1.0, 0.0], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
    dp = compute_physician_deltas(y_s, y_i)
    assert dp.shape == (5, 2)
    np.testing.assert_allclose(dp[:, 0], [0.0, 0.5, 1.0, 0.5, 0.0])
    np.testing.assert_allclose(dp[:, 1], [0.5, 1.0, -1.0, 0.5, 0.0])


def test_model_delta_values():
    cf = np.array([[0.3, 0.7], [0.5, 0.1]], dtype=np.float32)
    ca = np.array([[0.5, 0.4], [0.5, 0.9]], dtype=np.float32)
    dm = compute_model_deltas(cf, ca)
    np.testing.assert_allclose(dm[:, 0], [0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(dm[:, 1], [-0.3, 0.8], atol=1e-6)


def test_sign_agreement_mask_basic():
    # dp>0, dm>0 → agree; dp>0, dm<0 → disagree; dp=0 → False
    dp = np.array([[1.0, -0.5, 0.0]], dtype=np.float32)
    dm = np.array([[0.3, 0.2, 0.9]], dtype=np.float32)
    mask = sign_agreement_mask(dp, dm)
    assert mask[0, 0] == True   # both positive
    assert mask[0, 1] == False  # opposite signs
    assert mask[0, 2] == False  # dp==0


def test_sign_agreement_mask_zero_dm():
    # zero dm should NOT agree with non-zero dp
    dp = np.array([[0.5]], dtype=np.float32)
    dm = np.array([[0.0]], dtype=np.float32)
    mask = sign_agreement_mask(dp, dm)
    assert mask[0, 0] == False


def test_confirmatory_eligible_classes():
    # class0: 16 non-zero, class1: 5 non-zero, class2: 20 non-zero
    dp = np.zeros((20, 3), dtype=np.float32)
    dp[:16, 0] = 0.5
    dp[:5, 1] = -0.5
    dp[:, 2] = 1.0
    eligible, descriptive = confirmatory_eligible_classes(dp, ["A", "B", "C"], min_nonzero=15)
    assert "A" in eligible
    assert "C" in eligible
    assert "B" in descriptive


def test_nonzero_counts():
    dp = np.zeros((10, 2), dtype=np.float32)
    dp[:7, 0] = 0.5
    dp[2:5, 1] = -1.0
    counts = nonzero_counts_per_class(dp, ["X", "Y"])
    assert counts["X"] == 7
    assert counts["Y"] == 3
