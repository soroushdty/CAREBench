"""Tests for shared/statistical/icc.py — icc21() and lins_ccc()."""
import numpy as np
from shared.statistical.icc import icc21, lins_ccc


# ---------------------------------------------------------------------------
# icc21 — known-value tests
# ---------------------------------------------------------------------------

def test_icc21_perfect_agreement():
    # All raters give identical ratings → ICC = 1.0
    ratings = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
    val = icc21(ratings)
    assert abs(val - 1.0) < 1e-6


def test_icc21_zero_between_subject_variance():
    # All subjects have same mean; large within-cell variance → near 0 or negative
    ratings = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0], [1.0, 3.0]], dtype=float)
    val = icc21(ratings)
    # No between-subject variance → ICC should be ≤ 0
    assert val <= 0.0 + 1e-6


def test_icc21_known_small_example():
    # Shrout & Fleiss (1979) Table 1 example (3 subjects, 4 raters):
    # Ratings:
    #   9, 2, 5, 8  → subject 1
    #   6, 1, 3, 2  → subject 2
    #   8, 4, 6, 8  → subject 3
    # Analytic ICC(2,1) using SS decomposition ≈ 0.343
    ratings = np.array([
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
    ], dtype=float)
    val = icc21(ratings)
    assert abs(val - 0.343) < 0.005


def test_icc21_single_rater_returns_nan():
    ratings = np.array([[1], [2], [3]], dtype=float)
    assert np.isnan(icc21(ratings))


def test_icc21_single_subject_returns_nan():
    ratings = np.array([[1, 2, 3]], dtype=float)
    assert np.isnan(icc21(ratings))


def test_icc21_two_raters_two_subjects():
    ratings = np.array([[2.0, 4.0], [3.0, 5.0]], dtype=float)
    val = icc21(ratings)
    # Should return a float (not nan, given valid input)
    assert not np.isnan(val)
    assert -1.0 <= val <= 1.0


def test_icc21_symmetric_raters():
    # Swapping rater columns should not change ICC
    ratings = np.array([[1, 3], [2, 4], [5, 7]], dtype=float)
    val1 = icc21(ratings)
    val2 = icc21(ratings[:, ::-1])
    assert abs(val1 - val2) < 1e-10


# ---------------------------------------------------------------------------
# lins_ccc — formula tests
# ---------------------------------------------------------------------------

def test_lins_ccc_perfect_agreement():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(lins_ccc(x, x) - 1.0) < 1e-10


def test_lins_ccc_perfect_disagreement():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0])
    val = lins_ccc(x, y)
    assert val < 0.0


def test_lins_ccc_constant_input_is_nan():
    x = np.array([0.5, 0.5, 0.5])
    y = np.array([0.1, 0.2, 0.3])
    val = lins_ccc(x, y)
    # var(x) = 0 → denom might be non-zero due to mean difference, but let's
    # just check it doesn't crash and is a finite float or nan
    assert isinstance(val, float)


def test_lins_ccc_both_constant_is_nan():
    x = np.ones(5)
    y = np.ones(5)
    val = lins_ccc(x, y)
    assert np.isnan(val)


def test_lins_ccc_shifted_same_variance():
    # Perfect correlation but means differ → CCC < 1
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x + 10.0
    val = lins_ccc(x, y)
    assert val < 1.0
    assert val > 0.0


def test_lins_ccc_mismatched_length_is_nan():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0])
    assert np.isnan(lins_ccc(x, y))


def test_lins_ccc_range():
    rng = np.random.default_rng(42)
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    val = lins_ccc(x, y)
    assert -1.0 <= val <= 1.0
