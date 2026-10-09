"""
Tests for generic output-dimension handling in HypothesisAnalyzer.

Verifies that the analyzer works with D=2, D=3, and D=10 output dimensions,
and that _validate_inputs() rejects mismatched shapes.
"""

from __future__ import annotations

import pytest
import numpy as np

from shared.evaluation.hypothesis_analyzer import HypothesisAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockDataset:
    """Minimal dataset-like object for HypothesisAnalyzer."""

    def __init__(self, n_rows: int, n_dims: int, n_patients: int = 4) -> None:
        # Create patient IDs with multiple rows per patient
        self.patient_ids = np.array(
            [f"P{i % n_patients}" for i in range(n_rows)]
        )
        self.category_names = [f"dim_{j}" for j in range(n_dims)]


def _make_scores(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random scores in [0, 1] with shape (n, d)."""
    return rng.uniform(0.0, 1.0, size=(n, d))


def _make_delta_physician(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random delta_physician with some non-zero cells."""
    # Mix of zeros and non-zeros to exercise both paths
    delta = rng.uniform(-0.5, 0.5, size=(n, d))
    # Zero out ~30% of cells
    mask = rng.random(size=(n, d)) < 0.3
    delta[mask] = 0.0
    return delta


_CFG = {
    "bootstrap_seed": 42,
    "permutation_seed": 42,
    "n_bootstrap_resamples": 50,  # small for speed
    "n_permutations": 100,  # small for speed
}


# ---------------------------------------------------------------------------
# Parametrized tests for D=2, D=3, D=10
# ---------------------------------------------------------------------------


@pytest.fixture(params=[2, 3, 10], ids=["D=2", "D=3", "D=10"])
def setup(request):
    """Create analyzer and synthetic data for a given dimension count D."""
    d = request.param
    n = 20  # rows (multiple of 4 patients)
    rng = np.random.default_rng(123)

    dataset = _MockDataset(n, d)
    analyzer = HypothesisAnalyzer(dataset, _CFG)

    cf = _make_scores(n, d, rng)
    cc = _make_scores(n, d, rng)
    sc = _make_scores(n, d, rng)
    delta = _make_delta_physician(n, d, rng)

    return {
        "analyzer": analyzer,
        "cf": cf,
        "cc": cc,
        "sc": sc,
        "delta": delta,
        "n": n,
        "d": d,
    }


class TestComputeH1Generic:
    """H1 works with any output dimension count."""

    def test_h1_returns_valid_result(self, setup) -> None:
        result = setup["analyzer"].compute_h1(setup["cf"], setup["cc"])
        assert "mean_abs_delta" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "per_category" in result
        assert isinstance(result["mean_abs_delta"], float)
        assert not np.isnan(result["mean_abs_delta"])

    def test_h1_per_category_count_matches_d(self, setup) -> None:
        result = setup["analyzer"].compute_h1(setup["cf"], setup["cc"])
        assert len(result["per_category"]) == setup["d"]


class TestComputeH2Generic:
    """H2 works with any output dimension count."""

    def test_h2_returns_valid_result(self, setup) -> None:
        result = setup["analyzer"].compute_h2(
            setup["cf"], setup["cc"], setup["delta"]
        )
        assert "mean_alignment" in result
        assert "sign_agreement_rate" in result
        assert isinstance(result["sign_agreement_rate"], float)


class TestComputeH3Generic:
    """H3 works with any output dimension count."""

    def test_h3_returns_valid_result(self, setup) -> None:
        result = setup["analyzer"].compute_h3(
            setup["cf"], setup["cc"], setup["delta"]
        )
        assert "pearson_r" in result
        assert "p_value" in result
        assert "mean_delta_physician" in result
        assert "mean_delta_llm" in result
        assert len(result["mean_delta_physician"]) == setup["d"]
        assert len(result["mean_delta_llm"]) == setup["d"]


class TestComputeH4Generic:
    """H4 works with any output dimension count."""

    def test_h4_returns_valid_result(self, setup) -> None:
        result = setup["analyzer"].compute_h4(
            setup["cf"], setup["cc"], setup["sc"], setup["delta"]
        )
        assert "mean_diff" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "p_value" in result
        assert isinstance(result["mean_diff"], float)


# ---------------------------------------------------------------------------
# _validate_inputs rejection tests
# ---------------------------------------------------------------------------


class TestValidateInputs:
    """_validate_inputs raises ValueError on shape mismatches."""

    def test_mismatched_row_count(self) -> None:
        dataset = _MockDataset(10, 3)
        analyzer = HypothesisAnalyzer(dataset, _CFG)
        a = np.zeros((10, 3))
        b = np.zeros((8, 3))  # different N
        with pytest.raises(ValueError, match="Row count mismatch"):
            analyzer._validate_inputs(a, b)

    def test_mismatched_column_count(self) -> None:
        dataset = _MockDataset(10, 3)
        analyzer = HypothesisAnalyzer(dataset, _CFG)
        a = np.zeros((10, 3))
        b = np.zeros((10, 5))  # different D
        with pytest.raises(ValueError, match="Output dimension.*mismatch"):
            analyzer._validate_inputs(a, b)

    def test_no_2d_arrays(self) -> None:
        dataset = _MockDataset(10, 3)
        analyzer = HypothesisAnalyzer(dataset, _CFG)
        a = np.zeros((10,))  # 1-D only
        with pytest.raises(ValueError, match="No 2-D arrays"):
            analyzer._validate_inputs(a)

    def test_valid_inputs_returns_d(self) -> None:
        dataset = _MockDataset(10, 5)
        analyzer = HypothesisAnalyzer(dataset, _CFG)
        a = np.zeros((10, 5))
        b = np.zeros((10, 5))
        assert analyzer._validate_inputs(a, b) == 5

    def test_compute_h1_rejects_mismatched_d(self) -> None:
        dataset = _MockDataset(10, 3)
        analyzer = HypothesisAnalyzer(dataset, _CFG)
        cf = np.zeros((10, 3))
        cc = np.zeros((10, 5))  # wrong D
        with pytest.raises(ValueError):
            analyzer.compute_h1(cf, cc)

    def test_compute_h4_rejects_mismatched_n(self) -> None:
        dataset = _MockDataset(10, 3)
        analyzer = HypothesisAnalyzer(dataset, _CFG)
        cf = np.zeros((10, 3))
        cc = np.zeros((10, 3))
        sc = np.zeros((8, 3))  # wrong N
        delta = np.zeros((10, 3))
        with pytest.raises(ValueError):
            analyzer.compute_h4(cf, cc, sc, delta)
