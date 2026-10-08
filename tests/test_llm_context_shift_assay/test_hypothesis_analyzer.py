"""
Tests for tracks/reasoning/hypothesis_analyzer.py.

Covers:
- H1 mean absolute delta calculation (test_h1_mean_absolute_delta)
- H2 sign agreement rate calculation (test_h2_sign_agreement_rate)
- H3 Pearson correlation calculation (test_h3_correlation)
- H4 alignment difference calculation (test_h4_alignment_difference)
- Bootstrap CI containment (test_bootstrap_ci_containment)
- Permutation p-value range (test_permutation_p_value_range)
- H4 zero diff when scores equal (test_h4_zero_diff_when_scores_equal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import scipy.stats

from shared.evaluation.hypothesis_analyzer import (
    HypothesisAnalyzer,
    patient_cluster_bootstrap,
    permutation_test_h3,
    permutation_test_h4,
)


# ---------------------------------------------------------------------------
# Minimal PairedDataset stub
# ---------------------------------------------------------------------------


@dataclass
class _FakePairedDataset:
    """Minimal stub matching the PairedDataset interface."""

    patient_ids: np.ndarray
    category_names: list[str]


CATEGORY_NAMES = [
    "behavioral_health",
    "diagnoses",
    "disabilities",
    "infectious_diseases",
    "genetics",
    "medications",
    "sexual_reproductive_health",
    "social_determinants_of_health",
    "violence",
    "other",
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides: Any) -> dict[str, Any]:
    base = {
        "bootstrap_seed": 42,
        "permutation_seed": 42,
        "n_bootstrap_resamples": 200,   # small for speed in tests
        "n_permutations": 500,          # small for speed in tests
    }
    base.update(overrides)
    return base


def _make_analyzer(
    n_rows: int = 24,
    n_patients: int = 4,
    **cfg_overrides: Any,
) -> tuple[HypothesisAnalyzer, np.ndarray]:
    """Return (analyzer, patient_ids) for a synthetic dataset."""
    # Assign patient IDs round-robin so each patient has n_rows/n_patients rows
    patient_ids = np.array([f"P{i % n_patients}" for i in range(n_rows)])
    ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
    cfg = _make_cfg(**cfg_overrides)
    return HypothesisAnalyzer(ds, cfg), patient_ids


# ---------------------------------------------------------------------------
# H1 tests
# ---------------------------------------------------------------------------


class TestComputeH1:
    """H1 mean absolute delta with known inputs."""

    def test_h1_mean_absolute_delta(self) -> None:
        """H1 aggregate mean_abs_delta matches manual calculation."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        # context_free all zeros, correct_context all 0.2 → delta = 0.2 everywhere
        cf = np.zeros((n, 10))
        cc = np.full((n, 10), 0.2)

        result = analyzer.compute_h1(cf, cc)

        assert result["mean_abs_delta"] == pytest.approx(0.2, abs=1e-9)

    def test_h1_per_category_values(self) -> None:
        """Per-category mean_abs_delta matches column-wise manual calculation."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        # Column 0 has delta 0.5, all others 0.0
        cc = np.zeros((n, 10))
        cc[:, 0] = 0.5

        result = analyzer.compute_h1(cf, cc)

        assert result["per_category"]["behavioral_health"]["mean_abs_delta"] == pytest.approx(0.5, abs=1e-9)
        for cat in CATEGORY_NAMES[1:]:
            assert result["per_category"][cat]["mean_abs_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_h1_ci_keys_present(self) -> None:
        """H1 result contains ci_lower and ci_upper keys."""
        analyzer, _ = _make_analyzer()
        cf = np.random.default_rng(0).random((24, 10))
        cc = np.random.default_rng(1).random((24, 10))
        result = analyzer.compute_h1(cf, cc)
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_h1_per_category_ci_keys_present(self) -> None:
        """Each per-category entry has ci_lower and ci_upper."""
        analyzer, _ = _make_analyzer()
        cf = np.random.default_rng(0).random((24, 10))
        cc = np.random.default_rng(1).random((24, 10))
        result = analyzer.compute_h1(cf, cc)
        for cat in CATEGORY_NAMES:
            assert "ci_lower" in result["per_category"][cat]
            assert "ci_upper" in result["per_category"][cat]


# ---------------------------------------------------------------------------
# H2 tests
# ---------------------------------------------------------------------------


class TestComputeH2:
    """H2 sign agreement rate with known inputs."""

    def test_h2_sign_agreement_rate(self) -> None:
        """Sign agreement rate is 1.0 when LLM and physician deltas always agree."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        cc = np.full((n, 10), 0.3)   # delta_llm_correct = +0.3
        dp = np.full((n, 10), 0.5)   # delta_physician = +0.5 (same sign)

        result = analyzer.compute_h2(cf, cc, dp)

        assert result["sign_agreement_rate"] == pytest.approx(1.0, abs=1e-9)

    def test_h2_sign_agreement_rate_zero(self) -> None:
        """Sign agreement rate is 0.0 when LLM and physician deltas always disagree."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        cc = np.full((n, 10), -0.3)  # delta_llm_correct = -0.3
        dp = np.full((n, 10), 0.5)   # delta_physician = +0.5 (opposite sign)

        result = analyzer.compute_h2(cf, cc, dp)

        assert result["sign_agreement_rate"] == pytest.approx(0.0, abs=1e-9)

    def test_h2_mean_alignment(self) -> None:
        """Mean alignment = sign(dp) * delta_llm_correct averaged over non-zero cells."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        cc = np.full((n, 10), 0.4)   # delta_llm_correct = +0.4
        dp = np.full((n, 10), 1.0)   # sign = +1

        result = analyzer.compute_h2(cf, cc, dp)

        # alignment = sign(1.0) * 0.4 = 0.4
        assert result["mean_alignment"] == pytest.approx(0.4, abs=1e-9)

    def test_h2_all_zero_delta_physician_returns_nan(self) -> None:
        """H2 returns NaN for all metrics when all delta_physician values are zero."""
        analyzer, _ = _make_analyzer()
        cf = np.random.default_rng(0).random((24, 10))
        cc = np.random.default_rng(1).random((24, 10))
        dp = np.zeros((24, 10))

        result = analyzer.compute_h2(cf, cc, dp)

        assert np.isnan(result["mean_alignment"])
        assert np.isnan(result["sign_agreement_rate"])

    def test_h2_ci_keys_present(self) -> None:
        """H2 result contains all six expected keys."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        cc = np.full((n, 10), 0.3)
        dp = np.full((n, 10), 0.5)

        result = analyzer.compute_h2(cf, cc, dp)

        for key in [
            "mean_alignment",
            "ci_lower_alignment",
            "ci_upper_alignment",
            "sign_agreement_rate",
            "ci_lower_sign_agree",
            "ci_upper_sign_agree",
        ]:
            assert key in result


# ---------------------------------------------------------------------------
# H3 tests
# ---------------------------------------------------------------------------


class TestComputeH3:
    """H3 Pearson correlation with known inputs."""

    def test_h3_correlation_perfect_positive(self) -> None:
        """H3 Pearson r ≈ 1.0 when class-level deltas are perfectly correlated."""
        n = 20
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        # Construct scores so that mean_delta_llm == mean_delta_physician
        rng = np.random.default_rng(7)
        dp = rng.uniform(-0.5, 0.5, (n, 10))
        cf = np.zeros((n, 10))
        cc = cf + dp  # delta_llm_correct == delta_physician

        result = analyzer.compute_h3(cf, cc, dp)

        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-6)

    def test_h3_correlation_known_value(self) -> None:
        """H3 Pearson r matches scipy.stats.pearsonr for known class-level deltas."""
        n = 20
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        rng = np.random.default_rng(99)
        cf = rng.random((n, 10))
        cc = rng.random((n, 10))
        dp = rng.uniform(-0.5, 0.5, (n, 10))

        result = analyzer.compute_h3(cf, cc, dp)

        # Manually compute expected r
        delta_llm = cc - cf
        mean_dp = dp.mean(axis=0)
        mean_llm = delta_llm.mean(axis=0)
        expected_r, _ = scipy.stats.pearsonr(mean_dp, mean_llm)

        assert result["pearson_r"] == pytest.approx(expected_r, abs=1e-9)

    def test_h3_result_keys(self) -> None:
        """H3 result contains pearson_r, p_value, mean_delta_physician, mean_delta_llm."""
        analyzer, _ = _make_analyzer()
        cf = np.random.default_rng(0).random((24, 10))
        cc = np.random.default_rng(1).random((24, 10))
        dp = np.random.default_rng(2).uniform(-0.5, 0.5, (24, 10))

        result = analyzer.compute_h3(cf, cc, dp)

        assert "pearson_r" in result
        assert "p_value" in result
        assert "mean_delta_physician" in result
        assert "mean_delta_llm" in result
        assert result["mean_delta_physician"].shape == (10,)
        assert result["mean_delta_llm"].shape == (10,)

    def test_h3_p_value_range(self) -> None:
        """H3 permutation p-value is in [0, 1]."""
        analyzer, _ = _make_analyzer()
        cf = np.random.default_rng(0).random((24, 10))
        cc = np.random.default_rng(1).random((24, 10))
        dp = np.random.default_rng(2).uniform(-0.5, 0.5, (24, 10))

        result = analyzer.compute_h3(cf, cc, dp)

        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# H4 tests
# ---------------------------------------------------------------------------


class TestComputeH4:
    """H4 alignment difference with known inputs."""

    def test_h4_alignment_difference(self) -> None:
        """H4 mean_diff matches manual calculation."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        cc = np.full((n, 10), 0.6)   # delta_llm_correct = +0.6
        sc = np.full((n, 10), 0.2)   # delta_llm_shuffled = +0.2
        dp = np.full((n, 10), 1.0)   # sign = +1

        result = analyzer.compute_h4(cf, cc, sc, dp)

        # align_correct = 1 * 0.6 = 0.6
        # align_shuffled = 1 * 0.2 = 0.2
        # mean_diff = 0.6 - 0.2 = 0.4
        assert result["mean_diff"] == pytest.approx(0.4, abs=1e-9)

    def test_h4_zero_diff_when_scores_equal(self) -> None:
        """H4 mean_diff = 0 when correct_context_scores == shuffled_context_scores."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        rng = np.random.default_rng(5)
        cf = rng.random((n, 10))
        cc = rng.random((n, 10))
        sc = cc.copy()  # identical to correct context
        dp = rng.uniform(-0.5, 0.5, (n, 10))
        # Ensure some non-zero cells
        dp[dp == 0.0] = 0.1

        result = analyzer.compute_h4(cf, cc, sc, dp)

        assert result["mean_diff"] == pytest.approx(0.0, abs=1e-9)

    def test_h4_all_zero_delta_physician_returns_nan(self) -> None:
        """H4 returns NaN for all metrics when all delta_physician values are zero."""
        analyzer, _ = _make_analyzer()
        rng = np.random.default_rng(0)
        cf = rng.random((24, 10))
        cc = rng.random((24, 10))
        sc = rng.random((24, 10))
        dp = np.zeros((24, 10))

        result = analyzer.compute_h4(cf, cc, sc, dp)

        assert np.isnan(result["mean_diff"])
        assert np.isnan(result["p_value"])

    def test_h4_result_keys(self) -> None:
        """H4 result contains mean_diff, ci_lower, ci_upper, p_value."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        cf = np.zeros((n, 10))
        cc = np.full((n, 10), 0.5)
        sc = np.full((n, 10), 0.3)
        dp = np.full((n, 10), 1.0)

        result = analyzer.compute_h4(cf, cc, sc, dp)

        for key in ["mean_diff", "ci_lower", "ci_upper", "p_value"]:
            assert key in result

    def test_h4_p_value_range(self) -> None:
        """H4 permutation p-value is in [0, 1]."""
        n = 8
        patient_ids = np.array(["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3"])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        rng = np.random.default_rng(3)
        cf = rng.random((n, 10))
        cc = rng.random((n, 10))
        sc = rng.random((n, 10))
        dp = rng.uniform(-0.5, 0.5, (n, 10))
        dp[dp == 0.0] = 0.1

        result = analyzer.compute_h4(cf, cc, sc, dp)

        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap CI containment tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Bootstrap CI lower ≤ point estimate ≤ upper."""

    def test_bootstrap_ci_containment(self) -> None:
        """Bootstrap CI lower bound ≤ point estimate ≤ upper bound."""
        rng = np.random.default_rng(42)
        n = 40
        patient_ids = np.array([f"P{i % 5}" for i in range(n)])
        values = rng.random(n)

        point = float(np.mean(values))

        def _stat(idx: np.ndarray) -> float:
            return float(np.mean(values[idx]))

        ci_lo, ci_hi = patient_cluster_bootstrap(
            statistic_fn=_stat,
            data=values,
            patient_ids=patient_ids,
            n_resamples=500,
            rng=np.random.default_rng(0),
        )

        assert ci_lo <= point <= ci_hi

    def test_h1_bootstrap_ci_containment(self) -> None:
        """H1 bootstrap CI lower ≤ mean_abs_delta ≤ upper."""
        n = 24
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        rng = np.random.default_rng(7)
        cf = rng.random((n, 10))
        cc = rng.random((n, 10))

        result = analyzer.compute_h1(cf, cc)

        assert result["ci_lower"] <= result["mean_abs_delta"] <= result["ci_upper"]

    def test_h2_bootstrap_ci_containment(self) -> None:
        """H2 bootstrap CI lower ≤ mean_alignment ≤ upper."""
        n = 24
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        rng = np.random.default_rng(8)
        cf = rng.random((n, 10))
        cc = rng.random((n, 10))
        dp = rng.uniform(-0.5, 0.5, (n, 10))
        dp[dp == 0.0] = 0.1

        result = analyzer.compute_h2(cf, cc, dp)

        assert result["ci_lower_alignment"] <= result["mean_alignment"] <= result["ci_upper_alignment"]

    def test_h4_bootstrap_ci_containment(self) -> None:
        """H4 bootstrap CI lower ≤ mean_diff ≤ upper."""
        n = 24
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        ds = _FakePairedDataset(patient_ids=patient_ids, category_names=CATEGORY_NAMES)
        cfg = _make_cfg()
        analyzer = HypothesisAnalyzer(ds, cfg)

        rng = np.random.default_rng(9)
        cf = rng.random((n, 10))
        cc = rng.random((n, 10))
        sc = rng.random((n, 10))
        dp = rng.uniform(-0.5, 0.5, (n, 10))
        dp[dp == 0.0] = 0.1

        result = analyzer.compute_h4(cf, cc, sc, dp)

        assert result["ci_lower"] <= result["mean_diff"] <= result["ci_upper"]


# ---------------------------------------------------------------------------
# Permutation p-value range tests
# ---------------------------------------------------------------------------


class TestPermutationPValue:
    """Permutation p-values are in [0, 1]."""

    def test_permutation_h3_p_value_range(self) -> None:
        """permutation_test_h3 returns p-value in [0, 1]."""
        rng = np.random.default_rng(10)
        mean_dp = rng.uniform(-0.5, 0.5, 10)
        mean_llm = rng.uniform(-0.5, 0.5, 10)

        p = permutation_test_h3(mean_dp, mean_llm, n_permutations=200, rng=np.random.default_rng(0))

        assert 0.0 <= p <= 1.0

    def test_permutation_h4_p_value_range(self) -> None:
        """permutation_test_h4 returns p-value in [0, 1]."""
        rng = np.random.default_rng(11)
        n = 20
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        align_c = rng.random(n)
        align_s = rng.random(n)

        p = permutation_test_h4(align_c, align_s, patient_ids, n_permutations=200, rng=np.random.default_rng(0))

        assert 0.0 <= p <= 1.0

    def test_permutation_h3_deterministic(self) -> None:
        """permutation_test_h3 is deterministic given the same seed."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        mean_dp = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.15, -0.3, 0.2, -0.05, 0.0])
        mean_llm = np.array([0.2, -0.1, 0.25, -0.15, 0.1, 0.1, -0.2, 0.3, -0.1, 0.05])

        p1 = permutation_test_h3(mean_dp, mean_llm, n_permutations=100, rng=rng_a)
        p2 = permutation_test_h3(mean_dp, mean_llm, n_permutations=100, rng=rng_b)

        assert p1 == pytest.approx(p2, abs=1e-12)

    def test_permutation_h4_deterministic(self) -> None:
        """permutation_test_h4 is deterministic given the same seed."""
        n = 16
        patient_ids = np.array([f"P{i % 4}" for i in range(n)])
        align_c = np.linspace(0.1, 0.9, n)
        align_s = np.linspace(0.05, 0.85, n)

        p1 = permutation_test_h4(align_c, align_s, patient_ids, n_permutations=100, rng=np.random.default_rng(7))
        p2 = permutation_test_h4(align_c, align_s, patient_ids, n_permutations=100, rng=np.random.default_rng(7))

        assert p1 == pytest.approx(p2, abs=1e-12)
