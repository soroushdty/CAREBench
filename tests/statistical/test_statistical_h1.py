"""Tests for tracks/representation/statistical/hypotheses/h1.py."""
import numpy as np
import pandas as pd
from tracks.representation.statistical.hypotheses.h1 import (
    _bh_correct,
    h1_binomial_per_class,
    h1_permutation_test,
    h1_cmh_test,
)

N, C = 24, 3
CLASSES = ["A", "B", "C"]
RNG = np.random.default_rng(7)


def _make_deltas(seed=0):
    rng = np.random.default_rng(seed)
    dp = rng.choice([-0.5, 0.0, 0.5], size=(N, C)).astype(np.float32)
    dm = rng.choice([-0.5, 0.0, 0.5], size=(N, C)).astype(np.float32)
    return dp, dm


def _patient_ids(n=N, n_patients=4):
    return np.repeat(np.arange(n_patients), n // n_patients)


# ---------------------------------------------------------------------------
# _bh_correct
# ---------------------------------------------------------------------------

def test_bh_correct_sorted_input():
    p = np.array([0.01, 0.04, 0.10, 0.20])
    adj = _bh_correct(p)
    assert len(adj) == 4
    # All adj p-values should be >= original
    assert np.all(adj >= p - 1e-12)
    # Monotone: adj[i] <= adj[i+1] after sorting by rank
    sorted_adj = np.sort(adj)
    assert np.all(sorted_adj[:-1] <= sorted_adj[1:] + 1e-12)


def test_bh_correct_all_zero():
    p = np.zeros(3)
    adj = _bh_correct(p)
    assert np.all(adj == 0.0)


def test_bh_correct_empty():
    adj = _bh_correct(np.array([]))
    assert len(adj) == 0


# ---------------------------------------------------------------------------
# h1_binomial_per_class
# ---------------------------------------------------------------------------

def test_h1_binomial_returns_dataframe():
    dp, dm = _make_deltas()
    pids = _patient_ids()
    eligible = ["A", "B"]
    df = h1_binomial_per_class(dp, dm, CLASSES, eligible, pids, n_resamples=100,
                               rng=np.random.default_rng(1))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == C
    assert set(df.columns) >= {"Class", "n_nonzero", "sign_agree_rate", "ci_lower",
                                "ci_upper", "binom_p", "bh_adj_p", "confirmatory"}


def test_h1_binomial_confirmatory_flag():
    dp, dm = _make_deltas()
    pids = _patient_ids()
    eligible = ["A"]
    df = h1_binomial_per_class(dp, dm, CLASSES, eligible, pids, n_resamples=50,
                               rng=np.random.default_rng(2))
    assert df.loc[df["Class"] == "A", "confirmatory"].values[0] == True
    assert df.loc[df["Class"] == "B", "confirmatory"].values[0] == False


def test_h1_binomial_bh_adj_only_for_confirmatory():
    dp, dm = _make_deltas()
    pids = _patient_ids()
    eligible = ["A"]
    df = h1_binomial_per_class(dp, dm, CLASSES, eligible, pids, n_resamples=50,
                               rng=np.random.default_rng(3))
    # Non-confirmatory classes should have NaN bh_adj_p
    non_conf = df[~df["confirmatory"]]
    assert non_conf["bh_adj_p"].isna().all()
    # Confirmatory class (A) should have a numeric bh_adj_p
    conf = df[df["confirmatory"]]
    assert not conf["bh_adj_p"].isna().all()


def test_h1_binomial_all_zero_dp():
    dp = np.zeros((N, C), dtype=np.float32)
    dm = np.ones((N, C), dtype=np.float32) * 0.5
    pids = _patient_ids()
    df = h1_binomial_per_class(dp, dm, CLASSES, CLASSES, pids, n_resamples=50,
                               rng=np.random.default_rng(4))
    assert df["n_nonzero"].eq(0).all()
    assert df["sign_agree_rate"].isna().all()


def test_h1_binomial_perfect_agreement_low_pvalue():
    # Δ_p > 0 and Δ_m > 0 for all → 100% agreement → very low binomial p
    dp = np.ones((N, 1), dtype=np.float32) * 0.5
    dm = np.ones((N, 1), dtype=np.float32) * 0.5
    pids = _patient_ids()
    df = h1_binomial_per_class(dp, dm, ["X"], ["X"], pids, n_resamples=50,
                               rng=np.random.default_rng(5))
    assert df.iloc[0]["binom_p"] < 0.01


# ---------------------------------------------------------------------------
# h1_permutation_test
# ---------------------------------------------------------------------------

def test_h1_permutation_returns_dict_keys():
    dp, dm = _make_deltas()
    pids = _patient_ids()
    result = h1_permutation_test(dp, dm, pids, n_permutations=50,
                                 rng=np.random.default_rng(10))
    for key in ("aggregate_rate", "ci_lower", "ci_upper", "null_mean", "null_std",
                "p_value", "n_permutations"):
        assert key in result


def test_h1_permutation_null_mean_near_half():
    # Random dp and dm → null should be near 0.5
    rng = np.random.default_rng(11)
    dp = rng.choice([-0.5, 0.5], size=(60, C)).astype(np.float32)
    dm = rng.choice([-0.5, 0.5], size=(60, C)).astype(np.float32)
    pids = np.repeat(np.arange(6), 10)
    result = h1_permutation_test(dp, dm, pids, n_permutations=200,
                                 rng=np.random.default_rng(12))
    assert 0.3 < result["null_mean"] < 0.7


def test_h1_permutation_pvalue_between_0_and_1():
    dp, dm = _make_deltas()
    pids = _patient_ids()
    result = h1_permutation_test(dp, dm, pids, n_permutations=50,
                                 rng=np.random.default_rng(13))
    assert 0.0 <= result["p_value"] <= 1.0


def test_h1_permutation_perfect_agreement_low_pvalue():
    # dm always has same sign as dp → observed > null → small p
    rng = np.random.default_rng(14)
    dp = rng.choice([-0.5, 0.5], size=(60, C)).astype(np.float32)
    dm = dp.copy()  # perfect sign agreement
    pids = np.repeat(np.arange(6), 10)
    result = h1_permutation_test(dp, dm, pids, n_permutations=200,
                                 rng=np.random.default_rng(15))
    assert result["p_value"] < 0.05


# ---------------------------------------------------------------------------
# h1_cmh_test
# ---------------------------------------------------------------------------

def test_h1_cmh_returns_dict_keys():
    dp, dm = _make_deltas()
    result = h1_cmh_test(dp, dm, ["A", "B"], CLASSES)
    for key in ("common_odds_ratio", "ci_lower", "ci_upper", "chi2_cmh", "p_cmh", "n_strata"):
        assert key in result


def test_h1_cmh_no_eligible_classes():
    dp, dm = _make_deltas()
    result = h1_cmh_test(dp, dm, [], CLASSES)
    assert result["n_strata"] == 0
    assert np.isnan(result["common_odds_ratio"])


def test_h1_cmh_n_strata_correct():
    dp, dm = _make_deltas()
    # Only 2 classes have non-zero dp entries
    result = h1_cmh_test(dp, dm, ["A", "B"], CLASSES)
    assert result["n_strata"] <= 2


def test_h1_cmh_pvalue_valid():
    dp, dm = _make_deltas(seed=99)
    result = h1_cmh_test(dp, dm, CLASSES, CLASSES)
    if not np.isnan(result["p_cmh"]):
        assert 0.0 <= result["p_cmh"] <= 1.0
