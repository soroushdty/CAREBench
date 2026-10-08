"""Tests for tracks/representation/statistical/hypotheses/h2.py."""
import numpy as np
import pandas as pd
from tracks.representation.statistical.hypotheses.h2 import (
    h2_wilcoxon_per_class,
    h2_wasserstein_per_class,
    h2_macro_summary,
)

N, C = 24, 3
CLASSES = ["A", "B", "C"]


def _patient_ids(n=N, n_patients=4):
    return np.repeat(np.arange(n_patients), n // n_patients)


def _make_preds(seed=0):
    rng = np.random.default_rng(seed)
    y_int = rng.uniform(0, 1, size=(N, C)).astype(np.float64)
    y_cf = rng.uniform(0, 1, size=(N, C)).astype(np.float64)
    y_ca = rng.uniform(0, 1, size=(N, C)).astype(np.float64)
    return y_int, y_cf, y_ca


# ---------------------------------------------------------------------------
# h2_wilcoxon_per_class
# ---------------------------------------------------------------------------

def test_wilcoxon_returns_dataframe():
    y_int, y_cf, y_ca = _make_preds()
    pids = _patient_ids()
    df = h2_wilcoxon_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=100,
                               rng=np.random.default_rng(1))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == C
    assert set(df.columns) >= {"Class", "BS_cf", "BS_ca", "improvement",
                                "ci_lower", "ci_upper", "wilcoxon_stat",
                                "wilcoxon_p", "bh_adj_p"}


def test_wilcoxon_improvement_sign():
    # y_ca perfectly predicts y_int → BS_ca ~ 0, improvement > 0
    rng = np.random.default_rng(2)
    y_int = rng.uniform(0, 1, size=(N, C))
    y_cf = rng.uniform(0, 1, size=(N, C))
    y_ca = y_int.copy()  # perfect prediction
    pids = _patient_ids()
    df = h2_wilcoxon_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=50,
                               rng=np.random.default_rng(3))
    assert (df["improvement"] > 0).all()


def test_wilcoxon_equal_preds_returns_p1():
    # CF == CA → no improvement → Wilcoxon p should be 1.0 (or NaN)
    rng = np.random.default_rng(4)
    y_int = rng.uniform(0, 1, size=(N, C))
    y_same = rng.uniform(0, 1, size=(N, C))
    pids = _patient_ids()
    df = h2_wilcoxon_per_class(y_same, y_same, y_int, CLASSES, pids, n_resamples=50,
                               rng=np.random.default_rng(5))
    for _, row in df.iterrows():
        p = row["wilcoxon_p"]
        assert np.isnan(p) or p == 1.0


def test_wilcoxon_bh_adj_monotone():
    y_int, y_cf, y_ca = _make_preds(seed=6)
    pids = _patient_ids()
    df = h2_wilcoxon_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=50,
                               rng=np.random.default_rng(7))
    raw = df["cluster_p"].values  # BH is applied to the cluster p-value (#16)
    adj = df["bh_adj_p"].values
    # Each adjusted p >= raw p (BH cannot shrink p-values)
    for r, a in zip(raw, adj):
        if not np.isnan(r) and not np.isnan(a):
            assert a >= r - 1e-12


def test_wilcoxon_ci_ordered():
    y_int, y_cf, y_ca = _make_preds(seed=8)
    pids = _patient_ids()
    df = h2_wilcoxon_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=100,
                               rng=np.random.default_rng(9))
    assert (df["ci_lower"] <= df["ci_upper"] + 1e-10).all()


# ---------------------------------------------------------------------------
# h2_wasserstein_per_class
# ---------------------------------------------------------------------------

def test_wasserstein_returns_dataframe():
    y_int, y_cf, y_ca = _make_preds(seed=10)
    pids = _patient_ids()
    df = h2_wasserstein_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=100,
                                  rng=np.random.default_rng(11))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == C
    assert set(df.columns) >= {"Class", "W_cf", "W_ca",
                                "ci_lower_cf", "ci_upper_cf",
                                "ci_lower_ca", "ci_upper_ca"}


def test_wasserstein_nonnegative():
    y_int, y_cf, y_ca = _make_preds(seed=12)
    pids = _patient_ids()
    df = h2_wasserstein_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=50,
                                  rng=np.random.default_rng(13))
    assert (df["W_cf"] >= 0).all()
    assert (df["W_ca"] >= 0).all()


def test_wasserstein_perfect_pred_is_zero():
    rng = np.random.default_rng(14)
    y_int = rng.uniform(0, 1, size=(N, C))
    y_ca = y_int.copy()
    y_cf = rng.uniform(0, 1, size=(N, C))
    pids = _patient_ids()
    df = h2_wasserstein_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=50,
                                  rng=np.random.default_rng(15))
    np.testing.assert_allclose(df["W_ca"].values, 0.0, atol=1e-6)


def test_wasserstein_ci_ordered():
    y_int, y_cf, y_ca = _make_preds(seed=16)
    pids = _patient_ids()
    df = h2_wasserstein_per_class(y_cf, y_ca, y_int, CLASSES, pids, n_resamples=100,
                                  rng=np.random.default_rng(17))
    assert (df["ci_lower_cf"] <= df["ci_upper_cf"] + 1e-10).all()
    assert (df["ci_lower_ca"] <= df["ci_upper_ca"] + 1e-10).all()


# ---------------------------------------------------------------------------
# h2_macro_summary
# ---------------------------------------------------------------------------

def test_macro_summary_keys():
    y_int, y_cf, y_ca = _make_preds(seed=18)
    pids = _patient_ids()
    wilcoxon_df = h2_wilcoxon_per_class(y_cf, y_ca, y_int, CLASSES, pids,
                                        n_resamples=50, rng=np.random.default_rng(19))
    summary = h2_macro_summary(wilcoxon_df, y_cf, y_ca, y_int, pids,
                               n_resamples=50, rng=np.random.default_rng(20))
    for key in ("macro_BS_cf", "macro_BS_ca", "macro_improvement",
                "ci_lower_improvement", "ci_upper_improvement",
                "n_classes_significant", "n_classes_total"):
        assert key in summary


def test_macro_summary_n_classes_total():
    y_int, y_cf, y_ca = _make_preds(seed=21)
    pids = _patient_ids()
    wilcoxon_df = h2_wilcoxon_per_class(y_cf, y_ca, y_int, CLASSES, pids,
                                        n_resamples=50, rng=np.random.default_rng(22))
    summary = h2_macro_summary(wilcoxon_df, y_cf, y_ca, y_int, pids,
                               n_resamples=50, rng=np.random.default_rng(23))
    assert summary["n_classes_total"] == C
