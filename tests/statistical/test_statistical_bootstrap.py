"""Tests for shared/statistical/bootstrap.py."""
import numpy as np
from shared.statistical.bootstrap import patient_block_bootstrap, bootstrap_rate, bootstrap_scalar


def _make_data(n=24, n_patients=4, seed=0):
    rng = np.random.default_rng(seed)
    patient_ids = np.repeat(np.arange(n_patients), n // n_patients)
    values = rng.random(n)
    return values, patient_ids


def test_patient_block_bootstrap_returns_tuple():
    vals, pids = _make_data()
    fn = lambda idx: float(np.mean(vals[idx]))
    lo, hi = patient_block_bootstrap(fn, pids, n_resamples=200)
    assert lo <= hi
    assert not np.isnan(lo)
    assert not np.isnan(hi)


def test_patient_block_bootstrap_too_few_patients():
    vals = np.array([0.5, 0.3, 0.7])
    pids = np.array([1, 1, 1])  # only 1 unique patient
    fn = lambda idx: float(np.mean(vals[idx]))
    lo, hi = patient_block_bootstrap(fn, pids, n_resamples=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_rate_point_estimate():
    indicator = np.array([1, 1, 0, 1, 0, 1], dtype=float)
    pids = np.array([1, 1, 2, 2, 3, 3])
    point, lo, hi = bootstrap_rate(indicator, pids, n_resamples=200)
    assert abs(point - 4 / 6) < 1e-6
    assert lo <= point <= hi


def test_bootstrap_scalar_ci_coverage():
    rng = np.random.default_rng(42)
    n_patients = 6
    n_per_patient = 10
    n = n_patients * n_per_patient
    pids = np.repeat(np.arange(n_patients), n_per_patient)
    # True mean = 0.5
    values = rng.beta(2, 2, size=n)
    point, lo, hi = bootstrap_scalar(values, pids, n_resamples=500, rng=rng)
    assert lo <= point <= hi
    # CI should be reasonably narrow
    assert (hi - lo) < 0.5


def test_bootstrap_scalar_deterministic_with_seed():
    vals, pids = _make_data()
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    fn = np.mean
    _, lo1, hi1 = bootstrap_scalar(vals, pids, agg_fn=fn, n_resamples=100, rng=rng1)
    _, lo2, hi2 = bootstrap_scalar(vals, pids, agg_fn=fn, n_resamples=100, rng=rng2)
    assert lo1 == lo2
    assert hi1 == hi2
