"""Patient-level cluster bootstrap confidence intervals.

Resamples whole patient blocks with replacement so that CIs reflect
between-patient variability rather than within-patient noise.  The pattern
mirrors soft_brier_with_ci() in tracks/representation/training/shared/soft_label_utils.py.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def patient_block_bootstrap(
    metric_fn: Callable[[np.ndarray], float],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Patient-level cluster bootstrap CI for any scalar metric.

    Args:
        metric_fn: Callable that takes an index array (subset of 0..n-1)
                   and returns a scalar metric value.  Called once per
                   bootstrap resample plus once to build pt_to_ix.
        patient_ids: (n,) array of patient IDs, one per observation.
        n_resamples: Number of bootstrap resamples.
        rng: Optional numpy Generator; created fresh if None.
        ci_level: Coverage level, default 0.95.

    Returns:
        (ci_lower, ci_upper) at the requested coverage level.
        Returns (nan, nan) if fewer than 2 unique patients are present.
    """
    patient_ids = np.asarray(patient_ids)
    unique_pts = np.unique(patient_ids)

    if len(unique_pts) < 2:
        logger.warning(
            "Bootstrap CI skipped: fewer than 2 unique patients. Returning (nan, nan)."
        )
        return (float("nan"), float("nan"))

    if rng is None:
        rng = np.random.default_rng()

    pt_to_ix: dict[object, np.ndarray] = {
        pt: np.where(patient_ids == pt)[0] for pt in unique_pts
    }

    alpha = 1.0 - ci_level
    boot_vals = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sampled = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        idx = np.concatenate([pt_to_ix[pt] for pt in sampled])
        boot_vals[i] = metric_fn(idx)

    lo = float(np.percentile(boot_vals, 100.0 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def bootstrap_rate(
    indicator: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate + CI for a binary indicator (sign agreement, etc.).

    Args:
        indicator: (n,) boolean or 0/1 float array restricted to the
                   observations of interest (e.g. non-zero Δ_p rows only).
        patient_ids: (n,) patient IDs parallel to indicator.

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    indicator = np.asarray(indicator, dtype=float)
    point = float(np.mean(indicator))

    def _rate(idx: np.ndarray) -> float:
        return float(np.mean(indicator[idx]))

    lo, hi = patient_block_bootstrap(_rate, patient_ids, n_resamples, rng, ci_level)
    return point, lo, hi


def bootstrap_scalar(
    values: np.ndarray,
    patient_ids: np.ndarray,
    agg_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate + CI for a scalar aggregate over an array of values.

    Args:
        values: (n,) float array.
        patient_ids: (n,) patient IDs parallel to values.
        agg_fn: Aggregation function applied to resampled values array.

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    values = np.asarray(values, dtype=float)
    point = float(agg_fn(values))

    def _agg(idx: np.ndarray) -> float:
        return float(agg_fn(values[idx]))

    lo, hi = patient_block_bootstrap(_agg, patient_ids, n_resamples, rng, ci_level)
    return point, lo, hi
