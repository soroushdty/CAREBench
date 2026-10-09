"""Per-class Expected Calibration Error (ECE) with patient-level bootstrap CIs.

ECE per class is computed using equal-width probability bins (default 15).
Classes where ECE exceeds 0.10 are flagged.

Bootstrap resampling is at the patient level (block bootstrap) to preserve
within-patient correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.statistical.bootstrap import patient_block_bootstrap


def _ece_1d(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE for a single class: weighted mean |bin_confidence - bin_accuracy|."""
    p = np.asarray(y_pred, dtype=np.float64)
    t = np.asarray(y_true, dtype=np.float64)
    n = len(p)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == bins[-2]:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(t[mask].mean())
        ece += mask.sum() / n * abs(conf - acc)
    return float(ece)


def calibration_ece_per_class(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    class_list: list[str],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    n_bins: int = 15,
    ece_flag_threshold: float = 0.10,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-class ECE with 95% patient-level bootstrap CIs and >0.10 flag.

    Args:
        y_hat:       (n, n_classes) model predictions in [0, 1].
        y_true:      (n, n_classes) ground-truth labels (pair-aggregated, in {0, 0.5, 1}).
        class_list:  List of class names.
        patient_ids: (n,) patient IDs for cluster bootstrap.
        n_resamples: Bootstrap resamples (default 1000).
        n_bins:      Calibration bins (default 15).
        ece_flag_threshold: Classes with ECE > this value are flagged (default 0.10).
        rng:         Optional numpy Generator.

    Returns:
        DataFrame with columns:
            Class, ECE, ci_lower, ci_upper, flagged (bool: ECE > ece_flag_threshold)
    """
    yp = np.asarray(y_hat, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)
    pid = np.asarray(patient_ids)

    rows = []
    for c_idx, cls in enumerate(class_list):
        p_col = yp[:, c_idx]
        t_col = yt[:, c_idx]

        ece_pt = _ece_1d(p_col, t_col, n_bins)

        def _ece_fn(idx: np.ndarray, _p=p_col, _t=t_col) -> float:
            return _ece_1d(_p[idx], _t[idx], n_bins)

        ci_lo, ci_hi = patient_block_bootstrap(_ece_fn, pid, n_resamples, rng)

        rows.append({
            "Class": cls,
            "ECE": ece_pt,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "flagged": (not np.isnan(ece_pt)) and (ece_pt > ece_flag_threshold),
        })

    return pd.DataFrame(rows)
