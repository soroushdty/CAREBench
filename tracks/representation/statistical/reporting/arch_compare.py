"""Architectural comparison table with paired bootstrap differences.

Benchmarks the four-vector fusion head against alternative fusion
architectures and baselines.  Accepts a dict of predictions gathered from
separate pipeline runs (one per architecture) or from a multi-arch run.

Expected architecture keys:
  Fusion heads:   '4_vector', '2d', '3d', 'lowrank_bilinear'
  Baselines:      'passthrough', 'patient_id', 'stage1_only'
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate parameter counts per class per architecture
_PARAM_COUNTS: dict[str, str] = {
    "4_vector":         "4d (30,720/class at d=768)",
    "2d":               "2d (1,536/class at d=768)",
    "3d":               "3d (2,304/class at d=768)",
    "lowrank_bilinear": "2·d·r (12,288/class at d=768, r=8)",
    "passthrough":      "0 (no learned parameters)",
    "patient_id":       "d + n_patients",
    "stage1_only":      "0 (Stage 1 frozen)",
}


def _macro_brier(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def _macro_f1(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    from tracks.representation.training.shared.soft_label_utils import _soft_scores
    f1_scores = []
    for c in range(y_true.shape[1]):
        s = _soft_scores(y_true[:, c], y_pred[:, c], float(thresholds[c]))
        f1_scores.append(s["F1"])
    return float(np.mean(f1_scores))


def arch_comparison_table(
    arch_predictions: dict[str, np.ndarray],
    y_interview: np.ndarray,
    class_list: list[str],
    patient_ids: np.ndarray,
    thresholds: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    reference_arch: str = "2d",
) -> pd.DataFrame:
    """Architectural comparison table.

    For each architecture in arch_predictions:
      - Macro-Brier score (against y_interview)
      - Macro-F1 (using provided thresholds)
      - 95% patient-level bootstrap CI on both
      - Paired bootstrap difference vs. reference architecture

    Args:
        arch_predictions: {arch_name: (n, n_classes) predictions in [0,1]}.
        y_interview:      (n, n_classes) ground-truth interview labels.
        class_list:       List of class names.
        patient_ids:      (n,) patient IDs for cluster bootstrap.
        thresholds:       (n_classes,) decision thresholds.
        n_resamples:      Bootstrap resamples.
        reference_arch:   Architecture to compute paired differences against.

    Returns:
        DataFrame with columns:
            architecture, param_count, macro_brier, brier_ci_lower, brier_ci_upper,
            macro_f1, f1_ci_lower, f1_ci_upper,
            delta_brier_vs_ref, delta_brier_ci_lower, delta_brier_ci_upper,
            delta_f1_vs_ref, delta_f1_ci_lower, delta_f1_ci_upper
    """
    if rng is None:
        rng = np.random.default_rng()

    yi = np.asarray(y_interview, dtype=np.float64)
    pid = np.asarray(patient_ids)
    thresh = np.asarray(thresholds, dtype=np.float64)

    from shared.statistical.bootstrap import patient_block_bootstrap

    rows = []
    for arch_name, preds in arch_predictions.items():
        p = np.asarray(preds, dtype=np.float64)

        brier_pt = _macro_brier(p, yi)
        f1_pt = _macro_f1(p, yi, thresh)

        def _brier_fn(idx: np.ndarray) -> float:
            return _macro_brier(p[idx], yi[idx])

        def _f1_fn(idx: np.ndarray) -> float:
            return _macro_f1(p[idx], yi[idx], thresh)

        b_lo, b_hi = patient_block_bootstrap(_brier_fn, pid, n_resamples, rng)
        f_lo, f_hi = patient_block_bootstrap(_f1_fn, pid, n_resamples, rng)

        rows.append({
            "architecture": arch_name,
            "param_count": _PARAM_COUNTS.get(arch_name, "unknown"),
            "macro_brier": brier_pt,
            "brier_ci_lower": b_lo,
            "brier_ci_upper": b_hi,
            "macro_f1": f1_pt,
            "f1_ci_lower": f_lo,
            "f1_ci_upper": f_hi,
        })

    df = pd.DataFrame(rows)

    # Paired bootstrap differences vs reference architecture
    if reference_arch not in arch_predictions:
        logger.warning(
            "Reference architecture %r not in arch_predictions; "
            "paired bootstrap differences skipped.",
            reference_arch,
        )
        for col in ("delta_brier_vs_ref", "delta_brier_ci_lower", "delta_brier_ci_upper",
                    "delta_f1_vs_ref", "delta_f1_ci_lower", "delta_f1_ci_upper"):
            df[col] = float("nan")
        return df

    ref_preds = np.asarray(arch_predictions[reference_arch], dtype=np.float64)
    delta_rows: dict[str, list] = {
        "delta_brier_vs_ref": [],
        "delta_brier_ci_lower": [],
        "delta_brier_ci_upper": [],
        "delta_f1_vs_ref": [],
        "delta_f1_ci_lower": [],
        "delta_f1_ci_upper": [],
    }

    for _, row in df.iterrows():
        arch_name = row["architecture"]
        if arch_name not in arch_predictions:
            for k in delta_rows:
                delta_rows[k].append(float("nan"))
            continue

        p = np.asarray(arch_predictions[arch_name], dtype=np.float64)

        delta_b = _macro_brier(p, yi) - _macro_brier(ref_preds, yi)
        delta_f = _macro_f1(p, yi, thresh) - _macro_f1(ref_preds, yi, thresh)

        def _db(idx: np.ndarray) -> float:
            return _macro_brier(p[idx], yi[idx]) - _macro_brier(ref_preds[idx], yi[idx])

        def _df(idx: np.ndarray) -> float:
            return _macro_f1(p[idx], yi[idx], thresh) - _macro_f1(ref_preds[idx], yi[idx], thresh)

        db_lo, db_hi = patient_block_bootstrap(_db, pid, n_resamples, rng)
        df_lo, df_hi = patient_block_bootstrap(_df, pid, n_resamples, rng)

        delta_rows["delta_brier_vs_ref"].append(delta_b)
        delta_rows["delta_brier_ci_lower"].append(db_lo)
        delta_rows["delta_brier_ci_upper"].append(db_hi)
        delta_rows["delta_f1_vs_ref"].append(delta_f)
        delta_rows["delta_f1_ci_lower"].append(df_lo)
        delta_rows["delta_f1_ci_upper"].append(df_hi)

    for col, vals in delta_rows.items():
        df[col] = vals

    return df
