"""H2 (Contextual Alignment) hypothesis tests.

Two analyses per §3.7:
  1. Paired per-class Wilcoxon signed-rank test on Brier-score differences
  2. Per-class Wasserstein distance between prediction and label distributions
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon

from shared.statistical.bootstrap import bootstrap_scalar

logger = logging.getLogger(__name__)


def _brier_per_item(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-item Brier score: (ŷ - y)². Returns (n,) array."""
    return np.mean((np.asarray(y_hat, float) - np.asarray(y_true, float)) ** 2, axis=1)


def _brier_per_item_per_class(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-item-per-class squared error. Returns (n, n_classes)."""
    return (np.asarray(y_hat, float) - np.asarray(y_true, float)) ** 2


def _bh_correct(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (imported from h1 to avoid duplication)."""
    try:
        from scipy.stats import false_discovery_control
        return false_discovery_control(p_values, method="bh")
    except ImportError:
        pass
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1
    adj = np.minimum(1.0, p_values * n / ranks)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return adj


# ---------------------------------------------------------------------------
# 1. Per-class Wilcoxon signed-rank test on Brier-score differences
# ---------------------------------------------------------------------------

def h2_wilcoxon_per_class(
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    class_list: list[str],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test per class on Brier-score improvement (§3.7 H2).

    For each class c:
        BS_cf(i,c) = (ŷ_cf(i,c) - y_int(i,c))²
        BS_ca(i,c) = (ŷ_ca(i,c) - y_int(i,c))²
        diff(i,c)  = BS_cf(i,c) - BS_ca(i,c)   (positive = context-aware better)
    Wilcoxon signed-rank test: H₀: median(diff) = 0, alternative: greater.
    BH FDR correction (q=0.05) across classes.
    95% patient-level bootstrap CI on (BS_cf - BS_ca) mean improvement.

    Returns DataFrame with columns:
        Class, BS_cf, BS_ca, improvement, ci_lower, ci_upper,
        wilcoxon_stat, wilcoxon_p, bh_adj_p
    """
    cf = np.asarray(y_hat_cf, dtype=np.float64)
    ca = np.asarray(y_hat_ca, dtype=np.float64)
    yi = np.asarray(y_interview, dtype=np.float64)
    pid = np.asarray(patient_ids)

    rows = []
    raw_p: list[float] = []

    for c_idx, cls in enumerate(class_list):
        bs_cf_col = (cf[:, c_idx] - yi[:, c_idx]) ** 2
        bs_ca_col = (ca[:, c_idx] - yi[:, c_idx]) ** 2
        diff = bs_cf_col - bs_ca_col  # positive = CA better

        bs_cf_mean = float(bs_cf_col.mean())
        bs_ca_mean = float(bs_ca_col.mean())
        improve = float(diff.mean())

        # Patient-level bootstrap CI on mean improvement
        _, ci_lo, ci_hi = bootstrap_scalar(diff, pid, np.mean, n_resamples, rng)

        # Wilcoxon signed-rank test
        if np.all(diff == 0):
            stat, p = float("nan"), 1.0
        else:
            try:
                res = wilcoxon(diff, alternative="greater", zero_method="wilcox")
                stat, p = float(res.statistic), float(res.pvalue)
            except Exception as exc:
                logger.warning("Wilcoxon failed for class %s: %s", cls, exc)
                stat, p = float("nan"), float("nan")

        rows.append({
            "Class": cls,
            "BS_cf": bs_cf_mean,
            "BS_ca": bs_ca_mean,
            "improvement": improve,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "bh_adj_p": float("nan"),
        })
        raw_p.append(p if not np.isnan(p) else 1.0)

    df = pd.DataFrame(rows)
    if raw_p:
        adj = _bh_correct(np.array(raw_p))
        df["bh_adj_p"] = adj.tolist()

    return df


# ---------------------------------------------------------------------------
# 2. Per-class Wasserstein distance
# ---------------------------------------------------------------------------

def h2_wasserstein_per_class(
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    class_list: list[str],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-class Wasserstein-1 distance between predictions and interview labels (§5.4.3).

    W_cf(c) = W₁(ŷ_cf[:,c], y_int[:,c])
    W_ca(c) = W₁(ŷ_ca[:,c], y_int[:,c])

    95% patient-level bootstrap CIs on each distance.

    Returns DataFrame with columns:
        Class, W_cf, W_ca,
        ci_lower_cf, ci_upper_cf, ci_lower_ca, ci_upper_ca
    """
    cf = np.asarray(y_hat_cf, dtype=np.float64)
    ca = np.asarray(y_hat_ca, dtype=np.float64)
    yi = np.asarray(y_interview, dtype=np.float64)
    pid = np.asarray(patient_ids)

    rows = []
    for c_idx, cls in enumerate(class_list):
        cf_col = cf[:, c_idx]
        ca_col = ca[:, c_idx]
        yi_col = yi[:, c_idx]

        w_cf = float(wasserstein_distance(cf_col, yi_col))
        w_ca = float(wasserstein_distance(ca_col, yi_col))

        def _w_cf(idx: np.ndarray) -> float:
            return float(wasserstein_distance(cf_col[idx], yi_col[idx]))

        def _w_ca(idx: np.ndarray) -> float:
            return float(wasserstein_distance(ca_col[idx], yi_col[idx]))

        from shared.statistical.bootstrap import patient_block_bootstrap
        ci_cf_lo, ci_cf_hi = patient_block_bootstrap(_w_cf, pid, n_resamples, rng)
        ci_ca_lo, ci_ca_hi = patient_block_bootstrap(_w_ca, pid, n_resamples, rng)

        rows.append({
            "Class": cls,
            "W_cf": w_cf,
            "W_ca": w_ca,
            "ci_lower_cf": ci_cf_lo,
            "ci_upper_cf": ci_cf_hi,
            "ci_lower_ca": ci_ca_lo,
            "ci_upper_ca": ci_ca_hi,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Macro-average summary for H2
# ---------------------------------------------------------------------------

def h2_macro_summary(
    wilcoxon_df: pd.DataFrame,
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Macro-average Brier score and improvement with CI (§5.4.2).

    Returns dict with:
        macro_BS_cf, macro_BS_ca, macro_improvement,
        ci_lower_improvement, ci_upper_improvement,
        n_classes_significant (BH q<0.05)
    """
    cf = np.asarray(y_hat_cf, dtype=np.float64)
    ca = np.asarray(y_hat_ca, dtype=np.float64)
    yi = np.asarray(y_interview, dtype=np.float64)
    pid = np.asarray(patient_ids)

    bs_cf_all = (cf - yi) ** 2  # (n, n_classes)
    bs_ca_all = (ca - yi) ** 2
    diff_all = bs_cf_all - bs_ca_all  # positive = CA better

    macro_bs_cf = float(bs_cf_all.mean())
    macro_bs_ca = float(bs_ca_all.mean())
    macro_improve = float(diff_all.mean())

    def _macro_improve(idx: np.ndarray) -> float:
        return float((bs_cf_all[idx] - bs_ca_all[idx]).mean())

    from shared.statistical.bootstrap import patient_block_bootstrap
    ci_lo, ci_hi = patient_block_bootstrap(_macro_improve, pid, n_resamples, rng)

    n_sig = int((wilcoxon_df["bh_adj_p"] < 0.05).sum()) if "bh_adj_p" in wilcoxon_df else 0

    return {
        "macro_BS_cf": macro_bs_cf,
        "macro_BS_ca": macro_bs_ca,
        "macro_improvement": macro_improve,
        "ci_lower_improvement": ci_lo,
        "ci_upper_improvement": ci_hi,
        "n_classes_significant": n_sig,
        "n_classes_total": len(wilcoxon_df),
    }
