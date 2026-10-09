"""Context-induced entropy change analysis (exploratory).

Binary entropy H(p) applied per class per item to characterize whether the
model's predictive uncertainty changes from context-free to context-aware in
a manner consistent with physician behavior.

No hypothesis testing — descriptive / exploratory only.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """H(p) = -p·log₂(p) - (1-p)·log₂(1-p), with 0·log(0)=0 convention.

    Args:
        p: Array of probabilities in [0, 1], any shape.

    Returns:
        Entropy array same shape as p, in bits.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    # Avoid log(0) by using where
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            (p == 0.0) | (p == 1.0),
            0.0,
            -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p),
        )
    return h


def physician_entropy_change(
    y_survey: np.ndarray,
    y_interview: np.ndarray,
) -> np.ndarray:
    """H(y_interview) - H(y_survey) per (item, class).

    Pair-aggregated labels are in {0.0, 0.5, 1.0}:
      H(0.5) ≈ 1.0 bit  (maximal physician disagreement)
      H(0.0) = H(1.0) = 0.0 bits  (full agreement)

    A negative change means context resolved disagreement (interview→agreement).
    A positive change means context introduced disagreement.

    Returns:
        (n, n_classes) float64 array.
    """
    ys = np.asarray(y_survey, dtype=np.float64)
    yi = np.asarray(y_interview, dtype=np.float64)
    return binary_entropy(yi) - binary_entropy(ys)


def model_entropy_change(
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
) -> np.ndarray:
    """H(ŷ_ca) - H(ŷ_cf) per (item, class).

    Returns:
        (n, n_classes) float64 array.
    """
    cf = np.asarray(y_hat_cf, dtype=np.float64)
    ca = np.asarray(y_hat_ca, dtype=np.float64)
    return binary_entropy(ca) - binary_entropy(cf)


def entropy_pearson_r(
    phys_entropy_delta: np.ndarray,
    model_entropy_delta: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
) -> dict:
    """Pearson r between physician and model entropy change.

    Pooled across confirmatory-eligible classes.
    Treated as descriptive only — no hypothesis correction applied.

    Args:
        phys_entropy_delta:  (n, n_classes) physician entropy change.
        model_entropy_delta: (n, n_classes) model entropy change.
        eligible_classes:    Classes to pool over.
        class_list:          All class names (for indexing).

    Returns:
        dict with keys: pearson_r, p_value, n_observations
    """
    eligible_set = set(eligible_classes)
    col_indices = [i for i, cls in enumerate(class_list) if cls in eligible_set]

    if not col_indices:
        return {
            "pearson_r": float("nan"),
            "p_value_descriptive_only": float("nan"),
            "descriptive_only": True,
            "n_observations": 0,
        }

    ph = phys_entropy_delta[:, col_indices].ravel()
    mo = model_entropy_delta[:, col_indices].ravel()

    # Remove NaN pairs
    valid = ~(np.isnan(ph) | np.isnan(mo))
    ph = ph[valid]
    mo = mo[valid]

    if len(ph) < 3:
        return {
            "pearson_r": float("nan"),
            "p_value_descriptive_only": float("nan"),
            "descriptive_only": True,
            "n_observations": int(len(ph)),
        }

    r, p = pearsonr(ph, mo)
    return {
        "pearson_r": float(r),
        # Provided for completeness; NOT a confirmatory inference — no BH correction,
        # no decision threshold.  Treat as descriptive/exploratory only.
        "p_value_descriptive_only": float(p),
        "descriptive_only": True,
        "n_observations": int(len(ph)),
    }
