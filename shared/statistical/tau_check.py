"""τ=0.5 vs F1-optimal threshold robustness check (§3.4 / §5.1).

Per §3.4: the τ_c=0.5 robustness check verifies that the fixed threshold
lies within ~0.05 of the F1-optimal threshold selected during inner-fold
validation.  Classes where |τ_fixed - τ_f1opt| > 0.05 are flagged and both
thresholds are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tau_robustness_check(
    tau_fixed: float,
    avg_thresh_f1opt: np.ndarray,
    class_list: list[str],
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compare fixed τ against per-class F1-optimal thresholds.

    Args:
        tau_fixed:         Fixed threshold (cfg['tau'], default 0.5).
        avg_thresh_f1opt:  (n_classes,) mean inner-fold F1-optimal thresholds.
        class_list:        List of class names, length n_classes.
        tolerance:         Flag threshold; §3.4 uses 0.05.

    Returns:
        DataFrame with columns:
            Class, tau_fixed, tau_f1opt, abs_diff, within_tolerance
        Rows are sorted by abs_diff descending so flagged classes appear first.
    """
    thresholds = np.asarray(avg_thresh_f1opt, dtype=np.float64)
    rows = []
    for c_idx, cls in enumerate(class_list):
        tau_opt = float(thresholds[c_idx])
        diff = abs(tau_fixed - tau_opt)
        rows.append({
            "Class": cls,
            "tau_fixed": float(tau_fixed),
            "tau_f1opt": tau_opt,
            "abs_diff": diff,
            "within_tolerance": diff <= tolerance,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return df


def tau_robustness_summary(df: pd.DataFrame) -> str:
    """One-line summary sentence for §5.1 inline text."""
    n_total = len(df)
    n_flagged = int((~df["within_tolerance"]).sum())
    if n_flagged == 0:
        return (
            f"All {n_total} class thresholds lay within 0.05 of the F1-optimal "
            f"threshold (range: {df['abs_diff'].min():.3f}–{df['abs_diff'].max():.3f})."
        )
    flagged_cls = df[~df["within_tolerance"]]["Class"].tolist()
    return (
        f"{n_flagged} of {n_total} classes had |τ_fixed − τ_F1opt| > 0.05: "
        f"{', '.join(flagged_cls)}. Both thresholds are reported for these classes."
    )
