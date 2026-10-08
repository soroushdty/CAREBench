"""H1 (Context Sensitivity) hypothesis tests.

Three prespecified inferences per §3.7:
  1. Per-class exact binomial test (sign agreement rate, BH FDR q=0.05)
  2. Patient-clustered permutation test on aggregate sign agreement
  3. Cochran-Mantel-Haenszel (CMH) pooled cross-class inference
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from shared.statistical.bootstrap import bootstrap_rate
from shared.statistical.delta import sign_agreement_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BH correction helper (scipy ≥1.11 has false_discovery_control; fall back)
# ---------------------------------------------------------------------------

def _bh_correct(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    try:
        from scipy.stats import false_discovery_control
        return false_discovery_control(p_values, method="bh")
    except ImportError:
        pass
    # Manual BH
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1  # 1-based
    adj = np.minimum(1.0, p_values * n / ranks)
    # Enforce monotonicity (take cumulative minimum from largest rank)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return adj


# ---------------------------------------------------------------------------
# 1. Per-class binomial test
# ---------------------------------------------------------------------------

def h1_binomial_per_class(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    class_list: list[str],
    eligible_classes: list[str],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Exact binomial test per class on sign agreement rate (§3.7 H1 per-class).

    For each class c, restricted to items with Δ_p(i,c) ≠ 0:
      - Compute sign agreement rate (proportion where sign(Δ_m)==sign(Δ_p))
      - Exact binomial test: H₀: p = 0.5, alternative: greater
      - 95% patient-level bootstrap CI on the agreement rate
    BH FDR correction (q=0.05) applied across confirmatory-eligible classes only.

    Returns a DataFrame with columns:
      Class, n_nonzero, sign_agree_rate, ci_lower, ci_upper,
      binom_p, bh_adj_p, confirmatory
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    pid = np.asarray(patient_ids)
    eligible_set = set(eligible_classes)

    rows = []
    confirmatory_p_values: list[float] = []
    confirmatory_indices: list[int] = []

    for c_idx, cls in enumerate(class_list):
        nz_mask = dp[:, c_idx] != 0.0
        n_nz = int(nz_mask.sum())

        if n_nz == 0:
            rows.append({
                "Class": cls,
                "n_nonzero": 0,
                "sign_agree_rate": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "binom_p": float("nan"),
                "bh_adj_p": float("nan"),
                "confirmatory": cls in eligible_set,
            })
            continue

        agree = sign_agreement_mask(dp[:, c_idx : c_idx + 1], dm[:, c_idx : c_idx + 1])
        agree_1d = agree[:, 0][nz_mask]
        pid_nz = pid[nz_mask]

        k = int(agree_1d.sum())
        rate = float(k / n_nz)

        # Patient-level bootstrap CI
        point, ci_lo, ci_hi = bootstrap_rate(agree_1d, pid_nz, n_resamples, rng)

        # Exact binomial test
        result = binomtest(k, n_nz, p=0.5, alternative="greater")
        p_val = float(result.pvalue)

        row_idx = len(rows)
        rows.append({
            "Class": cls,
            "n_nonzero": n_nz,
            "sign_agree_rate": rate,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "binom_p": p_val,
            "bh_adj_p": float("nan"),  # filled below
            "confirmatory": cls in eligible_set,
        })

        if cls in eligible_set:
            confirmatory_p_values.append(p_val)
            confirmatory_indices.append(row_idx)

    df = pd.DataFrame(rows)

    if confirmatory_p_values:
        adj = _bh_correct(np.array(confirmatory_p_values), q=0.05)
        for adj_val, row_idx in zip(adj, confirmatory_indices):
            df.at[row_idx, "bh_adj_p"] = float(adj_val)

    return df


# ---------------------------------------------------------------------------
# 2. Patient-clustered permutation test
# ---------------------------------------------------------------------------

def h1_permutation_test(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    patient_ids: np.ndarray,
    n_permutations: int = 10_000,
    eligible_classes: list[str] | None = None,
    class_list: list[str] | None = None,
    rng: np.random.Generator | None = None,
    n_resamples: int = 1000,
) -> dict:
    """Patient-clustered permutation test on aggregate sign agreement (§3.7 H1 aggregate).

    Pooled over (item, class) pairs with non-zero physician delta across all
    confirmatory-eligible classes (or all classes if eligible_classes is None).

    Permutation: model deltas are permuted within patient (entire patient row
    vector is permuted together within each patient block), preserving
    within-patient correlation.

    Returns dict with keys:
        aggregate_rate, ci_lower, ci_upper, null_mean, null_std, p_value
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    pid = np.asarray(patient_ids)

    if rng is None:
        rng = np.random.default_rng()

    # Restrict to eligible classes if specified
    if eligible_classes is not None and class_list is not None:
        eligible_set = set(eligible_classes)
        col_mask = np.array([cls in eligible_set for cls in class_list])
        dp = dp[:, col_mask]
        dm = dm[:, col_mask]

    # Observed aggregate rate
    def _aggregate_rate(dp_: np.ndarray, dm_: np.ndarray) -> float:
        agree = sign_agreement_mask(dp_, dm_)
        nz = dp_ != 0.0
        total_nz = int(nz.sum())
        if total_nz == 0:
            return float("nan")
        return float(agree.sum() / total_nz)

    observed = _aggregate_rate(dp, dm)

    # Short-circuit: if there are no non-zero physician deltas (e.g. all dp == 0),
    # the statistic is undefined everywhere — return all NaN without running the
    # permutation loop (which would otherwise trigger np.nanmean("Mean of empty
    # slice") warnings on an all-NaN null_vals array).
    if np.isnan(observed):
        return {
            "aggregate_rate": observed,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "p_value": float("nan"),
            "n_permutations": n_permutations,
        }

    # Bootstrap CI on observed rate
    unique_pts = np.unique(pid)
    pt_to_ix = {pt: np.where(pid == pt)[0] for pt in unique_pts}

    def _boot_rate(idx: np.ndarray) -> float:
        return _aggregate_rate(dp[idx], dm[idx])

    from shared.statistical.bootstrap import patient_block_bootstrap
    ci_lo, ci_hi = patient_block_bootstrap(_boot_rate, pid, n_resamples, rng)

    # Null distribution: permute dm within each patient block
    null_vals = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        dm_perm = dm.copy()
        for pt in unique_pts:
            ix = pt_to_ix[pt]
            # Permute the row ordering within this patient's items
            dm_perm[ix] = dm_perm[rng.permutation(ix)]
        null_vals[i] = _aggregate_rate(dp, dm_perm)

    null_mean = float(np.nanmean(null_vals))
    null_std = float(np.nanstd(null_vals))
    p_value = float(np.mean(null_vals >= observed))

    return {
        "aggregate_rate": observed,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# 3. Cochran-Mantel-Haenszel test
# ---------------------------------------------------------------------------

def h1_cmh_test(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
) -> dict:
    """Class-stratified CMH test for pooled sign-agreement evidence (§3.7 H1 pooled).

    For each confirmatory-eligible class, builds a valid 2×2 contingency table
    stratified by physician-delta direction:

        Rows    — physician direction:  Δ_p > 0 (row 0) vs Δ_p < 0 (row 1)
        Columns — model direction:      Δ_m > 0 (col 0) vs Δ_m ≤ 0 (col 1)

        Cell (0,0): Δ_p>0, Δ_m>0  → physician-model both positive (agree)
        Cell (0,1): Δ_p>0, Δ_m≤0  → physician positive, model non-positive (disagree)
        Cell (1,0): Δ_p<0, Δ_m>0  → physician negative, model positive (disagree)
        Cell (1,1): Δ_p<0, Δ_m≤0  → physician-model both non-positive (agree)

        Columns are CONSISTENT across rows (both represent model direction),
        so the common odds ratio is interpretable:
            OR = (a·d)/(b·c) = r²/(1−r)²  where r = sign-agreement rate
            OR > 1  ⟺  sign agreement > 50%

    Strata where either physician direction is absent (all Δ_p in one direction)
    are skipped to avoid degenerate tables with a zero row margin.

    Combines tables via statsmodels StratifiedTable (common-odds-ratio test).

    Returns dict with keys:
        common_odds_ratio, ci_lower, ci_upper, chi2_cmh, p_cmh, n_strata
    Falls back to a manual Mantel-Haenszel calculation if statsmodels is
    unavailable.
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    eligible_set = set(eligible_classes)

    tables: list[np.ndarray] = []
    for c_idx, cls in enumerate(class_list):
        if cls not in eligible_set:
            continue
        dp_col = dp[:, c_idx]
        dm_col = dm[:, c_idx]

        # Physician-positive and physician-negative item masks
        pos = dp_col > 0.0
        neg = dp_col < 0.0

        # Skip degenerate strata: need at least one item in each physician direction
        if pos.sum() == 0 or neg.sum() == 0:
            continue

        # 2×2: rows = physician direction, cols = model direction (dm>0 vs dm≤0)
        # Columns have the SAME meaning across both rows — this is required for a
        # valid CMH common-odds-ratio test.
        a = int((pos & (dm_col > 0.0)).sum())   # Δ_p>0, Δ_m>0  (agree for pos physician)
        b = int((pos & (dm_col <= 0.0)).sum())  # Δ_p>0, Δ_m≤0  (disagree for pos physician)
        c = int((neg & (dm_col > 0.0)).sum())   # Δ_p<0, Δ_m>0  (disagree for neg physician)
        d = int((neg & (dm_col <= 0.0)).sum())  # Δ_p<0, Δ_m≤0  (agree for neg physician)

        # Skip if any margin is zero (degenerate for CMH)
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        tables.append(np.array([[a, b], [c, d]], dtype=float))

    if not tables:
        return {
            "common_odds_ratio": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "chi2_cmh": float("nan"),
            "p_cmh": float("nan"),
            "n_strata": 0,
        }

    try:
        from statsmodels.stats.contingency_tables import StratifiedTable
        st = StratifiedTable(np.stack(tables, axis=2))
        summary = st.test_null_odds()
        est = st.oddsratio_pooled
        ci = st.oddsratio_pooled_confint()
        return {
            "common_odds_ratio": float(est),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "chi2_cmh": float(summary.statistic),
            "p_cmh": float(summary.pvalue),
            "n_strata": len(tables),
        }
    except Exception as exc:
        logger.warning("statsmodels StratifiedTable failed (%s); using manual MH.", exc)
        return _manual_mh(tables)


def _manual_mh(tables: list[np.ndarray]) -> dict:
    """Manual Mantel-Haenszel common odds ratio (without CI)."""
    from scipy.stats import chi2

    num = denom = 0.0
    chi2_num = 0.0

    for t in tables:
        a, b = t[0, 0], t[0, 1]
        c, d = t[1, 0], t[1, 1]
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        denom += b * c / n
        E_a = (a + b) * (a + c) / n
        V_a = (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1)) if n > 1 else 0.0
        chi2_num += (a - E_a)

    or_mh = num / denom if denom != 0 else float("nan")
    stat = chi2_num ** 2 / max(sum(
        (t[0, 0] + t[0, 1]) * (t[1, 0] + t[1, 1]) * (t[0, 0] + t[1, 0]) *
        (t[0, 1] + t[1, 1]) / ((sum(t.flat)) ** 2 * (sum(t.flat) - 1))
        for t in tables
        if sum(t.flat) > 1
    ), 1e-12)

    p = float(1.0 - chi2.cdf(stat, df=1))
    return {
        "common_odds_ratio": float(or_mh),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "chi2_cmh": float(stat),
        "p_cmh": p,
        "n_strata": len(tables),
    }
