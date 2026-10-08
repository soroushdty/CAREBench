# Hypotheses (`tracks/representation/statistical/hypotheses`)

Canonical location for H1 and H2 hypothesis test implementations.

This package implements the two pre-specified confirmatory hypotheses: H1 (Context Sensitivity) and H2 (Contextual Alignment). All functions
operate on pre-computed delta and prediction arrays; they do not load data themselves.

---

## H1 — Context Sensitivity (`h1.py`)

**Research question:** Does the model update its predictions in response to patient context
in the same direction as the physicians?

H1 is evaluated via three independent inferences. All three are restricted to
confirmatory-eligible classes (those with ≥ 15 non-zero physician deltas) and
to items where the physician delta is non-zero (`Δ_p(i,c) ≠ 0`).

### `h1_binomial_per_class`

```python
h1_binomial_per_class(
    delta_p: np.ndarray,          # (n, n_classes) physician deltas
    delta_m: np.ndarray,          # (n, n_classes) model deltas
    class_list: list[str],
    eligible_classes: list[str],  # from confirmatory_eligible_classes()
    patient_ids: np.ndarray,      # (n,) for block bootstrap
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame
```

**Algorithm:**
1. For each class `c`, restrict to rows where `Δ_p[:, c] ≠ 0`.
2. Compute the sign agreement indicator: `sign(Δ_m(i,c)) == sign(Δ_p(i,c))`.
3. Count agreements `k` out of `n_nz` non-zero items.
4. Run `scipy.stats.binomtest(k, n_nz, p=0.5, alternative="greater")` — H₀: chance sign
   agreement rate = 0.5.
5. Compute a 95% patient-level bootstrap CI on the agreement rate via `bootstrap_rate()`.
6. Apply Benjamini-Hochberg FDR correction (q = 0.05) across all confirmatory-eligible
   classes only. Non-eligible classes receive `bh_adj_p = NaN`.

**Returns** a `pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `Class` | str | Class name |
| `n_nonzero` | int | Items with `Δ_p ≠ 0` for this class |
| `sign_agree_rate` | float | Proportion of non-zero items where signs agree |
| `ci_lower` | float | 95% bootstrap CI lower bound |
| `ci_upper` | float | 95% bootstrap CI upper bound |
| `binom_p` | float | Raw one-sided exact binomial p-value |
| `bh_adj_p` | float | BH-adjusted p-value (confirmatory classes only) |
| `confirmatory` | bool | Whether this class met the eligibility threshold |

**Saved to:** `h1_per_class.csv`

---

### `h1_permutation_test`

```python
h1_permutation_test(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    patient_ids: np.ndarray,
    n_permutations: int = 10_000,
    eligible_classes: list[str] | None = None,
    class_list: list[str] | None = None,
    rng: np.random.Generator | None = None,
    n_resamples: int = 1000,
) -> dict
```

**Algorithm:**
1. If `eligible_classes` is provided, restrict to those columns.
2. Pool all (item, class) pairs with `Δ_p ≠ 0` across eligible classes; compute the
   observed aggregate sign agreement rate.
3. For each of `n_permutations` replicates: permute the **row ordering** of `delta_m`
   within each patient block. This preserves within-patient correlation (all class
   columns for a patient's item shuffle together) while destroying the model–physician
   directional relationship. Compute the permuted aggregate rate.
4. P-value: `mean(null_vals >= observed)` (one-sided, upper tail).
5. Compute a 95% patient-level bootstrap CI on the observed rate.
6. If all physician deltas are zero (observed = NaN), the permutation loop is skipped
   and all values are returned as NaN to avoid spurious `mean of empty slice` warnings.

**Returns** a `dict` with keys:

| Key | Description |
|---|---|
| `aggregate_rate` | Observed pooled sign agreement rate |
| `ci_lower` | 95% bootstrap CI lower bound |
| `ci_upper` | 95% bootstrap CI upper bound |
| `null_mean` | Mean of the permutation null distribution |
| `null_std` | Standard deviation of the null distribution |
| `p_value` | Permutation p-value |
| `n_permutations` | Number of replicates used |

**Saved to:** `h1_aggregate.json`

---

### `h1_cmh_test`

```python
h1_cmh_test(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
) -> dict
```

**Algorithm:**

For each confirmatory-eligible class, builds a 2×2 contingency table stratified by
physician-delta direction:

```
                  Δ_m > 0    Δ_m ≤ 0
   Δ_p > 0  →  [ a         b ]   (physician-positive items)
   Δ_p < 0  →  [ c         d ]   (physician-negative items)
```

Columns have a **consistent meaning across both rows** (model direction), which makes the
common odds ratio interpretable: `OR = (a·d)/(b·c)`. When sign agreement exceeds 50%,
`OR > 1` in every valid stratum, so the CMH test pools evidence across classes coherently.

Classes where either physician direction is absent (all Δ_p one sign) are skipped to avoid
degenerate tables. Strata with any zero row or column margin are also skipped.

The test is computed via `statsmodels.stats.contingency_tables.StratifiedTable`. If
statsmodels is unavailable, a manual Mantel-Haenszel estimator is used (CI returned as NaN).

**Returns** a `dict` with keys:

| Key | Description |
|---|---|
| `common_odds_ratio` | MH pooled common odds ratio across eligible-class strata |
| `ci_lower` | 95% CI lower bound (statsmodels only; NaN with manual fallback) |
| `ci_upper` | 95% CI upper bound (statsmodels only; NaN with manual fallback) |
| `chi2_cmh` | CMH chi-squared statistic |
| `p_cmh` | CMH p-value (1 degree of freedom) |
| `n_strata` | Number of valid 2×2 tables included |

**Saved to:** `h1_cmh.json`

---

### `_bh_correct` (internal helper)

```python
_bh_correct(p_values: np.ndarray, q: float = 0.05) -> np.ndarray
```

Benjamini-Hochberg FDR correction. Tries `scipy.stats.false_discovery_control` (scipy ≥
1.11) and falls back to a manual implementation for older environments. Enforces cumulative
minimum from largest to smallest rank to guarantee monotonicity of adjusted p-values.
Returns adjusted p-values in the same order as the input.

---

## H2 — Contextual Alignment (`h2.py`)

**Research question:** Does including patient context move the model's predictions closer
to the physician interview labels (the ground truth)?

H2 uses the Brier score improvement `BS_cf − BS_ca` as the primary metric: positive values
mean the context-aware model is closer to the interview labels. A secondary Wasserstein
distance analysis measures distributional alignment.

### `h2_wilcoxon_per_class`

```python
h2_wilcoxon_per_class(
    y_hat_cf: np.ndarray,       # (n, n_classes) context-free predictions in [0,1]
    y_hat_ca: np.ndarray,       # (n, n_classes) context-aware predictions in [0,1]
    y_interview: np.ndarray,    # (n, n_classes) interview labels (ground truth)
    class_list: list[str],
    patient_ids: np.ndarray,    # (n,) for block bootstrap
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame
```

**Algorithm:**
1. For each class `c`, compute per-item squared errors:
   - `BS_cf(i,c) = (ŷ_cf(i,c) − y_int(i,c))²`
   - `BS_ca(i,c) = (ŷ_ca(i,c) − y_int(i,c))²`
   - `diff(i,c) = BS_cf(i,c) − BS_ca(i,c)` (positive = context-aware is better)
2. `scipy.stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")` — H₀:
   median improvement = 0.
3. 95% patient-level bootstrap CI on the mean improvement via `bootstrap_scalar()`.
4. BH FDR correction (q = 0.05) applied across all classes simultaneously.
5. If all `diff` values are exactly 0, the Wilcoxon test is skipped and `p = 1.0`.

**Returns** a `pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `Class` | str | Class name |
| `BS_cf` | float | Mean Brier score, context-free model |
| `BS_ca` | float | Mean Brier score, context-aware model |
| `improvement` | float | Mean `BS_cf − BS_ca` (positive = CA better) |
| `ci_lower` | float | 95% bootstrap CI lower bound on mean improvement |
| `ci_upper` | float | 95% bootstrap CI upper bound on mean improvement |
| `wilcoxon_stat` | float | Wilcoxon W statistic |
| `wilcoxon_p` | float | Raw one-sided Wilcoxon p-value |
| `bh_adj_p` | float | BH-adjusted p-value |

**Saved to:** `h2_brier.csv`

---

### `h2_wasserstein_per_class`

```python
h2_wasserstein_per_class(
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    class_list: list[str],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame
```

**Algorithm:**
1. For each class `c`, compute:
   - `W_cf(c) = W₁(ŷ_cf[:, c], y_int[:, c])` — Wasserstein-1 / Earth Mover's Distance
   - `W_ca(c) = W₁(ŷ_ca[:, c], y_int[:, c])`
2. 95% patient-level bootstrap CIs on each distance via `patient_block_bootstrap()`.

A lower Wasserstein distance means the model's prediction distribution is more aligned with
the interview label distribution. This is a secondary/sensitivity analysis;
no hypothesis test or FDR correction is applied.

**Returns** a `pd.DataFrame` with columns:

| Column | Description |
|---|---|
| `Class` | Class name |
| `W_cf` | Wasserstein-1 distance, context-free model |
| `W_ca` | Wasserstein-1 distance, context-aware model |
| `ci_lower_cf` / `ci_upper_cf` | 95% bootstrap CI on `W_cf` |
| `ci_lower_ca` / `ci_upper_ca` | 95% bootstrap CI on `W_ca` |

**Saved to:** `h2_wasserstein.csv`

---

### `h2_macro_summary`

```python
h2_macro_summary(
    wilcoxon_df: pd.DataFrame,  # output of h2_wilcoxon_per_class
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict
```

Computes macro-average Brier scores across all (item, class) pairs with a 95%
patient-level bootstrap CI. The `n_classes_significant` count is derived directly from
`wilcoxon_df["bh_adj_p"] < 0.05`.

**Returns** a `dict` with keys:

| Key | Description |
|---|---|
| `macro_BS_cf` | Overall macro-average Brier score, context-free |
| `macro_BS_ca` | Overall macro-average Brier score, context-aware |
| `macro_improvement` | Mean `BS_cf − BS_ca` across all (item, class) pairs |
| `ci_lower_improvement` | 95% bootstrap CI lower bound |
| `ci_upper_improvement` | 95% bootstrap CI upper bound |
| `n_classes_significant` | Count of classes with BH-adjusted p < 0.05 |
| `n_classes_total` | Total number of classes tested |

**Saved to:** `h2_brier_summary.json`

---

## Dependencies

- `scipy.stats`: `binomtest`, `wilcoxon`, `wasserstein_distance`, `false_discovery_control`
- `statsmodels.stats.contingency_tables`: `StratifiedTable` (optional; manual MH fallback)
- `shared.statistical.bootstrap`: `bootstrap_rate`, `bootstrap_scalar`, `patient_block_bootstrap`
- `shared.statistical.delta`: `sign_agreement_mask`
