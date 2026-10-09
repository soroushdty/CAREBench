# Reporting (`tracks/representation/statistical/reporting`)

Canonical location: `tracks/representation/statistical/reporting/`

This package generates the comparison tables, output figures, and stratum sub-analysis.
All functions consume pre-computed prediction
arrays or structured results from the metrics/hypotheses layers; they do not run
statistical tests themselves (with the exception of Mann-Whitney U in `stratum.py`).

---

## `arch_compare.py` — Architecture comparison table

### `arch_comparison_table`

```python
arch_comparison_table(
    arch_predictions: dict[str, np.ndarray],  # {arch_name: (n, n_classes) preds in [0,1]}
    y_interview: np.ndarray,                  # (n, n_classes) interview labels (ground truth)
    class_list: list[str],
    patient_ids: np.ndarray,                  # (n,) for cluster bootstrap
    thresholds: np.ndarray,                   # (n_classes,) decision thresholds
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    reference_arch: str = "2d",
) -> pd.DataFrame
```

**Algorithm:**

1. For each architecture in `arch_predictions`, compute:
   - **Macro-Brier**: `mean((y_pred − y_true)²)` over all (item, class) pairs
   - **Macro-F1**: per-class `_soft_scores` at the provided threshold, then macro average
   - 95% patient-level bootstrap CI on both metrics
2. Compute **paired bootstrap differences** against the reference architecture:
   - `Δ_brier = brier(arch) − brier(reference)` (positive = arch is worse)
   - `Δ_F1 = F1(arch) − F1(reference)` (positive = arch is better)
   - 95% bootstrap CIs on both differences, using the same patient resampling for both
     architectures within each replicate (preserving pairing)

If the reference architecture is not in `arch_predictions`, paired difference columns are
filled with NaN and a warning is logged.

**Returns** a `pd.DataFrame` with columns:

| Column                                          | Description                                                 |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `architecture`                                  | Architecture name key                                       |
| `param_count`                                   | Human-readable parameter count annotation (see table below) |
| `macro_brier`                                   | Macro-average Brier score                                   |
| `brier_ci_lower` / `brier_ci_upper`             | 95% bootstrap CI                                            |
| `macro_f1`                                      | Macro-average F1                                            |
| `f1_ci_lower` / `f1_ci_upper`                   | 95% bootstrap CI                                            |
| `delta_brier_vs_ref`                            | Brier difference vs reference (positive = worse)            |
| `delta_brier_ci_lower` / `delta_brier_ci_upper` | 95% CI on Brier difference                                  |
| `delta_f1_vs_ref`                               | F1 difference vs reference (positive = better)              |
| `delta_f1_ci_lower` / `delta_f1_ci_upper`       | 95% CI on F1 difference                                     |

**Architecture parameter counts** (at `d = 768`):

| Key                | Description                                        | Param count          |
| ------------------ | -------------------------------------------------- | -------------------- |
| `4_vector`         | `[e_i, c_p, e_i * c_p, \|e_i − c_p\|]` (reference) | 30,720 / class       |
| `2d`               | `[e_i, c_p]`                                       | 1,536 / class        |
| `3d`               | `[e_i, c_p, e_i * c_p]`                            | 2,304 / class        |
| `lowrank_bilinear` | `[e_i, c_p]` with LowRankBilinear head             | 12,288 / class (r=8) |
| `passthrough`      | Context concatenated but no learned parameters     | 0                    |
| `patient_id`       | Patient ID embedding lookup                        | d + n_patients       |
| `stage1_only`      | Stage 1 predictions, no Stage 2                    | 0                    |

**Saved to:** `arch_comparison.csv`

---

## `figures.py` — Output figures 1–4

All figures use `matplotlib` with `Agg` backend (non-interactive, safe for headless runs).
Each function saves its output to the provided `output_path` with
`fig.savefig(..., bbox_inches="tight")` and closes the figure to free memory. The parent
directory is created automatically if it does not exist.

### `figure1_delta_histogram`

```python
figure1_delta_histogram(
    delta_p: np.ndarray,     # (n, n_classes) physician delta matrix
    class_list: list[str],   # class names, length n_classes
    output_path: Path,
    dpi: int = 150,
) -> None
```

**What it shows (Figure 1):** A grid of per-class bar charts, five per row, showing the
distribution of physician delta values across all items. Bars are color-coded by delta
magnitude: dark red (−1.0), orange (−0.5), blue-gray (0.0), green (+0.5), dark green
(+1.0). Each panel is annotated with `n_nz` (the count of non-zero deltas), which is the
effective sample size for H1 per-class tests.

**Saved to:** `figures/figure1_delta_histogram.png`

---

### `figure2_interphysician_agreement`

```python
figure2_interphysician_agreement(
    individual_labels: dict,   # from icc.load_individual_physician_labels()
    class_list: list[str],
    output_path: Path,
    dpi: int = 150,
) -> None
```

**What it shows (Figure 2):** Per-patient grouped bar chart comparing the
inter-physician agreement rate (fraction of (item, class) pairs where both physicians gave
identical binary labels) in the survey phase (context-free) vs. the interview phase
(context-aware). Patients where the interview agreement is lower than the survey agreement
are flagged with a red star marker. Agreement rate is computed over all items where both
physicians provided a label for the same item.

Requires `individual_labels` to contain both `"test"` and `"interview"` splits. If
`individual_labels` is empty, the function logs a warning and returns without saving.

**Saved to:** `figures/figure2_interphysician_agreement.png`

---

### `figure3_icc_horizontal_plot`

```python
figure3_icc_horizontal_plot(
    icc_results: dict,     # from icc.rater_icc_analysis()
    output_path: Path,
    dpi: int = 150,
) -> None
```

**What it shows (Figure 3):** A horizontal dot plot with two sets of points:

- **Gray/steelblue points** (bottom, y < 0): One within-pair human-human ICC(2,1)
  per patient, sorted ascending. A shaded band shows ± 1 SD around the human-human mean.
- **Orange points** (top, y > 0): One model-vs-physician ICC(2,1) per (patient, physician), labeled by
  `P{patient}-Ph{physician}`.

The shaded band provides the human-human benchmark. The vertical dotted line at `ICC = 0`
anchors the scale.

Requires `icc_results` to contain both `"human_human_iccs"` and `"model_physician_iccs"`.

**Saved to:** `figures/figure3_icc_horizontal_plot.png`

---

### `figure4_entropy_scatter`

```python
figure4_entropy_scatter(
    phys_entropy_delta: np.ndarray,    # (n, n_classes) physician entropy change
    model_entropy_delta: np.ndarray,   # (n, n_classes) model entropy change
    eligible_classes: list[str],       # classes to include (pooled)
    class_list: list[str],
    pearson_r: float,                  # pre-computed from entropy.entropy_pearson_r()
    output_path: Path,
    dpi: int = 150,
) -> None
```

**What it shows (Figure 4):** A scatter plot where each point is one (item, class)
observation pooled across confirmatory-eligible classes. The x-axis is the physician
entropy change `H(y_interview) − H(y_survey)` and the y-axis is the model entropy change
`H(ŷ_ca) − H(ŷ_cf)`. Points are colored by class (tab10 colormap). Reference lines are
drawn at x = 0, y = 0, and y = x (perfect tracking). The Pearson r is annotated as
"descriptive only" — it is not a confirmatory inference.

Both axes are clamped to `[−1.15, 1.15]` (the theoretical entropy-change range is
`[−1, 1]` for binary labels in {0, 0.5, 1}).

**Saved to:** `figures/figure4_entropy_scatter.png`

---

## `stratum.py` — Repeated vs novel sub-analysis

### `assign_test_strata`

```python
assign_test_strata(
    item_texts_test: np.ndarray,    # (n_test,) item strings for the paired eval items
    item_texts_train: np.ndarray,   # (n_train,) item strings from the training set
) -> np.ndarray                     # (n_test,) str: 'repeated' or 'novel'
```

Assigns each test item to `"repeated"` if its normalized text appeared anywhere in the
training set (under any patient), or `"novel"` otherwise. Normalization uses
`shared.utils.text_utils.normalize_for_matching()` — the same function used in the
fold-construction loop — to ensure consistent text matching across pipeline stages.

The `"repeated"` stratum addresses the memorization concern: items seen during Stage 1
training under a different patient may be handled by the model through pattern recall
rather than genuine patient-context integration.

### `stratum_comparison`

```python
stratum_comparison(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    strata: np.ndarray,               # (n,) 'repeated' or 'novel' per item
    patient_ids: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame
```

**Per-stratum metrics:**

- **Sign agreement rate:** For each item, count the fraction of confirmatory-eligible
  class positions where `sign(Δ_m) == sign(Δ_p)` and `Δ_p ≠ 0`. Average over items
  with at least one non-zero eligible-class physician delta. 95% patient-level bootstrap CI.
- **Brier improvement:** `mean(BS_cf − BS_ca)` per item, averaged over all classes.
  95% patient-level bootstrap CI.

**Between-stratum tests:** Mann-Whitney U (two-sided) on the per-item sign agreement rate
values and per-item Brier improvement values between the repeated and novel strata.

**Returns** a `pd.DataFrame` with columns:

| Column                          | Description                                          |
| ------------------------------- | ---------------------------------------------------- |
| `stratum`                       | `"repeated"` or `"novel"`                            |
| `n_items`                       | Total items in this stratum                          |
| `sign_agree_rate`               | Mean per-item sign agreement rate (eligible classes) |
| `ci_lower_sar` / `ci_upper_sar` | 95% bootstrap CI                                     |
| `brier_improvement`             | Mean per-item Brier improvement (all classes)        |
| `ci_lower_bi` / `ci_upper_bi`   | 95% bootstrap CI                                     |
| `mannwhitney_sar_U`             | Mann-Whitney U statistic for sign agreement rate     |
| `mannwhitney_sar_p`             | Two-sided p-value for sign agreement rate            |
| `mannwhitney_bi_U`              | Mann-Whitney U statistic for Brier improvement       |
| `mannwhitney_bi_p`              | Two-sided p-value for Brier improvement              |

**Saved to:** `stratum_analysis.csv`

---

## Dependencies

- `matplotlib` (all figures — `Agg` backend, does not require a display)
- `scipy.stats`: `mannwhitneyu` (stratum comparison)
- `shared.utils.text_utils`: `normalize_for_matching` (stratum assignment)
- `tracks.representation.training.shared.soft_label_utils`: `_soft_scores` (arch comparison F1)
- `shared.statistical.bootstrap`: `patient_block_bootstrap`, `bootstrap_scalar`
- `shared.statistical.delta`: `sign_agreement_mask`
