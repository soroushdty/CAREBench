# CAREBench Training Shared Utilities

**Canonical path:** `tracks/representation/training/shared`

This directory houses the foundational algorithms and utilities used identically by both Stage 1 (Context-Free) and Stage 2 (Context-Aware) training phases. Every routine here is designed to be safe to call from inside threaded HP-search workers and to operate correctly on the soft, fractional physician-pair labels produced by `shared/reference/aggregation/paired_reference_mean.py`.

For the high-level pipeline flow that consumes these utilities, see the parent [training README](../README.md).

## File Index

| File                  | Purpose                                                                          |
| --------------------- | -------------------------------------------------------------------------------- |
| `soft_label_utils.py` | Soft-label-aware classification metrics (Brier, ROC AUC, PR curve, ECE)          |
| `threshold_tuning.py` | GPU-vectorized per-class decision threshold search with hard constraints         |
| `fit_calibrators.py`  | Per-class isotonic regression fitter with degenerate-fold fallback               |
| `apply_calibrators.py`| Apply fitted per-class calibrators to a probability matrix                       |
| `resolve_thresholds.py`| Resolve per-class thresholds for final reporting from `cfg["primary_threshold"]` |
| `lopo_cv.py`          | Leave-One-Patient-Out cross-validation split generator                           |
| `set_seeds.py`        | Global deterministic-seed setter for `torch`, `numpy`, `random`, hash seed       |

## `soft_label_utils.py`

CAREBench labels are continuous values in `[0, 1]` representing physician-pair agreement fractions (e.g. `0.5` denotes a tied disagreement within a pair). Standard `sklearn.metrics` functions reject non-binary `y_true` arrays, so this module re-implements the metric set natively for soft fractional targets.

### Tie-handling convention

`_soft_roc_auc` and `_soft_pr_curve` both use the standard trapezoidal / Wilcoxon–Mann–Whitney convention: samples with identical predicted probabilities are grouped into a tie-group; within each group, every positive receives credit for the negatives that scored strictly lower plus half of the negatives in the same tie-group. This matches `sklearn`'s behaviour when hard labels happen to be present, and naturally generalizes to soft labels.

### Functions

#### `macro_brier_score(y_true, probs) -> float`
GPU-accelerated macro-averaged Brier score. Used by the inner HP-selection loop in `stage1/hp_search.py`, called hundreds of times per pipeline run, so GPU throughput matters even though each individual call is small. Falls back to CPU if CUDA is unavailable.

#### `masked_class_data(y_true, probs, class_idx, eval_pos_threshold_or_cfg) -> (mask, y_col, probs_col)`
Return the slice of `(y_true, probs)` for one class restricted to samples that are clearly positive (`>= threshold`) or clearly negative (`<= 1 - threshold`). The ambiguous middle band is excluded. The default `threshold = 0.5` includes all samples (`y >= 0.5 | y <= 0.5` is always true). Accepts either a scalar threshold or a config dict with `eval_pos_threshold`.

#### `_soft_confusion_counts(y_true_col, probs_col, threshold) -> (tp, fp, tn, fn)`
Soft confusion-matrix counts. Each entry is a continuous mass (not an integer count) because the labels are fractional.

#### `_soft_scores(y_true_col, probs_col, threshold) -> dict`
Returns `{TP, FP, TN, FN, Precision, Recall, MCC, F1}` from soft confusion counts. MCC's denominator is floored at `1e-8` to avoid division by zero on degenerate folds.

#### `_soft_pr_curve(y_true_col, probs_col) -> (precision, recall)`
Soft Precision–Recall curve via cumulative tie-group aggregation. Returns `([1.0], [0.0])` when total positive mass is zero.

#### `_soft_average_precision(y_true_col, probs_col) -> float`
Step-function area under the soft PR curve (matches `sklearn.metrics.average_precision_score` for binary targets).

#### `_soft_roc_auc(y_true_col, probs_col) -> float`
Wilcoxon–Mann–Whitney U-statistic formulation of AUC. Iterates over ascending tie-groups, crediting positives with `neg_before + 0.5 * neg_group` for each group, then normalizes by `pos_mass * neg_mass`. Returns `0.5` when either class has zero mass.

#### `_soft_roc_curve(y_true_col, probs_col) -> (fpr, tpr)`
Soft ROC curve points, one per ascending tie-group threshold.

#### `masked_macro_f1(y_true, probs, thresholds, eval_pos_threshold_or_cfg) -> float`
Macro-averaged soft F1 across all classes. Used by the Stage 1 early-stopping monitor.

#### `masked_macro_metric(y_true, probs, thresholds, eval_pos_threshold_or_cfg, objective) -> float`
Configurable macro-averaged metric. Supported objectives:

| Objective | Computation                                                  |
| --------- | ------------------------------------------------------------ |
| `F1`      | Soft F1                                                      |
| `F0.5`    | Soft F-beta with β = 0.5                                     |
| `PRECISION` | Soft precision                                             |
| `RECALL`  | Soft recall                                                  |
| `MCC`     | Soft Matthews correlation coefficient                        |
| `AP`      | Soft average precision                                       |
| `AUC ROC` | Soft Wilcoxon AUC                                            |
| `AUC PR`  | Trapezoidal AUC under the soft PR curve                      |
| `YOUDEN`  | TPR − FPR at the supplied threshold                          |

#### `_soft_ece(y_true_col, probs_col, n_bins=15) -> float`
Expected Calibration Error using equal-width bins over `[0, 1]`. Empty bins contribute zero. Returns NaN for empty input or constant predictions.

#### `soft_ece_with_ci(y_true_col, probs_col, patient_ids, n_bins=15, n_resamples=1000, rng=None) -> (ece, ci_lower, ci_upper)`
ECE point estimate plus a 95 % patient-clustered bootstrap CI. Whole patient blocks are resampled (cluster bootstrap) so the CI reflects between-patient variability rather than within-patient observation noise. Falls back to NaN CI bounds when fewer than 2 unique patients are present.

#### `_soft_brier_score(y_true_col, probs_col) -> float`
Single-class mean squared error — no binning required.

#### `soft_brier_with_ci(y_true_col, probs_col, patient_ids, n_resamples=1000, rng=None) -> (brier, ci_lower, ci_upper)`
Brier point estimate plus a 95 % patient-clustered bootstrap CI, structurally identical to `soft_ece_with_ci`.

## `threshold_tuning.py`

Implements per-class decision threshold selection by optimizing a configurable objective with optional hard operational constraints.

### `_resolve_threshold_grid(cfg) -> np.ndarray`
Builds the candidate threshold grid from `cfg`. Either an explicit `threshold_search_grid` or a linspace from `threshold_search_min` (default 0.05) to `threshold_search_max` (default 0.99) with `threshold_search_points` (default 99) points. The grid is clipped to `[threshold_min_value, threshold_max_value]` and de-duplicated.

### `_threshold_tuning_gpu(probs_c, y_bin, grid, beta, min_precision, min_recall, max_pred_pos_rate, min_t, max_t, objective) -> (best_threshold, metrics)`

Vectorized threshold search evaluated in a single GPU broadcast pass. For `N` samples and `G` grid candidates:

1. Build an `(N, G)` boolean prediction matrix `p.unsqueeze(1) >= g.unsqueeze(0)`.
2. Reduce over the sample axis to `(G,)` per-threshold confusion counts.
3. Compute Precision, Recall, F1, F-beta, Youden, and Pred Pos Rate for every grid point in parallel.
4. Mask grid points violating any of the three hard constraints by setting their objective score to `-inf`.
5. If **every** grid point is masked, log a warning and silently fall back to the unconstrained best.
6. Return the argmax threshold and its metric dict.

Falls back to CPU if CUDA is unavailable. Supports objectives `F1`, `F0.5`, `FBETA`, `PRECISION`, `RECALL`, `YOUDEN`.

### `threshold_tuning(probs, Y_val, class_list, cfg, return_report=False) -> np.ndarray | (np.ndarray, pd.DataFrame)`

Public entry point. Iterates over classes, calling `_threshold_tuning_gpu` for each, and returns the per-class threshold vector. When `return_report=True` it also returns a per-class diagnostic DataFrame with columns `Class`, `Valid Count`, `Soft Prevalence`, `Threshold Objective`, `Threshold Beta`, `Selected Threshold`, `Selected Objective Score`, `Precision@Selected`, `Recall@Selected`, `F1@Selected`, `Fbeta@Selected`, `Pred Pos Rate@Selected`.

Classes with no valid samples after masking receive the default threshold `clip(0.5, min_t, max_t)` and NaN metric values in the report.

### Configuration keys consumed

| Key                                          | Default | Effect                                                            |
| -------------------------------------------- | ------- | ----------------------------------------------------------------- |
| `threshold_objective`                        | `F0.5`  | Optimization target                                               |
| `threshold_beta`                             | `0.5`   | β for F-beta computation                                          |
| `threshold_min_precision_floor`              | `None`  | Hard minimum precision; violators are masked                      |
| `threshold_min_recall_floor`                 | `None`  | Hard minimum recall; violators are masked                         |
| `threshold_max_predicted_positive_rate`      | `None`  | Hard ceiling on PPR; violators are masked                         |
| `threshold_min_value` / `threshold_max_value`| `0.0` / `1.0` | Bounds applied to the grid                                  |
| `threshold_search_grid`                      | `None`  | Explicit grid; overrides linspace parameters                      |
| `threshold_search_min/max/points`            | `0.05` / `0.99` / `99` | Linspace grid definition                           |

## `fit_calibrators.py`

Per-class probability calibrator fitter.

### `fit_calibrators(probs, Y_true, class_list, cfg) -> dict`

Returns `{class_name: calibrator}` with one entry per class. Supported `cfg["calibration_method"]` values:

- `"isotonic"` (default) — `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` fitted on the masked validation subset.
- `"none"` — stores `None` per class; downstream `apply_calibrators` will pass scores through unchanged.

The `models.ConstantCalibrator.ConstantCalibrator` fallback fires when any of these conditions hold on the masked subset:

| Condition                                  | Fallback constant       |
| ------------------------------------------ | ----------------------- |
| Empty subset                               | `ConstantCalibrator(0.5)` |
| Single unique label (all-pos or all-neg)   | `ConstantCalibrator(Y_bin.mean())` |
| `Y_bin.size < cfg["calibration_min_samples"]` (default 10) | `ConstantCalibrator(Y_bin.mean())` |

`Y_bin.mean()` is the maximum-likelihood estimate of label prevalence; it produces constant calibrated scores, which is the least-wrong option when the fold provides no calibration signal.

### Configuration keys consumed

| Key                       | Default     | Effect                                            |
| ------------------------- | ----------- | ------------------------------------------------- |
| `calibration_method`      | `"isotonic"`| Fitter selection                                  |
| `calibration_min_samples` | `10`        | Minimum samples before a `ConstantCalibrator` fallback |
| `eval_pos_threshold`      | `0.5`       | Positive/negative cut used by `masked_class_data` |

## `apply_calibrators.py`

### `apply_calibrators(calibrators, probs, class_list) -> np.ndarray`

Applies the fitted per-class calibrators to a `(n_samples, n_classes)` matrix.

The dispatch order per class is:

1. `cal is None` → passthrough.
2. `hasattr(cal, "predict_proba")` → use `cal.predict_proba(p_c)[:, 1]` (the positive-class column).
3. Otherwise → `cal.predict(p_c)` (used by `IsotonicRegression` and `ConstantCalibrator`).

The output is clipped to `[0, 1]`.

> **Important:** the column ordering of `probs` must match `class_list`, which must also match the ordering used at fit time. Mixing up the order produces silently wrong calibrated probabilities — there is no runtime check.

## `resolve_thresholds.py`

### `resolve_thresholds(cfg, class_list, tuned_thresholds=None, logger=logging) -> (np.ndarray, dict)`

Maps `cfg["primary_threshold"]` to the per-class threshold vector used for final metric reporting.

| Mode             | Per-class threshold                                            |
| ---------------- | -------------------------------------------------------------- |
| `"tau"`          | Constant `cfg["tau"]` (default 0.5)                            |
| `"fixed_f1"`     | Constant `cfg["fixed_f1_threshold"]` (default 0.3)             |
| `"inner_f1_opt"` / `"tuned"` | Pass-through of the supplied `tuned_thresholds` |
| anything else    | Constant 0.5 with a warning                                    |

Returns `(thresholds, meta)` where `meta = {"mode", "value"}` is forwarded into the metric metadata sidecar JSON.

## `lopo_cv.py`

### `lopo_splits(patient_ids) -> list[(train_indices, val_indices)]`

For each unique patient ID, emit one fold where that patient's rows form the validation set and all remaining rows form the training set. Folds are ordered by sorted unique patient IDs.

Raises `ValueError` if `patient_ids` contains fewer than 2 unique patients (a meaningful train/val split is impossible) or is not 1-D.

This is the same generator used for both the outer LOPO loop and every inner LOPO loop (Stage 1 HP search, Stage 2 α search, Stage 2 OOF for post-hoc calibration).

## `set_seeds.py`

### `set_seeds(seed, *, deterministic_algorithms=False) -> None`

Globally seeds:

- `numpy.random.seed`
- `random.seed`
- `torch.manual_seed`
- `torch.cuda.manual_seed` and `torch.cuda.manual_seed_all` (when CUDA is available)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `os.environ["PYTHONHASHSEED"]` (only via `setdefault`)

`deterministic_algorithms=True` calls `torch.use_deterministic_algorithms(True)`. **This must only be done once at process startup** — it is not thread-safe and can raise `RuntimeError` if another thread is mid-operation. The orchestrator passes `True` exactly once before the outer fold loop; per-fold and per-thread `set_seeds` calls leave it as `False`.
