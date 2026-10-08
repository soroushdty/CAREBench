# CAREBench Stage 1 Training

**Canonical path:** `tracks/representation/training/stage1`

This directory contains the logic for training the Context-Free (Stage 1) classification models. These models operate strictly on the raw item text embeddings produced by a frozen pretrained encoder (Bio-ClinicalBERT by default), establishing the baseline un-contextualized predictions ŷ_cf that Stage 2 must improve upon to support H2.

For the high-level pipeline architecture and how Stage 1 connects to Stage 2 and the orchestrator, see the parent [training README](../README.md).

## File Index

| File                    | Purpose                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `train_single_model.py` | Train one `MultiLabelModel` (MLP / Logistic Regression head) with early stopping     |
| `hp_search.py`          | Inner-LOPO hyperparameter candidate evaluator — designed for parallel ThreadPoolExecutor execution |
| `compute_bce_loss.py`   | Numerically stable weighted Binary Cross Entropy with Logits                         |

## `train_single_model.py`

### Signature

```python
def train_single_model(
    X_train, Y_train, X_val, Y_val, cfg, hp, pos_weight_vector,
    *,
    seed=None,
    log_context: str = "",
    epoch_curve_callback=None,
) -> (model, final_probs, best_thresholds)
```

### Inputs

- **`X_train, Y_train`** — Training tensors as numpy `float32` (auto-coerced via `shared.utils.array_utils.as_float32_array`).
- **`X_val, Y_val`** — Validation tensors.
- **`cfg`** — Pipeline config dict. Reads `global_seed`, `hidden_dims`, `dropout`, `activation`, `use_torch_compile`, `num_epochs` (or legacy `epochs`), `early_stopping_patience`, `objective`, `use_amp`, `log_epoch_interval`, `primary_threshold`, `tau`, `fixed_f1_threshold` (and the deprecated `f1_sweep`), and every key `threshold_tuning` consumes.
- **`hp`** — HP candidate dict with `lr`, `weight_decay`, `batch_size`, `weight_cap`.
- **`pos_weight_vector`** — Per-class positive weights (numpy `float32`, shape `(n_classes,)`).
- **`seed`** — Optional integer seed. Overrides `cfg['global_seed']` when provided. **Pass unique values per thread** so that concurrent inner-HP-search workers don't race on global RNG state.
- **`log_context`** — Optional string prepended to DEBUG epoch log lines (e.g. `"Fold 3 / 12 final"`). Inner HP search calls leave this unset.
- **`epoch_curve_callback`** — Optional callable `(epoch, loss, val_score, patience, is_best) -> None` invoked after every epoch. The caller owns all I/O so this function stays filesystem-free. Inner HP search calls leave this `None`.

### Returns

- **`model`** — Trained `MultiLabelModel` loaded with the best-scoring epoch's state dict (re-loaded post-training so the returned model is the early-stopping winner, not the final epoch).
- **`final_probs`** — `(n_val, n_classes)` post-best-state validation probabilities (sigmoid applied, computed under `torch.no_grad()` and the same AMP autocast as training).
- **`best_thresholds`** — `(n_classes,)` per-class thresholds at the best epoch (matches the `primary_threshold` mode used during early-stopping monitoring).

### Architecture choice

The model is built directly from `cfg`:

```python
model = MultiLabelModel(
    n_features  = X_train.shape[1],
    n_classes   = Y_train.shape[1],
    hidden_dims = list(cfg.get("hidden_dims", [])),
    dropout     = float(cfg.get("dropout", 0.3)),
    activation  = str(cfg.get("activation", "gelu")),
).to(DEVICE)
```

`hidden_dims=[]` collapses the head to pure logistic regression (a single linear layer; the `linear.weight` key is preserved for backward compatibility with older checkpoints). `hidden_dims=[64]` produces a one-hidden-layer MLP, etc.

### Throughput optimizations

- **`torch.compile`** — When `cfg["use_torch_compile"]` is `True` and CUDA is available, the model is compiled with `mode="reduce-overhead"`, `fullgraph=False`. This fuses forward + loss into fewer CUDA kernel launches. Compilation failures (CPU, Windows, PyTorch < 2) fall back silently to eager mode with a warning.
- **`torch.amp` Automatic Mixed Precision** — When `cfg["use_amp"]` is `True` and CUDA is available, both forward + loss and the validation forward pass run inside `torch.autocast("cuda")`, with a `torch.amp.GradScaler` for the backward pass. The scaler is the modern `torch.amp.GradScaler(device="cuda", ...)` — the deprecated `torch.cuda.amp.GradScaler` API is **not** used.
- **Dynamic mini-batch subsetting** — `effective_batch = min(hp["batch_size"], X_train.shape[0])`. When `effective_batch == X_train.shape[0]`, the trainer drops the inner permutation loop entirely and runs a single full-batch update per epoch. This eliminates per-batch kernel launch overhead on the small inner folds typical of LOPO-CV.
- **Thread-local RNG** — A per-call `torch.Generator` (on the appropriate device), `np.random.default_rng`, and `random.Random` are seeded from `effective_seed` so that concurrent threaded callers cannot race on `torch.randperm`'s global state.

### Early stopping monitor

Early stopping is governed by `cfg["objective"]` (default `"F1"`) computed via `shared.soft_label_utils.masked_macro_metric` on a per-epoch basis. Available objectives: `F1`, `F0.5`, `PRECISION`, `RECALL`, `MCC`, `AP`, `AUC ROC`, `AUC PR`, `YOUDEN`.

The threshold vector used for the monitor is determined by `cfg["primary_threshold"]`:

| Mode             | Effect                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------- |
| `"tau"`          | Constant `cfg["tau"]` (default 0.5) for every class                                     |
| `"fixed_f1"`     | Constant `cfg["fixed_f1_threshold"]` (default 0.3) for every class                      |
| `"f1_sweep"`     | **Deprecated** — emits a `DeprecationWarning`. Use `"fixed_f1"` instead.                |
| any other        | Per-epoch `threshold_tuning(...)` output (the GPU-vectorized search from `shared/`)     |

`patience` is incremented every epoch the validation score does not strictly exceed `best_score`. Training halts when `patience >= cfg["early_stopping_patience"]` (default 5, must be positive).

`cfg["num_epochs"]` (or legacy `cfg["epochs"]`) caps the total epoch budget (default 50, must be positive).

### Failure modes

- `num_epochs <= 0` or `early_stopping_patience <= 0` → `ValueError`.
- `val_score` is NaN every epoch → `best_state` is never updated; the trainer logs a warning and falls back to the final epoch's state dict so the caller still gets a usable model.

## `hp_search.py`

### `_run_hp_candidate(hp_t, X_tr, Y_tr, inner_splits, cfg, seed_base) -> (hp, mean_score, inner_oof_preds)`

Evaluates one HP candidate across all inner LOPO folds. Designed to run on a `ThreadPoolExecutor` worker (the orchestrator submits up to `cfg["n_hp_workers"]` candidates concurrently).

Inputs:

- **`hp_t`** — 1-tuple of `(head_config,)` where `head_config` is the `hidden_dims` list to evaluate.
- **`X_tr, Y_tr`** — Outer-fold training data (numpy `float32`).
- **`inner_splits`** — `[(train_ix, val_ix), ...]` from `shared.lopo_cv.lopo_splits(patient_ids_tr)`.
- **`cfg`** — Full pipeline config dict.
- **`seed_base`** — Unique integer per HP candidate (the orchestrator passes `(hp_idx + 1) * 1000`); added to `cfg["global_seed"]`. Distinct candidates therefore use disjoint seed ranges, and within a candidate each inner split offsets by `split_idx * 10` so fold seeds are also disjoint.

Construction of the candidate's `hp` dict:

```python
hp = {
    "lr":           cfg.get("lr", cfg.get("lr_options", [0.001])[0]),
    "weight_decay": cfg.get("weight_decay", 0.0001),
    "batch_size":   cfg.get("batch_size", 2048),
    "weight_cap":   cfg.get("weight_cap", cfg.get("weight_cap_options", [10.0])[0]),
    "selected_stage1_head_config": head_config,
}
```

Note that only `head_config` (the MLP topology) varies across candidates today; `lr`, `weight_decay`, `batch_size`, and `weight_cap` are all read from the same scalar config keys regardless of which candidate is being evaluated.

Per inner split:

1. Fit a `Preprocessor` (PCA, `cfg["pca_n_components"]` default 200, `cfg["pca_whiten"]` default `True`) on `xt`.
2. Compute per-class positive weights using either Cui 2019 ENS (default, controlled by `cfg["class_weight_mode"] == "cui"` and `cfg["class_weight_beta"]` default 0.999) or `inverse_frequency`. Both are clipped to `[1.0, hp["weight_cap"]]`.
3. Call `train_single_model(xt_p, yt, xv_p, yv, inner_cfg, hp, pos_w, seed=worker_seed)` where `inner_cfg = {**cfg, "hidden_dims": head_config}`.
4. Capture the inner-OOF predictions into the corresponding rows of `inner_oof_preds`.
5. Score the inner fold via `macro_brier_score(yv, val_p)` (GPU-accelerated soft Brier).

Returns:

- **`hp`** — The candidate dict (with `selected_stage1_head_config`).
- **`mean_score`** — Mean soft Brier across inner folds. Lower is better.
- **`inner_oof_preds`** — `(Y_tr.shape[0], Y_tr.shape[1])` array containing the inner-OOF Stage 1 predictions over all outer-training rows. The orchestrator uses these to derive `thresh_inner` and `fold_calibrators` for the winning candidate (leak-free because inner OOF never touches the held-out outer patient).

### Selection logic on the orchestrator side

The orchestrator runs the candidates in a `ThreadPoolExecutor(max_workers=n_hp_workers)`, collects all `(hp_candidate, avg_s, inner_oof)` results, and picks `argmin(avg_s)`. The winning candidate's `inner_oof` becomes:

- the input to `threshold_tuning(..., return_report=False)` to derive that fold's `thresh_inner`, and
- the input to `fit_calibrators(...)` to fit the per-class isotonic calibrators applied at inference time (and during fold postprocessing).

When `cfg["log_hp_search_grid"]` is `True`, the orchestrator persists the full grid scan to `fold_{i+1}_hp_search.csv`. On crash recovery, when this CSV is present but the fold checkpoint is not, the orchestrator skips re-running the search and reloads the best `selected_stage1_head_config` directly from the CSV; in that path Stage 2 Ridge fitting is skipped for the fold because the inner OOF is no longer available.

## `compute_bce_loss.py`

### `compute_weighted_bce_loss(logits, targets, pos_weight) -> torch.Tensor`

Thin wrapper around `torch.nn.functional.binary_cross_entropy_with_logits` with the `pos_weight` keyword forwarded:

```python
return F.binary_cross_entropy_with_logits(
    logits, targets, pos_weight=pos_weight
)
```

Using the functional API directly (rather than allocating an `nn.BCEWithLogitsLoss` module) eliminates Python object allocation and GC pressure in the training hot loop, which matters across the cumulative thousands of inner-HP × inner-LOPO × epoch loss evaluations per pipeline run. The function provides numerically stable weighted Binary Cross Entropy because `*_with_logits` fuses the sigmoid + BCE in a single log-sum-exp pass.

`pos_weight` is the per-class scalar from `_compute_pos_weights` (Cui 2019 ENS or inverse-frequency) capped at `hp["weight_cap"]`. The exact computation lives in the orchestrator's per-fold setup; this file is intentionally minimal so the loss surface is trivially auditable.
