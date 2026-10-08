# CAREBench Stage 2 Training

**Canonical path:** `tracks/representation/training/stage2`

This directory houses the logic for training Context-Aware (Stage 2) fusion models. Stage 2 acts as a meta-learner: it takes the Context-Free baseline predictions ŷ_cf produced by Stage 1, alongside contextual patient vectors c_p and item embeddings e_i, and outputs an updated, context-informed prediction ŷ_ca.

For the high-level pipeline architecture and how Stage 2 connects to Stage 1 and the orchestrator, see the parent [training README](../README.md).

## File Index

| File                    | Purpose                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `fit_stage2_fusion.py`  | Stage 2 fusion heads, inner-α LOPO HP search, prevalence-shift regularization, alternative architectures, post-hoc Stage 2 calibration |
| `stage2_context.py`     | Patient context encoding and fusion-feature assembly                                 |

## `fit_stage2_fusion.py`

### Models

#### `Stage2FusionModel(nn.Module)`

```python
ŷ_ca = σ(V^T z + b)
```

A Ridge-regularized linear head where `z` is a fusion vector (length depends on `fusion_strategy`). The forward pass returns **raw logits** (no sigmoid) so that `nn.functional.binary_cross_entropy_with_logits` can be used for numerically stable training with correct `pos_weight` semantics. Callers that need probabilities apply sigmoid explicitly. Weights and bias are initialized to zero.

#### `LowRankBilinearFusionModel(nn.Module)`

Per-class low-rank bilinear head:

```python
ŷ_c = e_i^T (U_c V_c^T) c_p + b_c
```

Input `Z` must be the concatenation `[e_i ‖ c_p]` with shape `(batch, 2d)`. `U_c` and `V_c` are learned per class with shape `(n_classes, d, r)`, giving 2·d·r parameters per class. The forward pass uses `torch.einsum("bd,cdr->bcr", ...)` for efficient batched per-class projection. Initialization: `randn * 0.01` for the factors, zeros for the bias. Validates `input_dim % 2 == 0` at construction.

#### `_PassthroughSentinel`

A trivial stand-in returned by `fit_stage2_fusion_fold` when the Stage 1 passthrough beats every learned fusion candidate on the inner Brier. It exposes `eval()` and `parameters()` so it walks through the same downstream save / restore code paths as a real `nn.Module`. `apply_stage2_fusion` detects it via `isinstance(...)` and short-circuits to `np.clip(y_cf, 0.0, 1.0)`. Its presence is the explicit signal that **H2 is not asserted for that fold** — the data did not support it.

### Loss + regularization

#### `_compute_pos_weights(y, weight_cap, cfg) -> np.ndarray`

Per-class positive-class weights capped to `[1.0, weight_cap]`. Uses Cui 2019 Effective Number of Samples (default) or inverse-frequency weighting, mirroring the Stage 1 logic. Two CAREBench-specific decisions:

- The positive mass is computed as the **soft sum** `valid.sum()` so that physician-disagreement labels (0.5) contribute 0.5 to positive mass rather than being binarized away by a `> 0.5` threshold.
- The inverse-frequency denominator uses the **per-class non-NaN row count**, not the total row count, so classes with many missing interview labels are not artificially upweighted by the dataset size.

#### `_train_one_model(Z_tr, y_int_tr, y_survey_tr, y_cf_tr, alpha, lr, num_epochs, patience, weight_cap, device, cfg, model_cls, model_kwargs, Z_val, y_val) -> nn.Module`

The Stage 2 training loop with three notable behaviours:

1. **Masked normalized BCE** — A boolean mask is built from `~np.isnan(y_int_tr)`. NaN positions are filled with `0.0` for tensor arithmetic and zeroed out by the mask in the loss; the masked sum is normalized by the valid count to keep gradient scale stable across folds regardless of NaN density. Crucially, **survey labels are not substituted into NaN interview cells** — this would conflate the Stage 2 target distribution.
2. **Optional weighted loss** — Defaults to **unweighted** BCE because Stage 1's input ŷ_cf is already calibrated; a `pos_weight` would shift Stage 2 predictions away from the calibrated CF baseline rather than correcting class imbalance. Weighted mode is available via `cfg["stage2_weighted_loss"] = True` for ablation only.
3. **Prevalence-shift regularization** — When `cfg["stage2_delta_reg_weight"] > 0`, an extra term `λ · mean_c((mean_b σ(logits)_c − mean_b ŷ_cf_c)²)` is added to the loss. The `mean_cf` reference is computed once before the training loop (constant). This penalises the head from drifting its global per-class mean away from Stage 1's mean — strongly encouraging patient-specific differential shifts rather than a universally biased global offset.

Early stopping:

- When `Z_val` and `y_val` are provided, the loop monitors held-out per-class Brier and tracks the best state for restoration.
- When they are absent (final retraining on the full outer-train fold), the loop trains for the full `num_epochs` without early stopping; ridge regularization (the `weight_decay=alpha` on the Adam optimizer) handles overfitting.
- `best_state` is only restored when validation monitoring was active. Without validation, restoring `best_state` would reset the model to its initial zero weights — this branch explicitly skips the restore.

#### `_brier_score(y_pred, y_true) -> float`

Mean per-class Brier Score with NaN-target masking. Returns `1.0` (worst possible) when no class has any non-NaN target.

### Per-fold orchestration

#### `fit_stage2_fusion_fold(Z_tr, y_int_tr, y_survey_tr, y_cf_tr, patient_ids_tr, class_list, cfg, baseline_metrics=None, metrics_df=None) -> (model_or_sentinel, calibrators_or_None)`

The main per-outer-fold Stage 2 fitter. Steps:

1. Resolve the model class from `cfg["fusion_strategy"]`: `"lowrank_bilinear"` → `LowRankBilinearFusionModel(r=cfg["lowrank_bilinear_r"])`, otherwise `Stage2FusionModel`.
2. **Inner LOPO α search** over `cfg["stage2_alpha_options"]` (default `[0.01, 0.1, 1.0, 10.0, 100.0]`):
   - For every inner split and every α, train a fresh model with `_train_one_model(..., Z_val, y_val)` and score it on the held-out inner patient via `_brier_score(preds, y_int_val)`.
   - Track per-α `hp_deltas` (mean prediction shift vs ŷ_cf) for diagnostic logging.
   - Score two **baselines** on the same inner folds for comparison: `passthrough` (returns ŷ_cf unchanged) and `item_plus_patient_onehot` (concatenates `[Z ‖ one-hot(patient)]`, fixed α=0.01).
3. **Selection** — pick the α with the lowest mean Brier across inner folds.
4. **Per-class delta diagnostic** — compare model mean delta to physician mean delta per class. Classes where the model shifts in the **opposite direction** to the physician trigger a warning, indicating the model is learning a spurious global bias rather than a patient-specific correction. The warning suggests increasing `stage2_delta_reg_weight`.
5. **Passthrough guardrail** — if the passthrough baseline mean Brier `<=` the best fusion mean Brier, return `_PassthroughSentinel(), None`. **H2 is not supported on this fold** — fall back to Stage 1 unchanged.
6. **OOF for post-hoc calibration** — re-run inner LOPO at `best_alpha` to collect Stage 2 OOF predictions, then fit isotonic per-class calibrators on the OOF when at least `cfg["calibration_min_samples"]` (default 10) valid rows are available.
7. **Final retraining** — train one last model on all `n_tr` items at `best_alpha` (no held-out monitoring; ridge handles regularisation).

Returns `(final_model, s2_calibrators)`. The orchestrator wraps these in a fold dict that also captures the optional Stage 2 PCA, the fusion strategy, the bilinear rank `r`, the `use_cf_passthrough` flag, and the class list.

#### `apply_stage2_fusion(artifact, Z, y_cf=None) -> np.ndarray`

Inference-time application:

1. If `artifact["model"]` is a `_PassthroughSentinel`, return `clip(y_cf, 0, 1)`. Raises if `y_cf` is missing.
2. If `artifact["use_cf_passthrough"]` and `y_cf` is provided, prepend it: `Z = [y_cf ‖ Z]`.
3. Forward `Z` through the model → sigmoid → numpy.
4. If post-hoc Stage 2 calibrators are present, apply them via `shared.apply_calibrators`.
5. Clip to `[0, 1]` and return as `float32`.

#### `project_stage2_embeddings(artifact, X_items, context_vectors) -> (X, ctx)`

Apply the artifact's stored PCA (if any) to item embeddings and to every patient context vector. When the artifact has no PCA, returns the inputs unchanged. Used by the orchestrator at inference time to keep dimensions consistent with the per-fold Stage 2 PCA fitted during training.

### Alternative architecture sweep

#### `_build_patient_id_baseline_features(X_items, patient_ids, reference_ids=None) -> np.ndarray`

Builds `[e_i ‖ one-hot(patient)]` features for the patient-ID baseline. `X_items` must be **raw item embeddings**, not fusion features. The one-hot encoding ranges over `reference_ids` (the training-patient set for this fold). Unseen-patient rows receive an all-zero one-hot block.

#### `fit_all_stage2_architectures_fold(X_items_tr, patient_ids_tr, context_vectors, y_int_tr, y_survey_tr, y_cf_tr, class_list, cfg, r=8) -> dict`

Trains the four alternative fusion architectures under their own independent inner LOPO α searches:

| Architecture        | Z construction                                         | Model                               |
| ------------------- | ------------------------------------------------------ | ----------------------------------- |
| `"2d"`              | `[e_i ‖ c_p]`                                          | `Stage2FusionModel`                 |
| `"3d"`              | `[e_i ‖ c_p ‖ e_i ⊙ c_p]`                              | `Stage2FusionModel`                 |
| `"lowrank_bilinear"`| `[e_i ‖ c_p]`                                          | `LowRankBilinearFusionModel(r=r)`   |
| `"patient_id"`      | `[e_i ‖ one-hot(patient)]`                             | `Stage2FusionModel`                 |

Returns `{arch_name: {"model": fitted_model}}`. The orchestrator forwards each fold's results into `arch_predictions` for the architecture statistical comparison. `passthrough` and `stage1_only` require no training and are assembled directly by the orchestrator from Stage 1 ŷ_cf.

### Configuration keys consumed

| Key                                  | Default                       | Effect                                                       |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| `fusion_strategy`                    | `"2d"` (in-code fallback and shipped [`config/training_config.yaml`](../../../config/training_config.yaml) value) | Choice of fusion vector / model class for the main head      |
| `stage2_alpha_options`               | `[0.01, 0.1, 1.0, 10.0, 100.0]` | Ridge α grid for inner LOPO search                          |
| `stage2_lr`                          | `0.01`                        | Adam learning rate                                           |
| `stage2_num_epochs`                  | `200`                         | Max epochs                                                   |
| `stage2_early_stopping_patience`     | `20`                          | Validation patience (only when `Z_val/y_val` provided)       |
| `weight_cap`                         | `50.0`                        | Per-class positive weight ceiling                            |
| `lowrank_bilinear_r`                 | `8`                           | Rank for `LowRankBilinearFusionModel`                        |
| `stage2_weighted_loss`               | `False`                       | Enable Stage-2 BCE pos_weighting (ablation only)             |
| `stage2_delta_reg_weight`            | `0.0`                         | Prevalence-shift regularization strength λ                   |
| `stage2_use_cf_passthrough`          | `False`                       | Prepend ŷ_cf to Z (also enables fold-pure cross-fitted CF)   |
| `stage2_pca_n_components`            | `None`                        | Optional Stage 2-internal PCA dimensionality                 |
| `class_weight_mode`                  | `"cui"`                       | Cui 2019 ENS or `"inverse_frequency"`                        |
| `class_weight_beta`                  | `0.999`                       | Cui β                                                        |
| `calibration_min_samples`            | `10`                          | Minimum OOF rows for post-hoc Stage 2 calibration            |

## `stage2_context.py`

### Patient context encoding

#### `_CONTEXT_FIELDS` (constant)

```
["summary", "medical_history", "allergies", "medication_history",
 "social_history", "labs", "radiology", "procedures"]
```

The eight sub-fields composing each patient's clinical snapshot.

#### `_CONTEXT_TEMPLATE` (constant)

```
"Summary: {summary}. Medical History: {medical_history}. "
"Allergies: {allergies}. Medication History: {medication_history}. "
"Social History: {social_history}. Labs: {labs}. "
"Radiology: {radiology}. Procedures: {procedures}."
```

The canonical serialization template — every Stage 2 patient string is built from this exact template so that the encoder sees a stable surface form.

#### `build_context_string(patient_ctx) -> str`

Serialize one patient's eight sub-fields into the canonical context string. Missing keys default to the empty string.

#### `build_context_vectors(context_json, model_id, cfg) -> dict`

Encode every patient's context string through the frozen encoder. Calls `shared.embeddings.compute_embeddings.compute_embeddings` with `schema='pandas'` and the config-driven pooling strategy. Returns `{patient_id_int: np.ndarray (d,)}`.

This function is called **once before the outer fold loop** and the result is cached for every subsequent fold's Stage 2 fit. Patient IDs that are not parseable as `int(float(...))` are skipped with a warning rather than crashing the cache build.

#### `ablate_field(ctx_dict, field_name) -> dict`

Return a deep copy of `ctx_dict` with `ctx_dict[field_name]` replaced by `""`.

#### `build_ablated_context_vector(patient_ctx, field_name, model_id, cfg) -> np.ndarray`

Re-encode a single patient's context with one sub-field set to `""`. This is the per-call helper used by the sub-field ablation study to attribute context-induced prediction shifts to individual clinical information sources.

### Fusion-feature assembly

#### `_build_2d_fusion(e_i, c_p) -> np.ndarray`
`np.concatenate([e_i, c_p])` → length `2d`.

#### `_build_3d_fusion(e_i, c_p) -> np.ndarray`
`np.concatenate([e_i, c_p, e_i * c_p])` → length `3d`.

#### `_build_lowrank_bilinear_fusion(e_i, c_p, r=8) -> np.ndarray`
`np.concatenate([e_i, c_p])` → length `2d`. Random projections are **not** used here because `LowRankBilinearFusionModel` requires learned (not fixed) per-class factors `U_c, V_c`.

#### `build_fusion_features(e_i, c_p, fusion_strategy="2d", r=8) -> np.ndarray`

Dispatcher returning one of the strategies below. `"2d"` is both the function-signature default and the catch-all: any unrecognized strategy string also routes to `_build_2d_fusion`.

| `fusion_strategy`     | Output length |
| --------------------- | ------------- |
| `"2d"` (default)      | `2 d`         |
| `"3d"`                | `3 d`         |
| `"4_vector"`          | `4 d`         |
| `"lowrank_bilinear"`  | `2 d`         |

The same `"2d"` default is set in [`config/training_config.yaml`](../../../config/training_config.yaml), so the in-code fallback and the shipped config agree.

#### `build_fusion_matrix(X_items, patient_ids, context_vectors, fusion_strategy="2d", r=8) -> np.ndarray`

Build the full `(n_items, fusion_dim)` matrix row by row. For each row, look up `context_vectors[int(patient_ids[i])]`; if absent, substitute zeros and emit a one-time warning per missing patient (so we don't flood the log). If a context vector's dimension does not match the item embedding dimension, raise `ValueError`. The output is cast to `float32`.

> The dimension match check exists because a mismatch usually indicates the context vectors and item embeddings were produced by different encoders or pooling strategies — both must use the same frozen encoder for the fusion to be meaningful.
