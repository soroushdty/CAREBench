# tracks/representation/training/

**Canonical path:** `tracks/representation/training/`

End-to-end training stack for the representation learning (Track 1) pipeline: nested Leave-One-Patient-Out cross-validation (LOPO-CV), a context-free Stage 1 classifier, an optional context-aware Stage 2 fusion head, and the soft-label metric and calibration utilities both stages depend on.

The package re-exports a single public entry point, `train_ensemble_pipeline`, via `tracks/representation/training/__init__.py` (lazy-imported to keep import time low).

## Layout

```
tracks/representation/training/
├── orchestrator/   # LOPO-CV control flow, metric aggregation, GPU monitor, figures
├── stage1/         # Context-free classifier head, HP search, weighted BCE loss
├── stage2/         # Context-aware fusion heads and per-context-entity encoding (was: per-patient)
└── shared/         # Soft-label metrics, threshold tuning, calibrators, LOPO splits, seeds
```

## Pipeline Architecture

The pipeline runs under nested Leave-One-Patient-Out cross-validation (LOPO-CV) with one outer fold per unique context entity (was: patient). Within each outer fold, an inner LOPO sweep selects hyperparameters before final retraining.

```
                          context_entity_ids_train  (was: patient_ids_train)
                                  │
                                  ▼
                   ┌── outer LOPO ──┐  (one fold per context entity)
                   │                │
                   ▼                ▼
            train (N-1 entities)   val (1 entity)
                   │
                   ├── inner LOPO ──→ Stage 1 HP search (parallel threads)
                   │                  ▼
                   │       inner-OOF Stage 1 predictions (ŷ_cf)
                   │                  │
                   │       fit_calibrators ──→ isotonic per class
                   │                  │
                   │       threshold_tuning ──→ per-class τ from inner OOF
                   │                  │
                   ▼                  ▼
            final Stage 1 retrain on outer-train fold
                   │
                   ▼
            Stage 1 predictions on outer-val + on test items
                   │
                   ▼
       ┌── if Stage 2 enabled ──┐
       │                        │
       ▼                        ▼
 build_fusion_matrix       fit_stage2_fusion_fold
 ([e_i ‖ c_p ‖ ...])      (inner LOPO α search + retrain)
       │                        │
       └────────────┬───────────┘
                    ▼
             Stage 2 ŷ_ca
                    │
       ┌────────────┴───────────────┐
       ▼                            ▼
  fold-pure metrics             transductive metrics
  (clean held-out)              (diagnostic only)
```

The orchestrator persists a per-fold checkpoint after each outer fold, so a crashed run can be resumed without recomputing completed folds. See [`orchestrator/README.md`](orchestrator/README.md) for the resume protocol and checkpoint schema.

## Stage 1 — Context-Free Classification

Stage 1 trains a Multi-Layer Perceptron (or pure Logistic Regression head when `hidden_dims=[]`) on PCA-projected item embeddings produced by a frozen pretrained encoder.

Key behaviours:
- `torch.amp` Automatic Mixed Precision and `torch.compile` (`reduce-overhead` mode) for maximum throughput, with eager fallback when unavailable.
- Dynamic mini-batch subsetting: when the configured batch size exceeds the dataset size the trainer collapses to a single full-batch update per epoch, saving kernel launches on small inner folds.
- Numerically stable weighted Binary Cross Entropy via `F.binary_cross_entropy_with_logits`. Per-class positive weights follow either Cui's Effective Number of Samples (default) or inverse-frequency weighting, capped to a configurable ceiling.
- Inner-fold HP search runs each candidate on its own background thread with a unique seed offset to avoid global RNG collisions; the winning HP is selected by the lowest mean inner Brier score.
- Early stopping monitors a configurable validation objective (Macro F1, Brier, F0.5, AUC ROC, AUC PR, Youden) computed against soft physician-pair labels.

See [`stage1/README.md`](stage1/README.md) for full details.

## Stage 2 — Context-Aware Fusion

Stage 2 is a meta-learner: it takes Stage 1's calibrated context-free predictions ŷ_cf alongside item embeddings e_i and patient-context vectors c_p, and outputs context-informed predictions ŷ_ca.

Heads:
- **`Stage2FusionModel`** — Ridge-regularized linear layer mapping a configurable fusion vector Z directly to logits.
- **`LowRankBilinearFusionModel`** — Per-class low-rank bilinear head ŷ_c = e_i^T (U_c V_c^T) c_p + b_c with 2·d·r parameters per class.

Fusion strategies (`stage2_context.build_fusion_matrix`):
- `"2d"` (default both in [`configs/training_config.yaml`](../../../configs/training_config.yaml) and as the in-code fallback when the key is missing) — `[e_i ‖ c_p]`
- `"3d"` — `[e_i ‖ c_p ‖ e_i ⊙ c_p]`
- `"4_vector"` — `[e_i ‖ c_p ‖ e_i ⊙ c_p ‖ |e_i − c_p|]`
- `"lowrank_bilinear"` — `[e_i ‖ c_p]` with per-class learned bilinear factors

Regularization:
- Ridge α selected by an inner LOPO sweep over `cfg["stage2_alpha_options"]`, scored by mean per-class Brier on the inner held-out patient.
- Optional **prevalence-shift regularization** (`stage2_delta_reg_weight`): an L2 penalty on the global mean of Stage 2 predictions vs the Stage 1 mean, encouraging patient-specific differential shifts rather than a universally biased global offset.
- A passthrough sentinel is returned automatically when no learned fusion head beats the Stage 1 passthrough on the inner Brier — `brier_improvement` (formerly H2) is not asserted when the data does not support it.

See [`stage2/README.md`](stage2/README.md) for full details, including the alternative-architecture sweep.

## Shared Utilities

The `shared/` submodule houses the foundational pieces used identically by both Stage 1 and Stage 2.

- **Soft-label metrics** — Because LM-ContextProbe labels are pair-aggregated reference observer (was: physician) fractions (e.g. `0.5` for tied disagreement), `sklearn.metrics` will reject the input. `soft_label_utils.py` re-implements the metric set natively for soft fractional targets, including macro Brier, Wilcoxon–Mann–Whitney AUC (with proper tie-group handling), soft PR curve, soft ECE with cluster-bootstrap CIs, and Brier with cluster-bootstrap CIs.
- **GPU-vectorized threshold search** — `threshold_tuning.py` evaluates an entire grid of candidate thresholds in a single broadcast pass, with hard constraint masking (`min_precision`, `min_recall`, `max_pred_pos_rate`) and silent fallback to the unconstrained best when the constraint is mathematically unsatisfiable.
- **Isotonic calibration** — `fit_calibrators.py` and `apply_calibrators.py` implement per-class 1-D isotonic regression with a `ConstantCalibrator` fallback for degenerate folds (empty subset, single unique label, or below `calibration_min_samples`).
- **Patient-grouped splits** — the outer and inner loops take their index pairs from `shared.cv.patient_splits`, configured by `cv.outer` and `cv.inner` (leave-one-patient-out by default, or grouped k-fold). Every split holds out whole patients.
- **Reproducibility** — `set_seeds.set_seeds` globally seeds `torch`, `numpy`, `random`, and `PYTHONHASHSEED`, with optional `torch.use_deterministic_algorithms(True)` (process startup only — not thread-safe).

See [`shared/README.md`](shared/README.md) for full details.

## Public API

`tracks/representation/training/__init__.py` re-exports a single symbol:

```python
from tracks.representation.training import train_ensemble_pipeline
```

This is the function called by `main.py` to drive the entire LOPO-CV training run. All other functions are submodule-internal and should be imported explicitly from their submodule paths if needed.

## Key Configuration Keys

The pipeline reads its config from `cfg`, typically loaded from [`configs/training_config.yaml`](../../../configs/training_config.yaml). The most relevant keys per submodule are:

| Group         | Keys                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| Stage 1       | `hidden_dims`, `dropout`, `activation`, `lr`, `weight_decay`, `batch_size`, `num_epochs`, `early_stopping_patience`, `objective`, `use_amp`, `use_torch_compile`, `pca_n_components`, `pca_whiten`, `class_weight_mode`, `class_weight_beta`, `weight_cap` |
| Stage 2       | `stage2_alpha_options`, `stage2_lr`, `stage2_num_epochs`, `stage2_early_stopping_patience`, `fusion_strategy`, `lowrank_bilinear_r`, `stage2_delta_reg_weight`, `stage2_use_cf_passthrough`, `stage2_pca_n_components`, `stage2_weighted_loss` |
| Calibration   | `calibration_method` (`"isotonic"` / `"none"`), `calibration_min_samples`                                    |
| Thresholds    | `primary_threshold` (`"tau"` / `"fixed_f1"` / `"inner_f1_opt"` / `"tuned"`), `tau`, `fixed_f1_threshold`, `threshold_objective`, `threshold_beta`, `threshold_min_precision_floor`, `threshold_min_recall_floor`, `threshold_max_predicted_positive_rate`, `threshold_search_grid`, `threshold_search_min`, `threshold_search_max`, `threshold_search_points`, `threshold_min_value`, `threshold_max_value` |
| Metrics       | `eval_pos_threshold`, `ece_n_bins`, `ece_n_resamples`                                                        |
| Reproducibility | `global_seed`                                                                                              |
| GPU monitor   | `GPU_MONITOR.enabled`, `GPU_MONITOR.poll_interval_s`, `GPU_MONITOR.active_util_threshold_pct`, `GPU_MONITOR.upgrade_sm_threshold_pct`, `GPU_MONITOR.save_chart` |
| HP search     | `n_hp_workers`, `log_hp_search_grid`                                                                         |

Refer to each submodule README for the full set of keys actually consumed by that submodule.
