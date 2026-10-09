# shared/cv/

**Canonical path:** `shared/cv/`

Patient-grouped cross-validation splits, shared by every track that trains on some patients and predicts for others.

Every split holds out whole patients: a patient's rows are all in the validation set or all in the training set, so a held-out patient's labels never reach the model.

## Contents

- `patient_splits.py`
  - `CVScheme(scheme, n_splits, seed)`: one loop's split scheme.
  - `resolve_cv(cfg, loop)`: reads `cfg["cv"][loop]` for `loop` = `outer` or `inner`.
  - `patient_splits(patient_ids, cv)`: returns the `(train_indices, val_indices)` pairs.
  - `lopo_splits(patient_ids)`: shorthand for leave-one-patient-out.
  - `held_out_patients(patient_ids, splits)`: the patients each fold holds out.
  - `patient_to_fold(patient_ids, splits)`: maps each patient to the fold that held it out. Fold-pure scoring uses this to score a test row with the fold whose model never saw that row's patient.

## Schemes

| `scheme` | Folds | `n_splits` | `seed` |
|----------|-------|------------|--------|
| `lopo` (default) | One per patient, in sorted patient-ID order | must be null | must be null |
| `group_kfold` | Patients are shuffled with `seed` and dealt into `n_splits` folds whose patient counts differ by at most one. If `n_splits` is at least the number of patients, each fold holds out one patient. | integer ≥ 2 | integer; null means `global_seed` |

Which patients share a fold depends only on the set of patient IDs and the seed, not on row order. Validation and training indices are sorted.

## Config

```yaml
cv:
  outer: {scheme: lopo, n_splits: null, seed: null}
  inner: {scheme: lopo, n_splits: null, seed: null}
```

If the `cv` block, a loop, or a key is missing, that loop uses `lopo`. Unknown loops or keys raise `ValueError`.
