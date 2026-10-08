# Changelog

## Unreleased

- **Track 1 CLI runs through `RepresentationTrack`** (#1). `python main.py --track representation` now builds the dataset adapter and training strategy from config (`adapter`, default `paired_context`; `strategy`, default `existing_ensemble`) and runs them through the framework. `PairedContextRepresentationAdapter` loads the workbook and context records with the same loaders as Track 3, and `ExistingEnsembleTrainingStrategy` now builds the Stage 2 context vectors and runs the statistical analysis. Outputs on `examples/synthetic` are unchanged, and each run folder also gets `run_manifest.json`, `adapter_manifest.json`, `artifact_index.json`, and `resolved_config.yaml`.
- `PairedContextRepresentationAdapter` now reads the Track 1 config keys (`DIR_DATASET`, `TRAIN_SHEET`, `DIR_CONTEXT`, `llm`, ...) instead of `dataset_path` / `stage2_enabled` / `embedding_model`.
- `DIR_CONTEXT` is resolved against the project root (`--dir`), like `DIR_DATASET`, instead of the current working directory.
- **P-values now respect clustering by patient** (#16). A new patient-cluster sign-flip test (`shared/statistical/cluster_tests.py`) flips each patient's cells together and is exact when there are few patients.
  - Track 3 H4: `permutation_test_h4` used to flip each cell independently, which ignored clustering. It now uses the cluster test. The output keys are unchanged.
  - Track 1 H1 per class and H2: new `cluster_p` column, now the confirmatory p-value; `bh_adj_p` and `n_classes_significant` are computed from it. `binom_p` and `wilcoxon_p` are kept for reference.
  - Track 1 CMH: new `p_cluster` key.
  - With exact enumeration the smallest attainable p-value is `2^-(number of patients)`; see `docs/methodology.md`.

## 0.1.0 — 2026-10-07

Initial public release.

- **Track 3 (reasoning)**: LLM context-shift assay. Runs each item context-free, with the correct patient context, and with shuffled context; tests H1–H4 against physician judgment shifts. Backends: `huggingface`, `local_transformers`, `dry_run`.
- **Track 1 (representation)**: Bio_ClinicalBERT embeddings, ensemble training with nested leave-one-patient-out CV, Stage 2 context fusion, and statistical analysis (H1/H2, ICC, ablation, calibration).
- **`paired_context` dataset adapter** for the paired context-free / correct-context data format (`docs/data_format.md`). Column and sheet names are configurable.
- **Synthetic example dataset** (`examples/synthetic/`) with its generator (`scripts/make_synthetic_dataset.py`).
- **Post-hoc analysis bundle** for Track 3 scores (`scripts/analyze_llm_context_effects.py`).
- Colab/Jupyter launcher notebook (`main_notebook.ipynb`).
- Reproducibility manifests, pinned requirements, and CI.
