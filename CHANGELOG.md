# Changelog

## Unreleased

- **Configurable label taxonomy** (#5). The output dimensions now come from config instead of being fixed to the ten SHARES categories. `classes` lists the workbook columns as before, and the new optional `class_definitions` sets each class's prompt definition and, if needed, its key (`docs/adapters.md`). A new `shared.label_space.LabelSpace` replaces the five separate copies of the category list: Track 3's prompt, response validation, dry-run responses, score files, reports and post-hoc bundle (`scripts/analyze_llm_context_effects.py --config`), and Track 1's label-space manifest. With the default ten classes, prompts and score files are unchanged, so cached responses stay valid.
- **Configurable prompt wording** (#5). The assay config's optional `prompt` section sets the Track 3 prompt's opening line (`intro`) and the word before "categories" (`category_type`, default `privacy`), which is also used in the Track 3 and post-hoc bundle reports. Defaults are unchanged.
- **Stale cached responses are run again.** The response cache is keyed by model, condition, patient and item. It now also compares the stored prompt hash with the current prompt, so a changed label space, definition or prompt setting no longer reuses answers to the old prompt. Caches made with the current prompts are unaffected.
- Track 3's `analysis_report.md` showed N/A in the per-category H1 table, because results were looked up by display name instead of key. It now shows the values.
- Track 1's Figure 1 (per-class delta histograms) sizes its grid to the number of classes. It was fixed at 2 × 5 and failed with more than ten classes.
- **No references to the private methodology document** (#21). Docstrings, comments and config no longer cite section, table or figure numbers from the unpublished protocol; where a pointer helps, they link to `docs/methodology.md`. Array shapes are written as `(n_pairs, n_classes)` instead of the original study's sizes. A new test (`tests/architecture/test_no_private_references.py`) fails if a section sign (U+00A7) appears outside `docs/`.
- Figure 3's legend now shows the actual number of model-vs-physician and human-human ICCs instead of the fixed counts 24 and 12.
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
