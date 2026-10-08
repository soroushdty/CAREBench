# Changelog

## 0.1.0 — 2026-10-07

Initial public release.

- **Track 3 (reasoning)**: LLM context-shift assay. Runs each item context-free, with the correct patient context, and with shuffled context; tests H1–H4 against physician judgment shifts. Backends: `huggingface`, `local_transformers`, `dry_run`.
- **Track 1 (representation)**: Bio_ClinicalBERT embeddings, ensemble training with nested leave-one-patient-out CV, Stage 2 context fusion, and statistical analysis (H1/H2, ICC, ablation, calibration).
- **`paired_context` dataset adapter** for the paired context-free / correct-context data format (`docs/data_format.md`). Column and sheet names are configurable.
- **Synthetic example dataset** (`examples/synthetic/`) with its generator (`scripts/make_synthetic_dataset.py`).
- **Post-hoc analysis bundle** for Track 3 scores (`scripts/analyze_llm_context_effects.py`).
- Colab/Jupyter launcher notebook (`main_notebook.ipynb`).
- Reproducibility manifests, pinned requirements, and CI.
