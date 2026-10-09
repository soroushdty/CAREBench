# CAREBench

**CAREBench** is a research framework for studying how language models respond to patient-level clinical context when classifying sensitive EHR data, and whether those responses match the context-dependent judgment shifts that physicians show.

It runs any dataset with human reference labels collected under two conditions: first **without** patient context, then **with** a patient's clinical snapshot. It measures how much a model's judgments move when context is added, and whether they move the same way the human judgments did.

The framework is a **functional analogy to physician decision-making**. Each evaluation track stands for one way a physician could reach a context-dependent judgment: drawing on general clinical knowledge (representation), learning from supervised case experience (adaptation), or deliberating over the chart at decision time (reasoning). Comparing the tracks against the physician reference tests which mechanism reproduces the physicians' context shifts. It does not claim to model how physicians think.

> This repository contains code and a fully **synthetic** example dataset only. It does not include data from any human-subjects study.

## Quickstart

```bash
python -m pip install -r requirements.txt
export PYTHONHASHSEED=42

# Track 3: LLM context-shift assay on the synthetic example, no LLM calls
python main.py --track reasoning -- --config configs/assay_config.yaml --dry_run --model dry_run_example
```

Outputs go to `output/assay/<run_id>/`. To use your own data, see [`docs/data_format.md`](docs/data_format.md).

## Architecture

The code is organized into three top-level packages (full tree under [Project Structure](#project-structure)):

- **`shared/`**: dataset-agnostic utilities, preprocessing, statistics, evaluation, and the adapter contracts (`shared/adapters/base.py`).
- **`tracks/`**: the evaluation pipelines: Track 1 representation and Track 3 reasoning. Track 2 (adaptation) is planned ([#13](https://github.com/soroushdty/CAREBench/issues/13)).
- **`adapters/`**: dataset adapters that translate a dataset into the framework's canonical objects. The built-in `paired_context` adapter serves both tracks.

Dependencies point one way: `tracks/` and `adapters/` import from `shared/`; `shared/` imports from neither, and `adapters/` never imports from `tracks/`. Architecture tests in `tests/architecture/` enforce this.

### Tracks

| Track | Physician analogue | Question | Status |
|-------|--------------------|----------|--------|
| 1 – Representation | General clinical knowledge, before any task-specific training | Is the context shift already latent in pretrained knowledge? | Available |
| 2 – Adaptation | Learning from supervised case experience (clinical training, feedback on real cases) | Can the shift be learned from labelled cases and carried over to new patients? | Planned ([#13](https://github.com/soroushdty/CAREBench/issues/13)) |
| 3 – Reasoning | Deliberating over the chart at decision time, without changing what the model knows | Does the shift come from reasoning at decision time? | Available |

- **[Track 1 — Representation](tracks/representation/README.md)** (`tracks/representation/`): Bio-ClinicalBERT embedding pipeline, multilabel classifier, context-aware fusion head, LOPO-CV, and statistical hypothesis analysis (H1–H2).
- **[Track 3 — Reasoning](tracks/reasoning/README.md)** (`tracks/reasoning/`): LLM context-shift assay pipeline. Runs each EHR item under three conditions (context-free, correct-context, shuffled-context) and tests H1–H4 against physician judgment shifts.

### Shared Layer

The [`shared/`](shared/) directory contains cross-track utilities, preprocessing, statistics, evaluation, and adapter contracts. Both tracks import from `shared.*` for general-purpose logic. See the subdirectory READMEs for details on each module.

---

## Research Overview

CAREBench implements an **LLM context-shift assay**. Human reference observers (for example, physicians working in fixed pairs) label EHR item strings into sensitive-data categories under two conditions. First they see the item without patient information (context-free, `reference_context_free`). Then they see it with a structured clinical snapshot of the patient (correct-context, `reference_correct_context`). These matched counterfactual labels allow a direct measurement of context-induced judgment shifts (`delta_reference`).

The central question is: **Does a model change its privacy-category judgment when patient context is added, and do those changes match the human judgment shifts?**

The primary endpoints are:

- **H1 — Context Sensitivity**: Does the LLM's mean absolute delta (correct-context minus context-free) exceed zero?
- **H2 — Directional Physician Alignment**: Do LLM context-induced deltas agree in sign with physician judgment deltas?
- **H3 — Class-Level Correspondence**: Do class-level mean LLM deltas correlate with class-level mean physician deltas across the privacy categories?
- **H4 — Correct vs Shuffled Context Control**: Does correct patient context produce stronger alignment with physician deltas than shuffled (wrong-patient) context?

The default label space is the ten sensitive-data categories used in the SHARES project: behavioral_health, diagnoses, disabilities, infectious_diseases, genetics, medications, sexual_reproductive_health, social_determinants_of_health, violence, other. Other taxonomies are set in config (`classes` and `class_definitions`; see [`docs/adapters.md`](docs/adapters.md#label-space-sharedlabel_spacepy)).

**Scope**: Claims are scoped to the patients and reference observers of the dataset being evaluated. The endpoints measure context-induced shifts and their agreement with the reference shifts, not classification accuracy. The one exception is Track 1's H2, which tests whether context lowers the Brier score against the correct-context reference labels.

Formal definitions of every endpoint, the unit of analysis, the statistical tests, and the design decisions behind them are in [`docs/methodology.md`](docs/methodology.md).

## Origins and Acknowledgments

CAREBench grew out of work in Dr. Adela Grando's **SHARES** project at Arizona State University on patient-controlled, granular segmentation of sensitive health data. Its paired context-free / correct-context design is modeled on the study by Kaufman et al., in which physicians categorized EHR items first without and then with additional patient context:

- Kaufman HJ, Banerjee I, Wei M, et al.; Grando MA. *New Automated Granular Data Segmentation Approach: Context Impacts Categorization by Physicians.* JMIR Preprints #66059, 2024. [doi:10.2196/preprints.66059](https://doi.org/10.2196/preprints.66059)
- Lee P, Dhadwal AS, Kaiser M, Dianaty S, et al.; Grando A. *Assessing the Effectiveness and Scalability of FHIR-Based Granular Data Segmentation Technology.* Appl Clin Inform. 2026;17(3):423–433. [doi:10.1055/a-2863-4129](https://doi.org/10.1055/a-2863-4129)

An early version of this analysis was presented as: Dianaty S, Kaiser M, Murcko A, Grando A. *Early Evidence for Context-Aware Large Language Models (LLMs) in Sensitive Health Data Classification.* AcademyHealth Annual Research Meeting 2026, Seattle, WA, May 30 – Jun 2, 2026.

The study data is not distributed with CAREBench. The bundled example dataset is synthetic.

## Project Status

CAREBench 0.1.0 is an early public release of a research framework, not an established benchmark.

- **Tested:** both tracks run end to end on the bundled synthetic dataset, under CI.
- **Not yet available:** results on real data, simulation studies of the endpoints' false-positive rate and power ([#18](https://github.com/soroushdty/CAREBench/issues/18)), a second dataset format, and a configurable label taxonomy. Known gaps in the current statistics are listed in [`docs/methodology.md`](docs/methodology.md#known-limitations).

**Development history.** CAREBench began as the analysis code for the study described under [Origins and Acknowledgments](#origins-and-acknowledgments) and was developed in a private repository from January to June 2026 (about 800 commits). It was then generalized into a dataset-agnostic framework. This public repository was created from a sanitized snapshot, so that study data and private development material are not in its git history. A short public commit history does not reflect the age of the code.

## LLM Context-Shift Assay (Track 3)

The assay runs each EHR item through a Hugging Face LLM under three conditions:

1. **Context-free**: Item text only, no patient information.
2. **Correct-context**: Item text plus the correct patient's clinical snapshot.
3. **Shuffled-context**: Item text plus a randomly selected different patient's clinical snapshot (control).

For each condition the LLM outputs a JSON object with probability scores for all ten privacy categories. The assay then computes physician and LLM judgment deltas and tests H1–H4.

Canonical entry point: `tracks/reasoning/run_assay.py`. See [`tracks/reasoning/README.md`](tracks/reasoning/README.md) for full details.

## Installation

```bash
python -m pip install -r requirements.txt
```

Requires Python >= 3.10.

## Usage

### Track 3 — Reasoning (dry run)

```bash
python main.py --track reasoning -- --config configs/assay_config.yaml --dry_run --model dry_run_example
```

The dry run generates deterministic mock scores and complete artifact trees without calling an LLM. See [`docs/running_example.md`](docs/running_example.md) for full guidance.

### Track 1 — Representation

```bash
python main.py --track representation -- --config configs/main_config.yaml
```

Runs the representation learning pipeline: preprocessing, embedding computation, ensemble training with LOPO-CV, and statistical hypothesis analysis.

The first run downloads the `emilyalsentzer/Bio_ClinicalBERT` embedding model (~440 MB) from Hugging Face.

### Track 3 with a real LLM

```bash
python main.py --track reasoning -- --config configs/assay_config.yaml --model meta-llama/Llama-3.1-8B-Instruct
```

Requires a valid `HF_TOKEN` environment variable.

### Track 1 general usage

```bash
python main.py --track representation -- [--dir .] [--hf_token TOKEN] [--config PATH]
```

Config resolution order when `--config` is omitted:

1. A directory named `configs/` in the repo root — if it contains `main_config.yaml`, use it.
2. Recursive search for `main_config.yaml` across the repo.
3. Error if not found.

Tokens may also be supplied via environment variables:

- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — HuggingFace API token

CLI args take precedence over environment variables when both are provided.

### Inject a custom Track 1 strategy

`tracks/representation/framework.py` composes a dataset adapter, a training strategy, and zero or more artifact writers through dependency injection. To plug in a custom strategy:

```python
from tracks.representation.framework import RepresentationTrack
from tracks.representation.strategies.base import TrainingStrategy, RepresentationDataset
from adapters.paired_context import PairedContextRepresentationAdapter

class MyStrategy:
    def fit_and_evaluate(self, dataset: RepresentationDataset, config: dict) -> dict:
        # ... your training/evaluation code ...
        return {"metrics": {...}, "predictions": {...}}

track = RepresentationTrack(
    adapter=PairedContextRepresentationAdapter(),
    strategy=MyStrategy(),
    artifact_writers=[...],
)
track.run(config)
```

See [`docs/architecture.md`](docs/architecture.md) for the adapter/strategy protocols and [`docs/adapters.md`](docs/adapters.md) for writing a new dataset adapter.

### Notebook

For Google Colab, Jupyter, or HPC: open `main_notebook.ipynb`.

## Assay Pipeline Stages

| Stage                    | Description                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Preprocessing**        | Loads the dataset workbook, standardizes item text via fuzzy matching, aggregates physician labels by pair, generates summary artifacts          |
| **Dataset Loading**      | Constructs paired physician consensus labels (survey and interview) and computes Delta_Physician (`delta_reference`) for all patient–item pairs  |
| **Context Building**     | Builds correct-context and shuffled-context patient snapshots with label leakage prevention                                                     |
| **LLM Prompting**        | Runs each item under three conditions (context-free, correct-context, shuffled-context) through a Hugging Face LLM                              |
| **Score Parsing**        | Validates LLM JSON outputs against the privacy-category schema and exports per-condition score CSVs                                             |
| **Hypothesis Analysis**  | Computes H1–H4 endpoints with patient-cluster bootstrap CIs and permutation tests                                                               |
| **Report Generation**    | Produces a markdown analysis report with all results, per-category breakdowns, and a limitations section                                        |

## Configuration

All configuration files live in `configs/`.

### LLM Assay Configuration

`configs/assay_config.yaml` controls the LLM context-shift assay:

| Key | Purpose |
|-----|---------|
| `backend` | `huggingface`, `local_transformers`, or `dry_run` |
| `model_ids` | List of Hugging Face model IDs to evaluate |
| `hf_token_env` | Environment variable name for the HF token |
| `temperature`, `max_tokens`, `top_p`, `seed` | LLM generation parameters |
| `shuffled_context_seed`, `bootstrap_seed`, `permutation_seed` | Reproducibility seeds |
| `n_bootstrap_resamples`, `n_permutations` | Statistical test parameters |
| `output_dir`, `cache_dir`, `scores_dir`, `reports_dir` | Output paths |

### Track 1 / Representation Configuration

Three YAML files control the Stage-1/Stage-2 pipeline:

| File                             | Purpose                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `configs/main_config.yaml`       | Dataset paths, schema column names, label space, embedding model, preprocessing controls |
| `configs/training_config.yaml`   | Model architecture, training hyperparameters, threshold strategy, HP search settings     |
| `configs/summary_config.yaml`    | Which preprocessing summary artifacts to generate (individual toggles per output)        |

## Roadmap

Planned work is tracked in [GitHub issues](https://github.com/soroushdty/CAREBench/issues). None of it is available yet:

- **Context perturbation suite**: field-level ablation, irrelevant-context insertion, field reordering, and item paraphrasing for the reasoning track, to test whether context shifts come from clinically relevant content.
- **TRIPOD-LLM report**: an auto-filled TRIPOD-LLM checklist per run, with links to the supporting artifacts and explicit TODOs for author-only items.
- **Multi-agent reference emulation**: blind LLM rater pairs with consensus and adjudication, mirroring the paired-physician reference design (aligned with the EviTrace multi-agent roadmap).
- **Track 2 — Adaptation** ([#13](https://github.com/soroushdty/CAREBench/issues/13)): the analogue of a physician learning from supervised case experience. A model is fine-tuned on labelled (patient context, item) cases from training patients, then evaluated on new patients under the same three conditions as Track 3. A learning curve shows how alignment with physicians grows with the number of patients learned from.
- **Patient-grouped cross-validation** ([#10](https://github.com/soroushdty/CAREBench/issues/10)): shared, configurable splits (leave-one-patient-out or grouped k-fold) for Tracks 1 and 2.
- **Shuffled-context condition for Track 1** ([#11](https://github.com/soroushdty/CAREBench/issues/11)): a wrong-patient context control, so Track 1 can test whether the model uses *this* patient's context, as Track 3's H4 does.
- **Same-model comparison across tracks** ([#12](https://github.com/soroushdty/CAREBench/issues/12)): one model family run frozen, fine-tuned, and prompted, so differences between tracks reflect the mechanism rather than the model.
- **Distribution-shift evaluation**: requires a second dataset in the paired-context format.
- **Other inputs and backends**: FHIR/OMOP ingestion, confidence-weighted or adjudicated reference aggregation, and hosted-API LLM backends.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — Package boundaries and dependency rules
- [`docs/data_format.md`](docs/data_format.md) — Input files expected by the paired-context adapter
- [`docs/adapters.md`](docs/adapters.md) — Adapter responsibilities and the paired-context adapter
- [`docs/running_example.md`](docs/running_example.md) — Commands, outputs, and caveats for the synthetic example
- [`docs/artifact_contracts.md`](docs/artifact_contracts.md) — Required output artifacts and schemas
- [`docs/data_model.md`](docs/data_model.md) — Canonical vocabulary and data model
- [`docs/methodology.md`](docs/methodology.md) — Endpoints, statistical tests, design rationale, and known limitations

## Project Structure

```
CAREBench/
├── shared/                    # Dataset-agnostic code shared by both tracks
│   ├── adapters/              # Adapter contracts (protocols, RepresentationDataset)
│   ├── io/                    # Dataset-neutral Excel reading
│   ├── utils/                 # General-purpose utilities (array, file, JSON, mapping, path, text)
│   ├── prerequisites/         # Config loading, reproducibility guards, logging setup
│   ├── preprocessing/         # Shared preprocessing (fuzzy matching, index mapping, standardization)
│   ├── embeddings/            # Embedding computation
│   ├── reference/             # Reference observer aggregation and audit diagnostics
│   ├── reporting/             # Input summary reporting infrastructure
│   ├── statistical/           # Shared metrics (bootstrap, ICC, entropy, delta, tau)
│   └── evaluation/            # Shared evaluation (hypothesis analysis, report generation, calibration)
├── tracks/                    # Track-specific pipelines
│   ├── representation/        # Track 1: embedding + training + statistical analysis
│   │   ├── strategies/        # Injectable training strategies
│   │   ├── training/          # Ensemble pipeline, calibration, CV, threshold tuning
│   │   ├── models/            # EnsemblePredictor, MultiLabelModel, ModelRegistry
│   │   └── statistical/       # H1–H2 hypotheses, run_analysis orchestrator, reporting
│   └── reasoning/             # Track 3: LLM assay pipeline
│       └── bundle/            # Post-hoc analysis bundle (deltas, H1–H4, report)
├── adapters/                  # Dataset adapters
│   └── paired_context/        # Paired-context adapter (column maps, labels, Track 1 and Track 3 adapters)
├── configs/                   # All YAML configuration files
├── examples/synthetic/        # Synthetic example dataset
├── scripts/                   # Synthetic dataset generator; post-hoc Track 3 analysis bundle
├── docs/                      # Architecture, data format, adapters, artifact contracts
├── tests/                     # Test suite (architecture, shared, tracks, adapters, audits)
├── data/                      # Local input data (git-ignored)
├── output/                    # Pipeline outputs (git-ignored)
├── main.py                    # CLI dispatcher (--track representation|reasoning)
└── main_notebook.ipynb        # Colab/Jupyter launcher
```

## Requirements

- Python >= 3.10
- PyTorch 2.x, Transformers 5.x, scikit-learn, pandas, numpy, scipy
- See `requirements.txt` for pinned versions

## Reproducibility

For full reproducibility, **PYTHONHASHSEED must be set as an environment variable before launching Python**:

```bash
export PYTHONHASHSEED=42
python main.py ...
```

Setting `PYTHONHASHSEED` at runtime (e.g., via `os.environ` in code) has no effect. This is required for deterministic hashing and reproducible results.

## Citation

If you use CAREBench, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), along with the Kaufman et al. study listed under [Origins and Acknowledgments](#origins-and-acknowledgments).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Soroush Dianaty.
