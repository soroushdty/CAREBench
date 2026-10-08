# Adapters

This document describes adapter responsibilities, the built-in paired-context adapter, and how to add a new dataset adapter.

## Adapter Responsibilities

An adapter is responsible for:

1. **Loading raw dataset files** — reading Excel workbooks, JSON files, or other data sources.
2. **Translating dataset-specific names to canonical keys** — column names, label names, sheet names.
3. **Producing canonical data containers** — `RepresentationDataset` for Track 1, `PairedDataset` for Track 3.
4. **Declaring provenance** — through an adapter manifest (adapter name, dataset paths, label space, row counts).

An adapter is **not** responsible for:

- Training or evaluation logic (lives in `tracks/`)
- Statistical analysis or scoring (lives in `shared/evaluation/` or `tracks/`)
- Reporting or visualization (lives in `shared/reporting/` or `tracks/`)
- Config resolution or CLI parsing (lives in `main.py` or track runners)

## Paired-Context Adapter

The built-in adapter lives at `adapters/paired_context/` and serves both tracks. It reads datasets in the paired-context data format described in [`data_format.md`](data_format.md). Everything dataset-specific is configurable, so datasets in this format need configuration only, not new code.

### Label Space (`adapters/paired_context/labels.py`)

The default ten output dimensions (the SHARES sensitive-data categories), with machine keys and display names:

| Canonical key | Display name |
|---------------|-------------|
| `behavioral_health` | Behavioral health |
| `diagnoses` | Diagnoses |
| `disabilities` | Disabilities |
| `infectious_diseases` | Infectious diseases |
| `genetics` | Genetics |
| `medications` | Medications |
| `sexual_reproductive_health` | Sexual and reproductive health |
| `social_determinants_of_health` | Social determinants of health |
| `violence` | Violence |
| `other` | Other |

### Column Mapping (`adapters/paired_context/column_map.py`)

`PairedContextColumnMap` maps workbook column and sheet names (by default `Patient`, `Physician`, `Item`; sheets `train`, `test`, `interview`) to canonical framework concepts (context_entity, reference_observer, task_instance, condition). `PairedContextColumnMap.from_config()` builds it from the `patient_col`, `physician_col`, `item_col`, `*_sheet`, and `physician_count` config keys.

### Context Loading (`adapters/paired_context/context_adapter.py`)

Loads per-entity clinical snapshots from `patient_summaries.json` and renders them in a fixed field order. Track 3's `ContextBuilder` uses these to build correct-context and shuffled-context prompts, with label-leakage checks.

### Reference Aggregation (`adapters/paired_context/reference_adapter.py`)

Computes observer-group consensus. It averages each category's labels across the `physician_count` observers who rated a (patient, item) pair, giving context-free and correct-context reference labels.

### Reasoning Adapter (`adapters/paired_context/reasoning_adapter.py`)

Combines the dataset, context, and reference adapters to produce the `PairedDataset` DTO that `tracks/reasoning/` consumes. Selected in `tracks/reasoning/dataset_loader.py` by `data.adapter` (default `paired_context`). Exposes `manifest()` for reproducibility.

### Representation Adapter (`adapters/paired_context/representation_adapter.py`)

Produces a `RepresentationDataset` (defined in `shared/adapters/base.py`) for `tracks/representation/framework.py`.

## Adding a New Dataset Adapter

To add support for a new dataset (e.g., a hypothetical "mayo" dataset whose layout differs from the paired-context format):

1. **Create the adapter package**: `adapters/mayo/__init__.py`, `adapters/mayo/labels.py`, `adapters/mayo/column_map.py`, etc.
2. **Implement the protocols** in `shared/adapters/base.py`: for Track 1, `RepresentationDatasetAdapter`. For Track 3, implement a reasoning adapter that produces a `PairedDataset`.
3. **Register and configure**: Add the reasoning adapter to `_ADAPTER_REGISTRY` in `tracks/reasoning/dataset_loader.py`, then create configs that select it (`data.adapter: mayo`).
4. **Add tests**: Place adapter-specific tests under `tests/adapters/mayo/`.
5. **Do NOT modify `shared/` or `tracks/`** to accommodate the new dataset. If canonical keys or protocols are insufficient, raise the gap in an architecture discussion before extending core schemas.

## What Must Not Go In `shared/` or `tracks/`

- Dataset-specific column names or sheet names
- Dataset-specific label display names
- Dataset-specific physician-pair, survey/interview, or patient-summary assumptions
- Branching on dataset identity (`if adapter == "<dataset>": ...`)
- Imports of any `adapters/*` submodule except through the documented injection points

## A Note on Future Adapters

Any future adapter examples in this document are **conceptual**: they illustrate structure but do not create runnable scaffold directories. Unsupported modules are never documented as if implemented.
