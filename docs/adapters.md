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

### Label Space (`shared/label_space.py`)

A `LabelSpace` is the ordered list of output dimensions. Each has a **key** (snake_case, used in score files, LLM responses, manifests and result tables), a **display name** (the workbook column name, shown in reports) and an optional **definition** (shown to the LLM in the Track 3 prompt).

The default is the ten SHARES sensitive-data categories (`DEFAULT_LABEL_SPACE`; `adapters/paired_context/labels.py` wraps it):

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

A different taxonomy needs configuration only. `classes` lists the workbook columns, and the optional `class_definitions` gives a definition string, or a mapping with `key` and/or `definition`, per class:

```yaml
classes:
  - Mood & anxiety
  - Drug therapy
  - Housing
class_definitions:
  Mood & anxiety: Depression, anxiety, and related mental health conditions
  Drug therapy:
    key: medications
    definition: Any prescribed or over-the-counter drug
```

In Track 3 these keys go under `data:`; in Track 1 they are top-level. Keys default to the name in snake_case (`Mood & anxiety` → `mood_anxiety`). A class named like a default category (for example `Genetics`) keeps that category's key and definition unless you override them. A class with no definition appears in the prompt by its key only, and Track 3 logs a warning. Keys and display names must be unique; config validation reports duplicates, malformed keys, and definitions for names that are not in `classes`.

The Track 3 prompt's opening line and the word before "categories" are set in the assay config's optional `prompt` section. The defaults suit the SHARES taxonomy:

```yaml
prompt:
  intro: You are a clinical documentation expert. Classify the following EHR item into note sections.
  category_type: ""        # default "privacy"; also used in the report text
```

With the default ten classes, no `class_definitions` and no `prompt` section, prompts are byte-identical to earlier versions, so cached responses stay valid. Track 3 compares each cached response's prompt hash with the current prompt and runs the call again when they differ, so changing the label space or the prompt never reuses answers to an older prompt.

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

1. **Create the adapter package**: `adapters/mayo/__init__.py`, `adapters/mayo/column_map.py`, etc. A taxonomy other than the default only needs config (see [Label Space](#label-space-sharedlabel_spacepy)); the adapter builds its `LabelSpace` with `LabelSpace.from_config`.
2. **Implement the protocols** in `shared/adapters/base.py`: for Track 1, `RepresentationDatasetAdapter`. For Track 3, implement a reasoning adapter that produces a `PairedDataset` with its `label_space`.
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
