# Data Model

This document defines the canonical vocabulary used across the framework and explains how adapters map dataset-specific terms into this vocabulary.

## Canonical Concepts

| Concept | Definition | Example (paired-context adapter binding) |
|---------|-----------|---------------------------|
| **Context entity** | An entity carrying contextual information used during evaluation | Patient |
| **Task instance** | An atomic unit to be evaluated | EHR item string |
| **Candidate** | A system producing scores | HuggingFace LLM or trained ensemble model |
| **Output dimension** | A canonical category against which a score is emitted | One of 10 privacy categories |
| **Condition** | The context configuration presented to the candidate | `context_free`, `correct_context`, `shuffled_context` |
| **Reference observer** | A human annotator producing reference labels | Physician |
| **Score** | A numeric value per (task_instance, context_entity, condition, output_dimension) | Privacy-category probability [0, 1] |
| **Delta** | Difference in scores across conditions | `correct_context - context_free` |

## Canonical Keys vs Display Labels

The framework distinguishes two forms for output-dimension identifiers:

### Canonical machine key
- Format: `snake_case`, ASCII-only, stable across time
- Used in: CSVs, JSON manifests, code, tests, config files, `artifact_index.json`
- Example: `sexual_reproductive_health`

### Display label
- Format: Human-readable, may contain spaces and capitalization
- Used in: Reports, figures, and the `label_space` field of `adapter_manifest.json`
- Example: `Sexual and reproductive health`

**Rule:** All programmatic interfaces (APIs, file columns, dict keys) use canonical keys. Display labels appear only in human-facing outputs and the `label_space` mapping that allows translation.

## Dataset-Specific Terms as Adapter-Owned Mappings

The following terms belong to the paired-context format and live only in `adapters/paired_context/` and its configuration. They are NOT shared concepts:

| Paired-context term | Canonical mapping |
|-------------|-------------------|
| Patient | `context_entity` |
| Physician | `reference_observer` |
| survey / `test` sheet | condition = `context_free` |
| interview / `interview` sheet | condition = `correct_context` |
| Delta_Physician | per-observer score delta across conditions (`delta_reference`) |
| "Behavioral health" (display) | display label for canonical key `behavioral_health` |

These terms may appear in:
- `adapters/paired_context/` source code
- Dataset config files
- Adapter test fixtures
- Data documentation

They must NOT appear in:
- `shared/` source code
- `tracks/` source code (except as pass-through from adapter DTOs)
- Canonical output file column headers

## Score Array Alignment

Scores can be represented in two forms:

### Wide/array form

NumPy array of shape `(n_rows, n_output_dimensions)`:
- Column order matches a declared `output_dimensions` list
- Used internally for computation efficiency
- The canonical column order is `sorted(label_space.keys())` unless the config explicitly declares a fixed ordering

### Long/tidy form

CSV or DataFrame with columns:
- `context_entity_id` — identifies the context entity for this row
- `task_instance_id` or `item_text` — identifies the task instance
- `output_dimension` — canonical key identifying which dimension
- `score` — the numeric value

### Conversion

Conversion between wide and long forms is lossless. The `output_dimensions` list provides the column-index-to-key mapping for wide form.

### Alignment rule

When multiple score arrays are compared (e.g., context-free vs correct-context), they MUST be aligned by the same `(context_entity_id, task_instance_id)` row ordering and the same `output_dimensions` column ordering.
