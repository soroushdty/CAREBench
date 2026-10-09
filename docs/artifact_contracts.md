# Artifact Contracts

This document defines the required output artifacts for representation and reasoning runs. Every contract is machine-checkable via `tests/artifact_contracts/`.

## Common Artifacts (Required by All Runs)

| File | Format | Required keys / contents |
|------|--------|--------------------------|
| `run_manifest.json` | JSON | `run_id`, `timestamp_utc`, `track`, `python_version`, `config_path`, `output_dir`, `entry_command` |
| `resolved_config.yaml` | YAML | Full resolved config as used (after overlays and defaults) |
| `adapter_manifest.json` | JSON | `adapter_name`, `adapter_class`, `label_space`, `reference_aggregation_policy` |
| `artifact_index.json` | JSON | Object mapping canonical artifact roles to actual file paths |

## Adapter Manifest Contents

The `adapter_manifest.json` provides reproducibility metadata about the adapter:

```json
{
  "adapter_name": "paired_context_reasoning",
  "adapter_class": "adapters.paired_context.reasoning_adapter.PairedContextReasoningAdapter",
  "dataset_path": "examples/synthetic/dataset.xlsx",
  "patient_summaries_path": "examples/synthetic/patient_summaries.json",
  "mapping_file_path": "examples/synthetic/mapping.json",
  "label_space": {
    "behavioral_health": "Behavioral health",
    "diagnoses": "Diagnoses",
    "disabilities": "Disabilities",
    "infectious_diseases": "Infectious diseases",
    "genetics": "Genetics",
    "medications": "Medications",
    "sexual_reproductive_health": "Sexual and reproductive health",
    "social_determinants_of_health": "Social determinants of health",
    "violence": "Violence",
    "other": "Other"
  },
  "reference_aggregation_policy": "paired_physician_consensus",
  "row_counts": {
    "n_patients": 6,
    "n_items": 51,
    "n_patient_item_pairs": 126
  }
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `adapter_name` | string | Human-readable adapter identifier |
| `adapter_class` | string | Fully qualified class path |
| `label_space` | object | `{canonical_key: display_label}` for all output dimensions |
| `reference_aggregation_policy` | string | How reference labels are aggregated |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `dataset_path` | string | Path to primary dataset file |
| `patient_summaries_path` | string | Path to patient summaries JSON |
| `mapping_file_path` | string | Path to item mapping JSON |
| `row_counts` | object | Dataset size metadata |

## Track 1 Artifact Contract

| Canonical role | Expected path | Required | Contents |
|---------------|---------------|----------|----------|
| `track1.predictions.context_free` | `predictions/context_free_*.npy` or CSV | yes | (n_test, n_output_dim) float array |
| `track1.predictions.correct_context` | `predictions/correct_context_*.npy` or CSV | if Stage 2 enabled | (n_test, n_output_dim) float array |
| `track1.metrics.per_dimension` | `metrics/per_class_metrics.csv` or equivalent | yes | Columns: `output_dimension`, `metric_name`, `value` |
| `track1.run_summary` | `run_summary.json` | yes | Top-level metrics, thresholds |
| `track1.bundle` | `model/ensemble_bundle.joblib` | if produced | Serialized model ensemble |
| `track1.checkpoints` | `model/checkpoints/` | if produced | Per-fold training checkpoints |

## Track 3 Artifact Contract

| Canonical role | Expected path | Required | Contents |
|---------------|---------------|----------|----------|
| `track3.scores.context_free` | `scores/<slug>/context_free_scores.csv` | yes | Columns: `patient_id`, `item_text`, + one per output dimension |
| `track3.scores.correct_context` | `scores/<slug>/correct_context_scores.csv` | yes | Same schema |
| `track3.scores.shuffled_context` | `scores/<slug>/shuffled_context_scores.csv` | yes | Same schema |
| `track3.hypothesis_summary` | `reports/<slug>/analysis_report.md` | yes | Endpoint results in markdown, under the names in `shared/endpoints.py` |
| `track3.run_summary` | `run_manifest.json` | yes | Includes model_id, dry_run flag, n_items |

## Column Constraints

Score CSVs and metric files MUST:

- Include output-dimension identifiers using **canonical machine keys** (e.g., `behavioral_health`), never display labels (e.g., `Behavioral Health`).
- Align with the `label_space` declared in `adapter_manifest.json`.

Display labels are surfaced through:
- The `label_space` mapping in `adapter_manifest.json`
- Report markdown files (human-readable)

## artifact_index.json

When existing filenames differ from canonical semantic roles, `artifact_index.json` provides the mapping:

```json
{
  "track3.scores.context_free": "scores/dry_run_example/context_free_scores.csv",
  "track3.scores.correct_context": "scores/dry_run_example/correct_context_scores.csv",
  "track3.scores.shuffled_context": "scores/dry_run_example/shuffled_context_scores.csv",
  "track3.hypothesis_summary": "reports/dry_run_example/analysis_report.md"
}
```

This allows downstream tools to locate artifacts by semantic role without hardcoding filenames.
