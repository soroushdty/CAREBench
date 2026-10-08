# shared/reporting/input_summary/

**Canonical path:** `shared/reporting/input_summary/`

Statistical summary and diagnostic artifact generation logic for the preprocessing pipeline. This module produces data-quality and preprocessing diagnostics for each processing stage and dataset split.

## Subdirectories

- **`logic/`** — Core summary logic: audits, delta computation, divisions, index mapping
- **`orchestrator/`** — Top-level summary orchestration (`summary.py`)
- **`utils/`** — Supporting utilities: cleanup, I/O, logging, settings loading

## Usage

When `ENABLE_SUMMARY: true` in `configs/main_config.yaml`, the pipeline writes a summary folder containing data-quality and preprocessing diagnostics for each stage and split.

A **stage** is one processing step: `raw`, `standardized`, `post_reference_observer_merge` (was: `post_physician_merge`), or `final`.
A **split** is one dataset partition: `train`, `test`, `interview`, or `combined`.

### Folder Structure

```
<summary_dir>/
├── preprocessing_stages/
│   └── <stage>/<split>/     # Summary files per stage and split
├── final/
│   └── delta/               # Context-free vs correct-context label comparison (was: survey-vs-interview)
└── audits/                  # Cross-stage and cross-split audit files
```

All outputs are individually toggleable via `SUMMARY.outputs.*` keys in `configs/summary_config.yaml`. Some stage/split folders may omit files if required columns are missing or the corresponding feature is disabled.
