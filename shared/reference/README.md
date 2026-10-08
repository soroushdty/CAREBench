# shared/reference/

**Canonical path:** `shared/reference/`

Reference observer (was: physician) reconciliation utilities shared across all evaluation tracks. This module provides logic for aggregating and auditing annotations from multiple reference observers.

## Subdirectories

- **`aggregation/`** — Pair-mean aggregation of reference observer labels (`paired_reference_mean.py`). Computes mean labels across reference observers (was: physicians) for each context entity (was: patient) and task instance (was: item).
- **`diagnostics/`** — Audit and diagnostic utilities for detecting duplicate or mismatched responses (`duplicate_response_audit.py`).

## Terminology

| Dataset-Specific Term | Canonical Term |
|---|---|
| `physician` / `human` rater | `reference_observer` |
| `Patient` | `context_entity_id` |
| `Item` | `task_instance` |
| `physician_survey` / `survey` | `reference_context_free` |
| `physician_interview` / `interview` | `reference_correct_context` |
