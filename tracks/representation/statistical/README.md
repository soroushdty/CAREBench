# tracks/representation/statistical/

Track 1 statistical analysis pipeline for representation learning.

This directory contains the statistical analysis modules. It includes:

- **`hypotheses/`** — Hypothesis tests H1 and H2 (representation learning endpoints)
- **`orchestrator/`** — Top-level statistical analysis orchestration (`run_analysis.py`)
- **`reporting/`** — Result reporting and figure generation (`arch_compare.py`, `figures.py`, `stratum.py`)

Shared metrics (bootstrap, delta, entropy, ICC, tau check, calibration) live in `shared/statistical/` and `shared/evaluation/` and are imported from there.

## Canonical path

`tracks/representation/statistical/`
