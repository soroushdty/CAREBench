# LM-ContextProbe Summary Orchestrator

**Canonical path:** `shared/reporting/input_summary/orchestrator`

This directory acts as the central conductor for the preprocessing summary pipeline.

## Core Module: `summary.py`

The `summary()` function is the master entry point called after data preprocessing has finished. It accepts in-memory snapshots of the dataset at four critical stages (Raw, Standardized, Post-Physician-Merge, Final). 

Its responsibilities include:
1. Orchestrating the creation of a structured `output/summary` folder.
2. Initializing the reporting configurations, plotting DPIs, and bar limits.
3. Invoking the logic modules to compute stage-by-stage audits, distributions, and missingness metrics.
4. Calling the outputs modules to systematically dump hundreds of diagnostic `.csv`, `.png`, and `.xlsx` artifacts into logically nested subdirectories.
5. Emitting detailed runtime configurations and index files to document the final state of the cleaned dataset.
