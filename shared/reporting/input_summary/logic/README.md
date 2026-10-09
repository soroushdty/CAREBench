# LM-ContextProbe Summary Logic

**Canonical path:** `shared/reporting/input_summary/logic`

This directory encapsulates the analytical logic responsible for computing descriptive statistics, audits, and metrics across the various data preprocessing stages. These modules do not handle file I/O directly; they compute the numerical summaries that the outputs directory visualizes or saves.

## Core Modules

### `audits.py`
Performs cross-stage diagnostic audits:
- **Standardization Impact**: Computes how the vocabulary shrank when transitioning from raw text to canonical standardized items (identifying many-to-one collapses).
- **Split Overlap**: Computes Jaccard similarities and intersections between train, test, and interview splits to ensure proper isolation or deliberate overlap.
- **Physician Disagreement**: Measures how often multiple physicians independently labeled the exact same patient/item pairs differently.
- **Missingness Matrix**: Calculates NaN density across all columns and dataset splits.

**Generated Files in `audits/`:**

| File                                        | Description                                                                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `missingness_summary.xlsx`                  | Consolidated missingness workbook — one sheet per stage; rows = splits, columns = dataset columns, values = missing percent |
| `stage_audit.csv`                           | Row, patient, item, and class-column counts for every stage and split                                                       |
| `standardization_impact_summary.csv`        | How item normalization changed item uniqueness and row counts                                                               |
| `raw_to_standardized_mapping.csv`           | Mapping frequencies from raw item text to standardized text                                                                 |
| `standardization_many_to_one_collapses.csv` | Standardized items that absorb multiple raw variants                                                                        |
| `split_overlap_*.csv`                       | Pairwise overlap metrics between splits for each stage                                                                      |
| `physician_*.csv`                           | Physician disagreement and merge-collapse diagnostics                                                                       |

### `delta.py`
Calculates the fundamental shift in physician labeling between Context-Free (Survey) and Context-Aware (Interview) phases. It tracks positive, negative, and zero deltas for all evaluated classes across the paired test split, separating out results by novel vs. repeated textual strata.

**Generated Files in `final/delta/`:**

Delta outputs compare survey vs. interview label values for paired test-set rows. A row is paired when a test-split patient appears in both survey and interview splits of the final combined snapshot. Only rows with non-null interview values are included.

**Stratum**: whether the patient-item appears in the training set (`repeated`) or not (`novel`).

| File                         | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `count_per_class.csv`        | Positive/negative/zero delta counts per class      |
| `percentage_per_class.csv`   | Positive/negative/zero delta percentages per class |
| `count_per_stratum.csv`      | Delta counts broken down by class and stratum      |
| `percentage_per_stratum.csv` | Delta percentages broken down by class and stratum |

### `divisions.py`
Provides utility functions to slice the master dataset by stage (`raw`, `standardized`, `post_physician_merge`, `final`) and split (`train`, `test`, `interview`, `combined`). It automatically resolves the correct class columns applicable to any given stage (e.g., stripping the `_survey` suffix if needed).

### `index_mapping.py`
Maintains logic for tracking vocabulary mappings across preprocessing boundaries.
