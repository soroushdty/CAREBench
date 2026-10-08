# shared/preprocessing/

**Canonical path:** `shared/preprocessing/`

Shared preprocessing logic usable by all evaluation tracks.

## Intended Contents

- `fuzzy_mapping.py` — fuzzy item matching
- `index_mapping.py` — index map construction
- `item_standardization.py` — JSON-based item standardization
- `adapters/` — dataset-specific adapter stubs (see `adapters/README.md`)

## Design Notes

The core preprocessing pipeline accepts column names as configuration parameters (`item_col`, `patient_col`, `physician_col`) rather than hard-coded strings. Dataset-specific column-mapping logic belongs in `adapters/`.
