# tracks/reasoning/

Track 3: LLM reasoning/assay pipeline for context-shift evaluation.

This track covers the full LLM-based assay workflow, including:

- **`run_assay.py`** — Main entry point for the LLM assay pipeline
- **`config_loader.py`** — Assay configuration loading
- **`context_builder.py`** — Context construction for LLM prompts
- **`dataset_loader.py`** — Dataset loading for the assay
- **`llm_client.py`** — LLM API client
- **`prompt_template.py`** — Prompt template management
- **`response_cache.py`** — LLM response caching
- **`score_parser.py`** — LLM output score parsing
- **`schema_validator.py`** — Response schema validation
- **`bundle/`** — Assay bundle construction (delta, hypotheses, reporting, summary, validation)

Shared analysis components (`HypothesisAnalyzer`, `ReportGenerator`) live in `shared/evaluation/` and are imported from there.

## Canonical path

`tracks/reasoning/`
