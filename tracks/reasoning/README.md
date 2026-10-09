# tracks/reasoning/

Track 3: LLM reasoning pipeline for context-shift evaluation.

This track covers the full LLM-based reasoning workflow, including:

- **`run_reasoning.py`** — Main entry point for the Track 3 pipeline
- **`config_loader.py`** — Reasoning configuration loading
- **`context_builder.py`** — Context construction for LLM prompts
- **`dataset_loader.py`** — Dataset loading for Track 3
- **`llm_client.py`** — LLM API client
- **`prompt_template.py`** — Prompt template management
- **`response_cache.py`** — LLM response caching
- **`score_parser.py`** — LLM output score parsing
- **`schema_validator.py`** — Response schema validation
- **`bundle/`** — Analysis bundle construction (delta, hypotheses, reporting, summary, validation)

Shared analysis components (`HypothesisAnalyzer`, `ReportGenerator`) live in `shared/evaluation/` and are imported from there.

## Canonical path

`tracks/reasoning/`
