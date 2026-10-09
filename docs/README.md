# Documentation Index

This directory contains canonical documentation for LM-ContextProbe, an evaluation framework for testing whether language models change their judgments with patient context the way physicians do. The framework is a study design, a set of endpoints with statistical tests, and pipelines that run them on any configured model and dataset. It is not a benchmark: it ships no fixed dataset or leaderboard.

| Document | Description |
|----------|-------------|
| [`architecture.md`](architecture.md) | Package boundaries, dependency rules, and injection flows |
| [`data_format.md`](data_format.md) | Input files expected by the built-in paired-context adapter |
| [`adapters.md`](adapters.md) | Adapter responsibilities, the paired-context adapter, adding new adapters |
| [`running_example.md`](running_example.md) | Commands, outputs, and caveats for the synthetic example |
| [`artifact_contracts.md`](artifact_contracts.md) | Required output artifacts and their schemas |
| [`data_model.md`](data_model.md) | Canonical vocabulary, machine keys vs display labels |
| [`methodology.md`](methodology.md) | Endpoints, statistical tests, protocol version, design rationale, and known limitations |
| [`releasing.md`](releasing.md) | Release checklist: protocol version, Zenodo archiving and DOI |
