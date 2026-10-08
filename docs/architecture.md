# Architecture

This document describes the package boundaries, dependency rules, and canonical injection flows for the CAREBench framework.

## Package Boundaries

| Package | MAY import | MUST NOT import |
|---------|-----------|-----------------|
| `shared/` | stdlib, third-party | `tracks/`, `adapters/`, `main` |
| `adapters/paired_context/` | stdlib, third-party, `shared/` | `tracks/`, other `adapters/*` |
| `tracks/representation/` | stdlib, third-party, `shared/`, `adapters/` (via injection only in `framework.py`) | `tracks.reasoning`, direct adapter construction outside runner |
| `tracks/reasoning/` | stdlib, third-party, `shared/`, `adapters/` (via injection in `run_assay.py`) | `tracks.representation`, direct adapter construction outside runner |
| `main.py` | `tracks/*`, `adapters/*`, `shared/*` | — |

## Dependency Direction Diagram

```
main.py
  │
  ├──▶ tracks/representation/
  │       ├──▶ shared/
  │       └──▶ adapters/paired_context/  (injected into RepresentationTrack; see below)
  │
  ├──▶ tracks/reasoning/
  │       ├──▶ shared/
  │       └──▶ adapters/paired_context/  (injected via run_assay.py)
  │
  └──▶ adapters/paired_context/
          └──▶ shared/
```

**Rule:** Arrows point only **downward**. No upward imports. No sideways imports between `tracks/*` children or between `adapters/*` children.

## Adapter Boundary

The adapter boundary is the interface at which dataset-specific knowledge enters the system:

- Dataset-specific column names, sheet names, label display names, physician-pair logic, and survey/interview semantics live **only** in `adapters/<dataset>/`.
- `shared/` and `tracks/` receive canonical objects (`RepresentationDataset`, `PairedDataset`, scored records with canonical `output_dimension` keys) and never branch on dataset identity.
- No module in `shared/` or `tracks/` contains `if adapter == "<dataset>"` or equivalent conditional logic.

## Track 1 Flow

The CLI builds a dataset adapter and a training strategy from config and runs them through `RepresentationTrack`:

```
1. CLI: python main.py --track representation -- --config configs/main_config.yaml
2. main.py dispatches to tracks.representation.runner.main()
3. runner.py:
   a. load_prerequisites() resolves configs and runs the reproducibility guards
   b. builds the adapter named by `adapter` (default: paired_context) and the
      strategy named by `strategy` (default: existing_ensemble)
   c. RepresentationTrack(adapter, strategy).run(config)
4. RepresentationTrack.run(config):
   a. adapter.load_representation_dataset(config) → RepresentationDataset
      PairedContextRepresentationAdapter:
        - reads the workbook via PairedContextDatasetAdapter.load_sheets()
          and the context records via PairedContextContextAdapter
          (the same loaders Track 3 uses)
        - shared.preprocessing.preprocess(cfg, dfs=sheets): item
          standardization, physician aggregation, input summaries
        - compute_embeddings() embeds the items
   b. strategy.fit_and_evaluate(dataset, config) → result dict
      ExistingEnsembleTrainingStrategy:
        - builds Stage 2 context vectors from the context records
        - train_ensemble_pipeline(): nested LOPO-CV training, calibration,
          Stage 2 fusion
        - run_statistical_analysis(): H1/H2, ICC, ablation, and calibration
          outputs (when statistical_analysis.enabled)
   c. Each writer.write(result, output_dir) → persists artifacts
   d. Framework emits run_manifest.json, adapter_manifest.json,
      artifact_index.json, and resolved_config.yaml in the run folder
```

Custom adapters and strategies can be injected into `RepresentationTrack` directly, or registered in `_ADAPTERS` / `_STRATEGIES` in `tracks/representation/runner.py` to make them selectable from config.

### Strategy Protocol

Any class implementing `fit_and_evaluate(dataset: RepresentationDataset, config: dict) -> dict` can be injected as a strategy. See `tracks/representation/strategies/base.py` for the formal protocol definition.

## Track 3 Adapter/Data Flow

```
1. CLI: python main.py --track reasoning -- --config configs/assay_config.yaml --dry_run
2. main.py dispatches to tracks.reasoning.run_assay.main()
3. run_assay.py:
   a. Loads and validates config from YAML
   b. DatasetLoader loads the paired dataset (via PairedContextReasoningAdapter internally)
   c. ContextBuilder builds correct-context and shuffled-context patient snapshots
   d. LLMClient (or DryRunClient) runs inference for each condition
   e. ScoreParser validates JSON against the canonical privacy-category schema
   f. HypothesisAnalyzer computes H1–H4 results
   g. ReportGenerator produces the markdown report
4. Runner emits run_manifest.json, adapter_manifest.json, artifact_index.json,
   condition score CSVs, hypothesis summary, and markdown report
```
