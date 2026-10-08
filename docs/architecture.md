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

## Track 1 Flows

Track 1 currently has two entry paths:

**CLI pipeline** (what `main.py` runs):

```
1. CLI: python main.py --track representation -- --config configs/main_config.yaml
2. main.py dispatches to tracks.representation.runner.main()
3. runner.py:
   a. load_prerequisites() resolves configs and runs the reproducibility guards
   b. shared.preprocessing.preprocess(cfg) loads the workbook (column/sheet names from config)
   c. compute_embeddings() embeds items (and patient contexts for Stage 2)
   d. train_ensemble_pipeline() runs nested LOPO-CV training, calibration, and Stage 2 fusion
   e. run_statistical_analysis() writes H1/H2, ICC, ablation, and calibration outputs
```

**Programmatic framework** (`RepresentationTrack`, for custom adapters and strategies):

```
RepresentationTrack(adapter, strategy, writers).run(config):
   a. adapter.load_representation_dataset(config) → RepresentationDataset
      (e.g. PairedContextRepresentationAdapter from adapters.paired_context)
   b. strategy.fit_and_evaluate(dataset, config) → result dict
      (e.g. ExistingEnsembleTrainingStrategy)
   c. Each writer.write(result, output_dir) → persists artifacts
   d. Framework emits run_manifest.json, adapter_manifest.json, artifact_index.json
```

Routing the CLI pipeline through `RepresentationTrack` is on the roadmap.

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
