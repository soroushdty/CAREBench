# tracks/

This package contains track-specific evaluation pipelines for the LM-ContextProbe evaluation framework. Every track runs the same study design and reports the same named endpoints; they differ in how the model gets to use the patient context.

## Tracks

- **`representation/`** — Track 1: representation learning pipeline (embedding, training, statistical analysis)
- **`reasoning/`** — Track 3: LLM reasoning/assay pipeline (context-shift evaluation using large language models)

Track 2 (adaptation) is planned; see [#13](https://github.com/soroushdty/LM-ContextProbe/issues/13). Each track is a functional analogue of one way a physician could reach a context-dependent judgment: general clinical knowledge (Track 1), learning from supervised case experience (Track 2), or deliberating at decision time (Track 3).

## Usage

Each track is a self-contained pipeline. New code should import from the canonical track paths:

```python
from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline
from tracks.reasoning.run_assay import main as run_assay
```

## Canonical path

`tracks/`
