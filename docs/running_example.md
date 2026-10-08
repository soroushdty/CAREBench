# Running the Example

This guide gives commands, prerequisites, expected outputs, and troubleshooting for running CAREBench on the bundled synthetic example dataset (`examples/synthetic/`). To run on your own data, see [`data_format.md`](data_format.md).

## Commands

### Track 3 — Reasoning (dry run, no LLM calls)

```bash
python main.py --track reasoning -- --config configs/assay_config.yaml --dry_run --model dry_run_example
```

### Track 3 — Reasoning (real LLM)

```bash
export HF_TOKEN=your_huggingface_token
python main.py --track reasoning -- --config configs/assay_config.yaml --model meta-llama/Llama-3.1-8B-Instruct
```

### Track 1 — Representation (full training run)

```bash
python main.py --track representation -- --config configs/main_config.yaml
```

The first run downloads the `emilyalsentzer/Bio_ClinicalBERT` embedding model (~440 MB) from Hugging Face.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | See `.python-version` |
| Dependencies | `python -m pip install -r requirements.txt` |
| Dataset workbook | Path declared in config (`examples/synthetic/dataset.xlsx` by default) |
| Patient summaries | Path declared in config (`examples/synthetic/patient_summaries.json` by default) |
| Mapping file | Path declared in config (`examples/synthetic/mapping.json` by default) |
| `PYTHONHASHSEED=42` | Set as an environment variable before Python starts, for reproducibility |
| HF token | Only needed for real LLM inference (Track 3 without `--dry_run`) |

## Expected Outputs

### Track 1 Outputs

Under `output/<run_id>/`:

| Artifact | Description |
|----------|-------------|
| `run_manifest.json` | Run metadata (run_id, timestamp, track, config, strategy) |
| `adapter_manifest.json` | Adapter provenance (name, paths, label space, row counts) |
| `resolved_config.yaml` | Full resolved configuration as used |
| `artifact_index.json` | Maps canonical roles to actual file paths |
| `model/` | Trained ensemble artifacts, checkpoints |
| `metrics/` | Per-output-dimension metrics CSVs |
| `predictions/` | Prediction arrays (context-free, correct-context if enabled) |

### Track 3 Outputs

Under `output/assay/<run_id>/`:

| Artifact | Description |
|----------|-------------|
| `run_manifest.json` | Run metadata (run_id, timestamp, track, model, dry_run flag) |
| `adapter_manifest.json` | Adapter provenance |
| `artifact_index.json` | Maps canonical roles to actual file paths |
| `scores/<model>/context_free_scores.csv` | Context-free condition scores |
| `scores/<model>/correct_context_scores.csv` | Correct-context condition scores |
| `scores/<model>/shuffled_context_scores.csv` | Shuffled-context condition scores |
| `reports/<model>/analysis_report.md` | Hypothesis analysis report (H1–H4) |

## Interpretation Caveats

> **The example dataset is synthetic.** Its patients, snapshots and labels are invented by `scripts/make_synthetic_dataset.py`. Results on it only show that the pipeline works and carry no scientific meaning.

> **Dry-run outputs are mock.** They come from a deterministic synthetic backend for pipeline validation only. Never report them as LLM performance.

> **Claims are scoped to the evaluated cohort.** Results describe the patients and reference observers in the dataset you run on. They are not reported as Brier score, F1, AUROC, or prediction accuracy.

## Troubleshooting

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| `FileNotFoundError` for the dataset | Dataset file missing | Put the dataset at the path declared in config, or update the config path |
| `KeyError` on a privacy category | Label mismatch between config and adapter | Check that the `classes` in the config match `adapters/paired_context/labels.py`; run the label tests |
| `LabelLeakageError` | An item's text appears in its own patient's snapshot | Reword the snapshot, or set `skip_label_leakage_check: true` if this is intended |
| Score CSV has zero rows | Dataset loading silently dropped rows | Check `adapter_manifest.json` for row counts; enable verbose logging |
| All-NaN score column | Category name mismatch | Check that the canonical keys in the config match `adapters/paired_context/labels.py` |
| `ConfigError: Missing required...` | Config YAML incomplete | Compare against `configs/assay_config.yaml` for required keys |
