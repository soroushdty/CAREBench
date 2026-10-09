"""Shared fixtures for artifact contract tests.

Provides minimal synthetic runs that exercise manifest emission
without requiring real training or LLM inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def expected_label_space() -> dict[str, str]:
    """Return the default label space."""
    from adapters.paired_context import labels as paired_context_labels

    return paired_context_labels.label_manifest()


@pytest.fixture
def minimal_track1_run(tmp_path: Path) -> Path:
    """Run a minimal Track 1 pipeline via the framework with a stub strategy.

    Returns the output directory containing manifests and artifacts.
    """
    from tracks.representation.framework import RepresentationTrack
    from tracks.representation.strategies.base import RepresentationDataset
    from adapters.paired_context import labels as paired_context_labels

    output_dir = tmp_path / "track1_output"
    output_dir.mkdir()

    # Build a tiny synthetic dataset
    n_train, n_test, n_dim = 10, 5, 10
    emb_dim = 32
    dims = paired_context_labels.keys()

    dataset = RepresentationDataset(
        X_train=np.random.randn(n_train, emb_dim).astype(np.float32),
        Y_train=np.random.rand(n_train, n_dim).astype(np.float32),
        patient_ids_train=np.array([f"P{i}" for i in range(n_train)]),
        X_test=np.random.randn(n_test, emb_dim).astype(np.float32),
        Y_test=np.random.rand(n_test, n_dim).astype(np.float32),
        patient_ids_test=np.array([f"P{i}" for i in range(n_test)]),
        output_dimensions=dims,
    )

    # Stub adapter that returns our synthetic dataset
    class _StubAdapter:
        @property
        def name(self) -> str:
            return "paired_context_representation"

        def load_representation_dataset(self, config: dict[str, Any]) -> RepresentationDataset:
            return dataset

    # Stub strategy that returns minimal results
    class _StubStrategy:
        def fit_and_evaluate(
            self, ds: RepresentationDataset, config: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "train": {"loss": 0.1},
                "test": {"loss": 0.2},
                "test_probs_cf": np.random.rand(n_test, n_dim).astype(np.float32),
            }

    config: dict[str, Any] = {
        "output_dir": str(output_dir),
        "RUN_ID": "test_run_001",
        "_config_path": "configs/main_config.yaml",
    }

    track = RepresentationTrack(
        adapter=_StubAdapter(),
        strategy=_StubStrategy(),
        artifact_writers=[],
    )
    track.run(config)

    return output_dir


@pytest.fixture
def minimal_track3_run(tmp_path: Path) -> Path:
    """Run a minimal Track 3 manifest emission with synthetic data.

    This does NOT run the full assay pipeline (which requires real data).
    Instead it directly calls the manifest helpers to produce the expected
    artifact tree, simulating what a successful dry-run would produce.

    Returns the output directory containing manifests and artifacts.
    """
    from shared.prerequisites.manifests import (
        write_adapter_manifest,
        write_artifact_index,
        write_resolved_config,
        write_run_manifest,
    )
    from adapters.paired_context import labels as paired_context_labels

    output_dir = tmp_path / "track3_output"
    output_dir.mkdir()

    # Write manifests
    write_run_manifest(
        output_dir,
        run_id="test_run_003",
        track="reasoning",
        entry_command="python main.py --track reasoning -- --dry_run",
        config_path="configs/assay_config.yaml",
        dry_run=True,
        extra={"model_ids": ["dry_run_example"]},
    )

    write_adapter_manifest(
        output_dir,
        adapter_name="paired_context_reasoning",
        adapter_class="adapters.paired_context.PairedContextReasoningAdapter",
        label_space=paired_context_labels.label_manifest(),
        dataset_path="examples/synthetic/dataset.xlsx",
        patient_summaries_path="examples/synthetic/patient_summaries.json",
    )

    # Create synthetic score CSVs
    import pandas as pd

    canonical_keys = paired_context_labels.keys()
    n_rows = 6
    scores_dir = output_dir / "scores" / "dry_run_example"
    scores_dir.mkdir(parents=True)

    for condition in ("context_free", "correct_context", "shuffled_context"):
        rows = []
        for i in range(n_rows):
            row = {
                "patient_id": f"P{i % 3}",
                "item_text": f"item_{i}",
            }
            for key in canonical_keys:
                row[key] = float(np.random.rand())
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(scores_dir / f"{condition}_scores.csv", index=False)

    # Create synthetic report
    reports_dir = output_dir / "reports" / "dry_run_example"
    reports_dir.mkdir(parents=True)
    report_path = reports_dir / "analysis_report.md"
    report_path.write_text(
        "# Analysis Report\n\nDry-run results for Behavioral health and other dimensions.\n",
        encoding="utf-8",
    )

    # Write artifact index
    write_artifact_index(
        output_dir,
        {
            "track3.scores.context_free": "scores/dry_run_example/context_free_scores.csv",
            "track3.scores.correct_context": "scores/dry_run_example/correct_context_scores.csv",
            "track3.scores.shuffled_context": "scores/dry_run_example/shuffled_context_scores.csv",
            "track3.hypothesis_summary": "reports/dry_run_example/analysis_report.md",
        },
    )

    # Write resolved config
    write_resolved_config(
        output_dir,
        {
            "backend": "dry_run",
            "model_ids": ["dry_run_example"],
            "data": {
                "dataset_path": "examples/synthetic/dataset.xlsx",
                "patient_summaries_path": "examples/synthetic/patient_summaries.json",
            },
        },
    )

    return output_dir
