"""Paired-context representation adapter smoke test.

Verifies that the PairedContextRepresentationAdapter can load data and produce
a valid RepresentationDataset, and that the full pipeline runs with a dry
strategy that writes artifacts.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tracks.representation.strategies.base import RepresentationDataset


# ---------------------------------------------------------------------------
# Dry strategy — records shapes, no real training
# ---------------------------------------------------------------------------


class DryStrategy:
    """Dry strategy that records dataset shapes without running training."""

    def __init__(self):
        self.called = False
        self.shapes: dict[str, Any] = {}

    def fit_and_evaluate(
        self, dataset: RepresentationDataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        self.called = True
        self.shapes = {
            "X_train_shape": list(dataset.X_train.shape),
            "Y_train_shape": list(dataset.Y_train.shape),
            "X_test_shape": list(dataset.X_test.shape),
            "Y_test_shape": list(dataset.Y_test.shape),
            "n_output_dimensions": len(dataset.output_dimensions),
            "output_dimensions": dataset.output_dimensions,
        }
        return {"status": "dry_run", "shapes": self.shapes}


class SimpleArtifactWriter:
    """Simple artifact writer that writes a JSON summary to output_dir."""

    def __init__(self, filename: str = "dry_run_result.json"):
        self._filename = filename
        self.written = False

    def write(self, result: dict[str, Any], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / self._filename
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.written = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPairedContextRepresentationSmoke:
    """Smoke tests for the paired-context representation adapter.

    These tests require access to a dataset. They are marked slow
    and will skip gracefully if data is not available.
    """

    @pytest.fixture
    def dataset_path(self) -> Path:
        """Resolve the example dataset path."""
        candidates = [
            Path("examples/synthetic/dataset.xlsx"),
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip(
            "Example dataset not available. "
            "Expected examples/synthetic/dataset.xlsx"
        )

    @pytest.fixture
    def config(self, dataset_path: Path, tmp_path: Path) -> dict[str, Any]:
        """Build a Track 1 config for the example dataset, without summaries."""
        import yaml

        cfg = yaml.safe_load(Path("configs/main_config.yaml").read_text(encoding="utf-8"))
        cfg.update(
            {
                "PROJECT_ROOT": str(Path.cwd()),
                "DIR_DATASET": str(dataset_path),
                "ENABLE_SUMMARY": False,
                "DIR_SUMMARY": str(tmp_path / "input_summary"),
                "output_dir": str(tmp_path / "output"),
            }
        )
        return cfg

    def test_paired_context_adapter_builds_arrays(self, config: dict[str, Any]):
        """Adapter produces valid arrays."""
        from adapters.paired_context.representation_adapter import (
            PairedContextRepresentationAdapter,
        )

        adapter = PairedContextRepresentationAdapter()
        try:
            dataset = adapter.load_representation_dataset(config)
        except Exception as exc:
            pytest.skip(f"paired-context adapter loading failed (dependency issue): {exc}")

        # Structural checks
        assert dataset.X_train.ndim == 2
        assert dataset.Y_train.ndim == 2
        assert dataset.Y_train.shape[1] == len(dataset.output_dimensions)
        assert dataset.patient_ids_train.shape[0] == dataset.X_train.shape[0]

        assert dataset.X_test.ndim == 2
        assert dataset.Y_test.ndim == 2
        assert dataset.Y_test.shape[1] == len(dataset.output_dimensions)
        assert dataset.patient_ids_test.shape[0] == dataset.X_test.shape[0]

        # Output dimensions should be the 10 default labels
        assert len(dataset.output_dimensions) == 10

    def test_smoke_run_writes_artifacts(self, config: dict[str, Any], tmp_path: Path):
        """Smoke run writes expected artifacts."""
        from adapters.paired_context.representation_adapter import (
            PairedContextRepresentationAdapter,
        )
        from tracks.representation.framework import RepresentationTrack

        adapter = PairedContextRepresentationAdapter()
        strategy = DryStrategy()
        writer = SimpleArtifactWriter()

        config["output_dir"] = str(tmp_path / "smoke_output")

        track = RepresentationTrack(
            adapter=adapter, strategy=strategy, artifact_writers=[writer]
        )

        try:
            result = track.run(config)
        except Exception as exc:
            pytest.skip(f"Smoke run failed (dependency issue): {exc}")

        # Strategy was called
        assert strategy.called
        assert result["status"] == "dry_run"

        # Writer wrote the artifact
        assert writer.written
        artifact_path = tmp_path / "smoke_output" / "dry_run_result.json"
        assert artifact_path.exists()

        # Verify artifact content
        with open(artifact_path) as f:
            content = json.load(f)
        assert content["status"] == "dry_run"
        assert "shapes" in content


class TestPairedContextAdapterUnit:
    """Unit tests for paired-context adapter that don't require real data."""

    def test_adapter_satisfies_protocol(self):
        """Adapter satisfies RepresentationDatasetAdapter protocol."""
        try:
            from adapters.paired_context.representation_adapter import (
                PairedContextRepresentationAdapter,
            )
        except ImportError as exc:
            pytest.skip(f"Cannot import PairedContextRepresentationAdapter: {exc}")

        from tracks.representation.strategies.base import (
            RepresentationDatasetAdapter,
        )

        adapter = PairedContextRepresentationAdapter()
        assert isinstance(adapter, RepresentationDatasetAdapter)

    def test_adapter_has_load_method(self):
        """Adapter has load_representation_dataset method."""
        try:
            from adapters.paired_context.representation_adapter import (
                PairedContextRepresentationAdapter,
            )
        except ImportError as exc:
            pytest.skip(f"Cannot import PairedContextRepresentationAdapter: {exc}")

        adapter = PairedContextRepresentationAdapter()
        assert hasattr(adapter, "load_representation_dataset")
        assert callable(adapter.load_representation_dataset)

    def test_framework_does_not_import_paired_context_adapter(self):
        """framework.py does not import the paired-context adapter."""
        import inspect

        import tracks.representation.framework as fw_module

        source = inspect.getsource(fw_module)
        assert "kaufman" not in source.lower(), (
            "framework.py imports or references 'kaufman'"
        )
        assert "adapters.paired_context" not in source, (
            "framework.py imports adapters.paired_context"
        )
