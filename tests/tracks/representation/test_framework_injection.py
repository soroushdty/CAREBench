"""Framework injection tests for Track 1 representation.

Verifies that the RepresentationTrack orchestrator works with fake (in-memory)
adapter, strategy, and writer components — proving real dependency injection
without accessing dataset files or heavyweight ML dependencies.

"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tracks.representation.framework import RepresentationTrack
from tracks.representation.strategies.base import RepresentationDataset


# ---------------------------------------------------------------------------
# Fake components
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Fake adapter that returns a small synthetic RepresentationDataset."""

    def __init__(self, n_train: int = 10, n_test: int = 5, n_dims: int = 3):
        self.n_train = n_train
        self.n_test = n_test
        self.n_dims = n_dims
        self.called = False

    def load_representation_dataset(
        self, config: dict[str, Any]
    ) -> RepresentationDataset:
        self.called = True
        rng = np.random.default_rng(42)
        return RepresentationDataset(
            X_train=rng.random((self.n_train, 8), dtype=np.float32),
            Y_train=rng.integers(0, 2, (self.n_train, self.n_dims)).astype(
                np.float32
            ),
            patient_ids_train=np.array(
                [f"P{i}" for i in range(self.n_train)]
            ),
            X_test=rng.random((self.n_test, 8), dtype=np.float32),
            Y_test=rng.integers(0, 2, (self.n_test, self.n_dims)).astype(
                np.float32
            ),
            patient_ids_test=np.array(
                [f"P{i}" for i in range(self.n_test)]
            ),
            output_dimensions=[f"dim_{i}" for i in range(self.n_dims)],
        )


class FakeStrategy:
    """Fake strategy that records its call and returns a result dict."""

    def __init__(self):
        self.called = False
        self.received_dataset: RepresentationDataset | None = None
        self.received_config: dict[str, Any] | None = None

    def fit_and_evaluate(
        self, dataset: RepresentationDataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        self.called = True
        self.received_dataset = dataset
        self.received_config = config
        return {"status": "ok", "metrics": {"macro_f1": 0.85}}


class FakeWriter:
    """Fake artifact writer that records its call."""

    def __init__(self):
        self.called = False
        self.received_result: dict[str, Any] | None = None
        self.received_output_dir: Path | None = None

    def write(self, result: dict[str, Any], output_dir: Path) -> None:
        self.called = True
        self.received_result = result
        self.received_output_dir = output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFrameworkInjection:
    """Test suite for framework dependency injection."""

    def test_framework_runs_with_fake_components(self, tmp_path: Path):
        """Framework runs end-to-end with fakes."""
        adapter = FakeAdapter()
        strategy = FakeStrategy()
        writer = FakeWriter()

        track = RepresentationTrack(
            adapter=adapter, strategy=strategy, artifact_writers=[writer]
        )
        config = {"output_dir": str(tmp_path)}
        result = track.run(config)

        # Adapter was called
        assert adapter.called, "Adapter was not called"

        # Strategy was called with the dataset
        assert strategy.called, "Strategy was not called"
        assert strategy.received_dataset is not None
        assert strategy.received_dataset.X_train.shape == (10, 8)
        assert strategy.received_dataset.output_dimensions == [
            "dim_0",
            "dim_1",
            "dim_2",
        ]

        # Writer was called with the result
        assert writer.called, "Writer was not called"
        assert writer.received_result == result
        assert writer.received_output_dir == tmp_path

        # Result is correct
        assert result["status"] == "ok"
        assert result["metrics"]["macro_f1"] == 0.85

    def test_framework_no_dataset_branching(self):
        """framework.py has no dataset-specific branching."""
        source = inspect.getsource(RepresentationTrack)
        # Also get the module source
        import tracks.representation.framework as fw_module

        module_source = inspect.getsource(fw_module)

        prohibited_terms = [
            "kaufman",
            '"Patient"',
            '"Item"',
            '"Physician"',
            '"survey"',
            '"interview"',
            "from funcs",
            "import funcs",
            "from legacy",
            "import legacy",
            "from models",
            "import models",
            "adapters.paired_context",
        ]

        for term in prohibited_terms:
            assert term.lower() not in module_source.lower(), (
                f"Prohibited term '{term}' found in framework.py source"
            )

    def test_multiple_writers_all_called(self, tmp_path: Path):
        """All injected writers are called."""
        adapter = FakeAdapter()
        strategy = FakeStrategy()
        writer1 = FakeWriter()
        writer2 = FakeWriter()

        track = RepresentationTrack(
            adapter=adapter,
            strategy=strategy,
            artifact_writers=[writer1, writer2],
        )
        config = {"output_dir": str(tmp_path)}
        track.run(config)

        assert writer1.called, "Writer 1 was not called"
        assert writer2.called, "Writer 2 was not called"
        assert writer1.received_result == writer2.received_result

    def test_no_writers_still_succeeds(self, tmp_path: Path):
        """Framework works with zero writers."""
        adapter = FakeAdapter()
        strategy = FakeStrategy()

        track = RepresentationTrack(
            adapter=adapter, strategy=strategy, artifact_writers=[]
        )
        config = {"output_dir": str(tmp_path)}
        result = track.run(config)

        assert result["status"] == "ok"
        assert strategy.called

    def test_result_passes_through_from_strategy(self, tmp_path: Path):
        """Strategy result is returned unchanged."""
        adapter = FakeAdapter()
        strategy = FakeStrategy()

        track = RepresentationTrack(adapter=adapter, strategy=strategy)
        config = {"output_dir": str(tmp_path)}
        result = track.run(config)

        assert result == {"status": "ok", "metrics": {"macro_f1": 0.85}}

    def test_run_manifest_records_strategy_cv_splits(self, tmp_path: Path):
        """A strategy's ``cv`` record is copied into run_manifest.json."""
        import json

        cv = {
            "outer": {"scheme": "group_kfold", "n_splits": 2, "seed": 0},
            "inner": {"scheme": "lopo", "n_splits": None, "seed": None},
            "n_outer_folds": 2,
            "folds": [{"fold_idx": 0, "held_out_patient_ids": [1, 3]}],
        }

        class CVStrategy(FakeStrategy):
            def fit_and_evaluate(self, dataset, config):
                return {**super().fit_and_evaluate(dataset, config), "cv": cv}

        track = RepresentationTrack(adapter=FakeAdapter(), strategy=CVStrategy())
        track.run({"output_dir": str(tmp_path)})

        manifest = json.loads((tmp_path / "run_manifest.json").read_text())
        assert manifest["cv"] == cv

    def test_run_manifest_without_strategy_cv(self, tmp_path: Path):
        import json

        track = RepresentationTrack(adapter=FakeAdapter(), strategy=FakeStrategy())
        track.run({"output_dir": str(tmp_path)})

        assert "cv" not in json.loads((tmp_path / "run_manifest.json").read_text())

    def test_output_dir_created(self, tmp_path: Path):
        """Framework creates output_dir if it doesn't exist."""
        adapter = FakeAdapter()
        strategy = FakeStrategy()
        out = tmp_path / "nested" / "output"

        track = RepresentationTrack(adapter=adapter, strategy=strategy)
        config = {"output_dir": str(out)}
        track.run(config)

        assert out.exists()
        assert out.is_dir()

    def test_protocols_are_runtime_checkable(self):
        """Protocols are runtime_checkable."""
        from tracks.representation.strategies.base import (
            ArtifactWriter,
            RepresentationDatasetAdapter,
            TrainingStrategy,
        )

        adapter = FakeAdapter()
        strategy = FakeStrategy()
        writer = FakeWriter()

        assert isinstance(adapter, RepresentationDatasetAdapter)
        assert isinstance(strategy, TrainingStrategy)
        assert isinstance(writer, ArtifactWriter)
