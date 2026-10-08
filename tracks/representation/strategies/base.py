"""Track 1 framework contracts — protocols and data containers.

Defines protocol contracts for training strategies and artifact writers.
The data container (`RepresentationDataset`) and the dataset-adapter protocol
(`RepresentationDatasetAdapter`) live in ``shared.adapters.base`` so that
dataset adapters can implement them without importing ``tracks``; they are
re-exported here for convenience.

Canonical path: tracks/representation/strategies/base.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from shared.adapters.base import RepresentationDataset, RepresentationDatasetAdapter

__all__ = [
    "ArtifactWriter",
    "RepresentationDataset",
    "RepresentationDatasetAdapter",
    "RepresentationRunSpec",
    "TrainingStrategy",
]


# ---------------------------------------------------------------------------
# Runtime metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentationRunSpec:
    """Runtime metadata for a representation track run."""

    run_id: str
    output_dir: Path
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingStrategy(Protocol):
    """Protocol for training strategies."""

    def fit_and_evaluate(
        self, dataset: RepresentationDataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run training/evaluation and return a result dict."""
        ...


@runtime_checkable
class ArtifactWriter(Protocol):
    """Protocol for artifact writers that persist strategy results."""

    def write(self, result: dict[str, Any], output_dir: Path) -> None:
        """Write artifacts from the strategy result to output_dir."""
        ...
