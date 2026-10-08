"""Minimal shared adapter contracts (typing.Protocol).

These protocols define the structural interfaces that dataset-specific adapters
must satisfy. Tracks depend on these stable interfaces rather than concrete
adapter implementations.

Canonical path: shared/adapters/base.py

Design:
    - Uses typing.Protocol for structural subtyping (no inheritance required).
    - Test fakes can satisfy these protocols without inheriting from any
      concrete adapter class.
    - These contracts SHALL NOT import adapters.*, tracks.*, pandas-heavy
      parsing code, or model/training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal dataset adapter contract.

    Every dataset adapter must expose a human-readable name and a manifest
    describing its configuration for reproducibility.
    """

    @property
    def name(self) -> str:
        """Human-readable adapter name (e.g. 'paired_context_dataset')."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Return a reproducibility manifest describing the adapter state.

        The manifest should include dataset path, active column map,
        active label mapping, and any relevant policy information.
        """
        ...


@runtime_checkable
class ReasoningDatasetAdapter(Protocol):
    """Adapter contract for reasoning (Track 3) dataset loading.

    Provides the full pipeline needed to produce reasoning-ready records:
    loading the workbook, aggregating reference observers, and returning
    a DTO accepted by the reasoning track.
    """

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Return a reproducibility manifest."""
        ...

    def load_reasoning_dataset(self) -> Any:
        """Load and return a reasoning-ready dataset DTO.

        The return type is intentionally `Any` at the protocol level to
        avoid coupling the shared contract to a specific DTO class.
        Concrete adapters document their actual return type.
        """
        ...


@runtime_checkable
class ContextAdapter(Protocol):
    """Adapter contract for context record loading.

    Provides access to context records (e.g. patient summaries) by a
    generic entity identifier.
    """

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Return a reproducibility manifest."""
        ...

    def load_context_records(self) -> dict[str, Any]:
        """Load and return context records keyed by context_entity_id.

        Returns a mapping of entity ID → context fields.
        """
        ...


@runtime_checkable
class ReferenceAdapter(Protocol):
    """Adapter contract for reference observer aggregation.

    Wraps dataset-specific aggregation policy (e.g. paired-mean with
    a specific observer count) behind a generic interface.
    """

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Return a reproducibility manifest."""
        ...

    def aggregate(self, df: Any, *, class_cols: list[str]) -> Any:
        """Aggregate reference observer rows into consensus rows.

        Parameters
        ----------
        df : DataFrame-like
            Raw per-observer records.
        class_cols : list[str]
            Label/class column names to aggregate.

        Returns
        -------
        Aggregated DataFrame-like with one row per (entity, task_instance).
        """
        ...


# ---------------------------------------------------------------------------
# Track 1 (representation) data container and adapter protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentationDataset:
    """Canonical data container for the Track 1 framework.

    Required fields
    ---------------
    X_train : np.ndarray
        Training embeddings (n_train, embedding_dim).
    Y_train : np.ndarray
        Training labels (n_train, n_output_dimensions).
    patient_ids_train : np.ndarray
        Per-row context entity IDs for training data.
    X_test : np.ndarray
        Held-out test embeddings (n_test, embedding_dim).
    Y_test : np.ndarray
        Held-out test labels (n_test, n_output_dimensions).
    patient_ids_test : np.ndarray
        Per-row context entity IDs for test data.
    output_dimensions : list[str]
        Ordered list of output dimension names (class labels).

    Optional fields
    ---------------
    item_strings_train : np.ndarray | None
        Item text strings parallel to X_train.
    item_strings_test : np.ndarray | None
        Item text strings parallel to X_test.
    Y_test_context_free : np.ndarray | None
        Reference context-free labels for test data.
    Y_test_correct_context : np.ndarray | None
        Reference correct-context labels for test data.
    context_vectors : dict[str, np.ndarray] | None
        Context vectors keyed by context entity ID (for Stage 2).
    metadata : dict[str, Any]
        Additional metadata (embedding model, config snapshot, etc.).
    """

    # Required
    X_train: np.ndarray
    Y_train: np.ndarray
    patient_ids_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    patient_ids_test: np.ndarray
    output_dimensions: list[str]

    # Optional
    item_strings_train: np.ndarray | None = None
    item_strings_test: np.ndarray | None = None
    Y_test_context_free: np.ndarray | None = None
    Y_test_correct_context: np.ndarray | None = None
    context_vectors: dict[str, np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RepresentationDatasetAdapter(Protocol):
    """Protocol for dataset adapters that produce a RepresentationDataset."""

    def load_representation_dataset(
        self, config: dict[str, Any]
    ) -> RepresentationDataset:
        """Load and return a fully constructed RepresentationDataset."""
        ...
