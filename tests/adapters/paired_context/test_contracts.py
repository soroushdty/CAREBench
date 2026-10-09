"""Tests for shared adapter protocol satisfaction.

Verifies that:
1. Fake adapters can satisfy protocols without inheriting from paired-context adapter classes.
2. paired-context adapters satisfy the shared protocols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shared.adapters.base import (
    ContextAdapter,
    DatasetAdapter,
    ReasoningDatasetAdapter,
    ReferenceAdapter,
)


# ---------------------------------------------------------------------------
# Fake adapters (no paired-context adapter inheritance)
# ---------------------------------------------------------------------------


class FakeDatasetAdapter:
    """Minimal fake satisfying DatasetAdapter protocol."""

    @property
    def name(self) -> str:
        return "fake_dataset"

    def manifest(self) -> dict[str, Any]:
        return {"adapter": "fake"}


class FakeReasoningAdapter:
    """Minimal fake satisfying ReasoningDatasetAdapter protocol."""

    @property
    def name(self) -> str:
        return "fake_reasoning"

    def manifest(self) -> dict[str, Any]:
        return {"adapter": "fake_reasoning"}

    def load_reasoning_dataset(self) -> Any:
        return {"records": []}


class FakeContextAdapter:
    """Minimal fake satisfying ContextAdapter protocol."""

    @property
    def name(self) -> str:
        return "fake_context"

    def manifest(self) -> dict[str, Any]:
        return {"adapter": "fake_context"}

    def load_context_records(self) -> dict[str, Any]:
        return {"entity_1": {"summary": "test"}}


class FakeReferenceAdapter:
    """Minimal fake satisfying ReferenceAdapter protocol."""

    @property
    def name(self) -> str:
        return "fake_reference"

    def manifest(self) -> dict[str, Any]:
        return {"adapter": "fake_reference"}

    def aggregate(self, df: Any, *, class_cols: list[str]) -> Any:
        return df


# ---------------------------------------------------------------------------
# Tests: Fakes satisfy protocols
# ---------------------------------------------------------------------------


class TestFakesSatisfyProtocols:
    """Verify fake adapters satisfy shared protocols without paired-context adapter inheritance."""

    def test_fake_dataset_adapter(self):
        fake = FakeDatasetAdapter()
        assert isinstance(fake, DatasetAdapter)

    def test_fake_reasoning_adapter(self):
        fake = FakeReasoningAdapter()
        assert isinstance(fake, ReasoningDatasetAdapter)

    def test_fake_context_adapter(self):
        fake = FakeContextAdapter()
        assert isinstance(fake, ContextAdapter)

    def test_fake_reference_adapter(self):
        fake = FakeReferenceAdapter()
        assert isinstance(fake, ReferenceAdapter)


# ---------------------------------------------------------------------------
# Tests: paired-context adapters satisfy protocols
# ---------------------------------------------------------------------------


class TestPairedContextAdaptersSatisfyProtocols:
    """Verify paired-context adapters satisfy the shared protocols."""

    def test_paired_context_dataset_adapter(self, synthetic_excel: Path):
        from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter
        from adapters.paired_context import labels

        adapter = PairedContextDatasetAdapter(
            dataset_path=synthetic_excel,
            class_cols=labels.display_names(),
        )
        assert isinstance(adapter, DatasetAdapter)

    def test_paired_context_reasoning_adapter(self, synthetic_excel: Path, summaries_file: Path):
        from adapters.paired_context.reasoning_adapter import PairedContextReasoningAdapter
        from adapters.paired_context import labels

        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=labels.display_names(),
        )
        assert isinstance(adapter, ReasoningDatasetAdapter)

    def test_paired_context_context_adapter(self, summaries_file: Path):
        from adapters.paired_context.context_adapter import PairedContextContextAdapter

        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        assert isinstance(adapter, ContextAdapter)

    def test_paired_context_reference_adapter(self):
        from adapters.paired_context.reference_adapter import PairedContextReferenceAdapter

        adapter = PairedContextReferenceAdapter()
        assert isinstance(adapter, ReferenceAdapter)
