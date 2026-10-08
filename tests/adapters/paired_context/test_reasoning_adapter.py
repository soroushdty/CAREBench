"""Tests for adapters/paired_context/reasoning_adapter.py — PairedContextReasoningAdapter.

Includes a dry-run smoke test that verifies data flows through the adapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adapters.paired_context.reasoning_adapter import PairedContextReasoningAdapter, PairedDataset
from adapters.paired_context import labels as paired_context_labels


CLASSES = paired_context_labels.display_names()


class TestReasoningAdapterSmoke:
    """Smoke test: load through adapter and verify DTO shape."""

    def test_load_reasoning_dataset(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        ds = adapter.load_reasoning_dataset()

        assert isinstance(ds, PairedDataset)
        assert ds.survey_consensus.shape[1] == 10
        assert ds.interview_consensus.shape[1] == 10
        assert ds.delta_physician.shape == ds.survey_consensus.shape
        assert len(ds.patient_ids) == len(ds.item_texts)
        assert len(ds.category_names) == 10

    def test_delta_is_interview_minus_survey(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        ds = adapter.load_reasoning_dataset()

        expected_delta = ds.interview_consensus - ds.survey_consensus
        np.testing.assert_array_almost_equal(ds.delta_physician, expected_delta)

    def test_consensus_values_in_unit_interval(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        ds = adapter.load_reasoning_dataset()

        assert np.all(ds.survey_consensus >= 0.0)
        assert np.all(ds.survey_consensus <= 1.0)
        assert np.all(ds.interview_consensus >= 0.0)
        assert np.all(ds.interview_consensus <= 1.0)


class TestManifest:
    """Verify manifest output."""

    def test_manifest_has_all_sections(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        m = adapter.manifest()
        assert "adapter" in m
        assert "dataset" in m
        assert "context" in m
        assert "reference" in m
        assert "label_mapping" in m
        assert "column_map" in m

    def test_name(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        assert adapter.name == "paired_context_reasoning"


class TestAccessors:
    """Verify composed adapter accessors."""

    def test_context_adapter_accessible(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        ctx = adapter.context_adapter
        assert ctx.name == "paired_context_context"

    def test_column_map_accessible(self, synthetic_excel: Path, summaries_file: Path):
        adapter = PairedContextReasoningAdapter(
            dataset_path=synthetic_excel,
            patient_summaries_path=summaries_file,
            class_cols=CLASSES,
        )
        assert adapter.column_map.context_entity_col == "Patient"
