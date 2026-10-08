"""Tests for adapters/paired_context/reference_adapter.py — PairedContextReferenceAdapter.

Verifies aggregation shapes, canonical output-dimension keys, and observer
count enforcement.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context.reference_adapter import PairedContextReferenceAdapter
from adapters.paired_context import labels as paired_context_labels


CLASSES = paired_context_labels.display_names()


def _make_raw_df(
    patients: list[int],
    items: list[str],
    physicians: list[int],
    label_value: float = 0.0,
) -> pd.DataFrame:
    """Build a raw per-observer DataFrame."""
    rows = []
    for pat, item in zip(patients, items):
        for phys in physicians:
            row = {"Patient": pat, "Physician": phys, "Item": item}
            for cls in CLASSES:
                row[cls] = label_value
            rows.append(row)
    return pd.DataFrame(rows)


class TestAggregationShape:
    """Verify output shapes after aggregation."""

    def test_two_physicians_aggregated_to_one_row_per_pair(self):
        """Two observer rows per (entity, task_instance) collapse to one."""
        patients = [1, 1, 2, 2]
        items = ["a", "b", "a", "b"]
        df = _make_raw_df(patients, items, [1, 2])

        adapter = PairedContextReferenceAdapter()
        result = adapter.aggregate(df, class_cols=CLASSES)

        # 4 unique (patient, item) pairs → 4 rows
        assert len(result) == 4

    def test_class_columns_preserved(self):
        """All class columns remain in aggregated output."""
        df = _make_raw_df([1], ["a"], [1, 2])
        adapter = PairedContextReferenceAdapter()
        result = adapter.aggregate(df, class_cols=CLASSES)

        for cls in CLASSES:
            assert cls in result.columns

    def test_output_is_mean_of_observers(self):
        """Aggregated values are the mean of observer labels."""
        rows = [
            {"Patient": 1, "Physician": 1, "Item": "a", **{c: 0.0 for c in CLASSES}},
            {"Patient": 1, "Physician": 2, "Item": "a", **{c: 1.0 for c in CLASSES}},
        ]
        df = pd.DataFrame(rows)

        adapter = PairedContextReferenceAdapter()
        result = adapter.aggregate(df, class_cols=CLASSES)

        # Mean of 0 and 1 is 0.5
        for cls in CLASSES:
            assert result[cls].iloc[0] == pytest.approx(0.5)


class TestObserverCountEnforcement:
    """Verify the two-observer assumption lives in the adapter."""

    def test_default_expects_two(self):
        adapter = PairedContextReferenceAdapter()
        assert adapter._column_map.expected_reference_observer_count == 2

    def test_custom_observer_count(self):
        cmap = PairedContextColumnMap(expected_reference_observer_count=3)
        adapter = PairedContextReferenceAdapter(column_map=cmap)
        assert adapter._column_map.expected_reference_observer_count == 3


class TestManifest:
    """Verify manifest output."""

    def test_manifest_fields(self):
        adapter = PairedContextReferenceAdapter()
        m = adapter.manifest()
        assert m["adapter"] == "paired_context_reference"
        assert m["aggregation_policy"] == "paired_reference_mean"
        assert m["expected_observer_count"] == 2
        assert "context_entity_col" in m
        assert "reference_observer_col" in m
        assert "task_instance_col" in m

    def test_name(self):
        adapter = PairedContextReferenceAdapter()
        assert adapter.name == "paired_context_reference"


class TestCanonicalOutputDimensions:
    """Verify output dimensions use the default label space."""

    def test_aggregated_columns_match_label_space(self):
        """Class columns in output match the requested class_cols."""
        df = _make_raw_df([1], ["a"], [1, 2])
        adapter = PairedContextReferenceAdapter()
        result = adapter.aggregate(df, class_cols=CLASSES)

        for cls in CLASSES:
            assert cls in result.columns
