"""Tests for adapters/paired_context/dataset_adapter.py — PairedContextDatasetAdapter.

Verifies sheet/column validation, triplet correspondence, and manifest output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context.dataset_adapter import (
    DatasetValidationError,
    PairedContextDatasetAdapter,
)
from adapters.paired_context import labels as paired_context_labels


CLASSES = paired_context_labels.display_names()


class TestSheetAndColumnValidation:
    """Verify that required sheets and columns are validated."""

    def test_load_sheets_success(self, synthetic_excel: Path):
        adapter = PairedContextDatasetAdapter(
            dataset_path=synthetic_excel,
            class_cols=CLASSES,
        )
        sheets = adapter.load_sheets()
        assert "train" in sheets
        assert "test" in sheets
        assert "interview" in sheets

    def test_load_sheets_missing_sheet(self, tmp_path: Path):
        """Raise DatasetValidationError when a required sheet is missing."""
        path = tmp_path / "incomplete.xlsx"
        df = pd.DataFrame({"Patient": [1], "Physician": [1], "Item": ["a"]})
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="train", index=False)
            df.to_excel(writer, sheet_name="test", index=False)
            # Missing 'interview' sheet

        adapter = PairedContextDatasetAdapter(
            dataset_path=path,
            class_cols=CLASSES,
        )
        with pytest.raises(DatasetValidationError, match="sheet"):
            adapter.load_sheets()

    def test_load_sheets_missing_column(self, tmp_path: Path):
        """Raise DatasetValidationError when a required column is missing."""
        path = tmp_path / "bad_cols.xlsx"
        # Missing 'Item' column
        df = pd.DataFrame({"Patient": [1], "Physician": [1], "Wrong": ["a"]})
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="train", index=False)
            df.to_excel(writer, sheet_name="test", index=False)
            df.to_excel(writer, sheet_name="interview", index=False)

        adapter = PairedContextDatasetAdapter(
            dataset_path=path,
            class_cols=[],
        )
        with pytest.raises(DatasetValidationError, match="Item"):
            adapter.load_sheets()

    def test_file_not_found(self, tmp_path: Path):
        """Raise DatasetValidationError when file does not exist."""
        adapter = PairedContextDatasetAdapter(
            dataset_path=tmp_path / "nonexistent.xlsx",
            class_cols=CLASSES,
        )
        with pytest.raises((DatasetValidationError, FileNotFoundError)):
            adapter.load_sheets()


class TestTripletCorrespondence:
    """Verify triplet correspondence validation."""

    def test_matching_triplets_pass(self, synthetic_excel: Path):
        adapter = PairedContextDatasetAdapter(
            dataset_path=synthetic_excel,
            class_cols=CLASSES,
        )
        # Should not raise
        _train, _cf, _cc = adapter.load_and_aggregate()

    def test_mismatched_triplets_raise(self, tmp_path: Path):
        """Raise DatasetValidationError when triplets don't match."""
        path = tmp_path / "mismatch.xlsx"

        def make_df(patients, items, physicians, label=0.0):
            rows = []
            for pat, item in zip(patients, items):
                for phys in physicians:
                    row = {"Patient": pat, "Physician": phys, "Item": item}
                    for cls in CLASSES:
                        row[cls] = label
                    rows.append(row)
            return pd.DataFrame(rows)

        test_df = make_df([1, 1], ["a", "b"], [1, 2])
        # Interview has extra row
        interview_df = make_df([1, 1, 2], ["a", "b", "a"], [1, 2])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            test_df.to_excel(writer, sheet_name="train", index=False)
            test_df.to_excel(writer, sheet_name="test", index=False)
            interview_df.to_excel(writer, sheet_name="interview", index=False)

        adapter = PairedContextDatasetAdapter(
            dataset_path=path,
            class_cols=CLASSES,
        )
        with pytest.raises(DatasetValidationError, match="[Tt]riplet"):
            adapter.load_and_aggregate()


class TestManifest:
    """Verify manifest output."""

    def test_manifest_contains_required_fields(self, synthetic_excel: Path):
        adapter = PairedContextDatasetAdapter(
            dataset_path=synthetic_excel,
            class_cols=CLASSES,
        )
        m = adapter.manifest()
        assert "adapter" in m
        assert "dataset_path" in m
        assert "column_map" in m
        assert "label_mapping" in m
        assert "reference_aggregation" in m

    def test_manifest_adapter_name(self, synthetic_excel: Path):
        adapter = PairedContextDatasetAdapter(
            dataset_path=synthetic_excel,
            class_cols=CLASSES,
        )
        assert adapter.name == "paired_context_dataset"


class TestCustomColumnMap:
    """Verify adapter uses injected column map."""

    def test_custom_column_map_used(self, tmp_path: Path):
        """Adapter resolves columns via injected map."""
        path = tmp_path / "custom.xlsx"
        df = pd.DataFrame({
            "Subject": [1, 1],
            "Rater": [1, 2],
            "Question": ["q1", "q1"],
            "cat_a": [0.0, 1.0],
        })
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="training", index=False)
            df.to_excel(writer, sheet_name="survey", index=False)
            df.to_excel(writer, sheet_name="eval", index=False)

        custom_map = PairedContextColumnMap(
            context_entity_col="Subject",
            reference_observer_col="Rater",
            task_instance_col="Question",
            train_sheet="training",
            context_free_sheet="survey",
            correct_context_sheet="eval",
        )
        adapter = PairedContextDatasetAdapter(
            dataset_path=path,
            class_cols=["cat_a"],
            column_map=custom_map,
        )
        sheets = adapter.load_sheets()
        assert "training" in sheets
        assert "survey" in sheets
        assert "eval" in sheets
