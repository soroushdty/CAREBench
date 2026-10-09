"""Tests for adapters/paired_context/column_map.py — PairedContextColumnMap.

Verifies default values, frozen immutability, and non-default injection.
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from adapters.paired_context.column_map import PairedContextColumnMap


class TestDefaultValues:
    """Verify default column map values match the default paired-context schema."""

    def test_context_entity_col_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.context_entity_col == "Patient"

    def test_reference_observer_col_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.reference_observer_col == "Physician"

    def test_task_instance_col_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.task_instance_col == "Item"

    def test_train_sheet_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.train_sheet == "train"

    def test_context_free_sheet_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.context_free_sheet == "test"

    def test_correct_context_sheet_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.correct_context_sheet == "interview"

    def test_context_free_suffix_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.context_free_suffix == "_survey"

    def test_correct_context_suffix_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.correct_context_suffix == "_interview"

    def test_expected_reference_observer_count_default(self):
        cmap = PairedContextColumnMap()
        assert cmap.expected_reference_observer_count == 2

    def test_sheet_names_returns_all_three(self):
        cmap = PairedContextColumnMap()
        assert cmap.sheet_names() == ["train", "test", "interview"]


class TestFrozenBehavior:
    """Verify the column map is immutable."""

    def test_cannot_set_attribute(self):
        cmap = PairedContextColumnMap()
        with pytest.raises(FrozenInstanceError):
            cmap.context_entity_col = "Subject"  # type: ignore


class TestNonDefaultInjection:
    """Verify non-default values can be injected for testing."""

    def test_custom_column_names(self):
        cmap = PairedContextColumnMap(
            context_entity_col="Subject",
            reference_observer_col="Rater",
            task_instance_col="Question",
        )
        assert cmap.context_entity_col == "Subject"
        assert cmap.reference_observer_col == "Rater"
        assert cmap.task_instance_col == "Question"

    def test_custom_sheet_names(self):
        cmap = PairedContextColumnMap(
            train_sheet="training",
            context_free_sheet="survey",
            correct_context_sheet="eval",
        )
        assert cmap.sheet_names() == ["training", "survey", "eval"]

    def test_custom_observer_count(self):
        cmap = PairedContextColumnMap(expected_reference_observer_count=3)
        assert cmap.expected_reference_observer_count == 3

    def test_to_dict_includes_all_fields(self):
        cmap = PairedContextColumnMap()
        d = cmap.to_dict()
        expected_keys = {
            "context_entity_col",
            "reference_observer_col",
            "task_instance_col",
            "train_sheet",
            "context_free_sheet",
            "correct_context_sheet",
            "context_free_suffix",
            "correct_context_suffix",
            "expected_reference_observer_count",
        }
        assert set(d.keys()) == expected_keys


class TestFromConfig:
    """Verify PairedContextColumnMap.from_config maps run-config keys to fields."""

    def test_maps_all_recognized_keys(self):
        cmap = PairedContextColumnMap.from_config(
            {
                "patient_col": "subject_id",
                "physician_col": "rater_id",
                "item_col": "text",
                "train_sheet": "tr",
                "test_sheet": "no_context",
                "interview_sheet": "with_context",
                "physician_count": 3,
            }
        )
        assert cmap.context_entity_col == "subject_id"
        assert cmap.reference_observer_col == "rater_id"
        assert cmap.task_instance_col == "text"
        assert cmap.sheet_names() == ["tr", "no_context", "with_context"]
        assert cmap.expected_reference_observer_count == 3

    def test_missing_keys_keep_defaults(self):
        assert PairedContextColumnMap.from_config({"item_col": "text"}) == PairedContextColumnMap(
            task_instance_col="text"
        )

    def test_none_values_keep_defaults(self):
        assert PairedContextColumnMap.from_config({"patient_col": None}) == PairedContextColumnMap()

    def test_ignores_unrelated_keys(self):
        assert PairedContextColumnMap.from_config({"classes": ["Other"]}) == PairedContextColumnMap()
