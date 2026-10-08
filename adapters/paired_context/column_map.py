"""Paired-context column and sheet mapping.

All dataset-specific column names, sheet names, and structural assumptions
are consolidated here. Other modules receive a PairedContextColumnMap instance
rather than hardcoding these values.

Canonical path: adapters/paired_context/column_map.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PairedContextColumnMap:
    """Typed, immutable mapping of paired-context dataset field names and conventions.

    Default values correspond to the default CAREBench paired-context schema used by
    the bundled synthetic example (``examples/synthetic/dataset.xlsx``). Use
    :meth:`from_config` to build a map from a run configuration.

    Parameters
    ----------
    context_entity_col : str
        Column identifying the context entity (patient). Default: ``"Patient"``.
    reference_observer_col : str
        Column identifying the reference observer (physician). Default: ``"Physician"``.
    task_instance_col : str
        Column identifying the task instance (item). Default: ``"Item"``.
    train_sheet : str
        Workbook sheet name for training data. Default: ``"train"``.
    context_free_sheet : str
        Workbook sheet name for context-free reference condition (survey/test).
        Default: ``"test"``.
    correct_context_sheet : str
        Workbook sheet name for correct-context reference condition (interview).
        Default: ``"interview"``.
    context_free_suffix : str
        Suffix used for context-free reference condition columns.
        Default: ``"_survey"``.
    correct_context_suffix : str
        Suffix used for correct-context reference condition columns.
        Default: ``"_interview"``.
    expected_reference_observer_count : int
        Expected number of reference observers per (entity, task_instance) group.
        Default: ``2``.
    """

    # Column names
    context_entity_col: str = "Patient"
    reference_observer_col: str = "Physician"
    task_instance_col: str = "Item"

    # Sheet names
    train_sheet: str = "train"
    context_free_sheet: str = "test"
    correct_context_sheet: str = "interview"

    # Condition suffixes
    context_free_suffix: str = "_survey"
    correct_context_suffix: str = "_interview"

    # Structural assumptions
    expected_reference_observer_count: int = 2

    @classmethod
    def from_config(cls, data_cfg: dict[str, Any]) -> "PairedContextColumnMap":
        """Build a column map from a run configuration's ``data`` section.

        Recognized keys (all optional; missing keys keep the defaults):
        ``patient_col``, ``physician_col``, ``item_col``, ``train_sheet``,
        ``test_sheet`` (context-free condition), ``interview_sheet``
        (correct-context condition), and ``physician_count``.
        """
        config_to_field = {
            "patient_col": "context_entity_col",
            "physician_col": "reference_observer_col",
            "item_col": "task_instance_col",
            "train_sheet": "train_sheet",
            "test_sheet": "context_free_sheet",
            "interview_sheet": "correct_context_sheet",
            "physician_count": "expected_reference_observer_count",
        }
        overrides = {
            field: data_cfg[key]
            for key, field in config_to_field.items()
            if data_cfg.get(key) is not None
        }
        return cls(**overrides)

    def sheet_names(self) -> list[str]:
        """Return all three sheet names in canonical order."""
        return [self.train_sheet, self.context_free_sheet, self.correct_context_sheet]

    def to_dict(self) -> dict[str, str | int]:
        """Serialize to a plain dict for manifest/config use."""
        return {
            "context_entity_col": self.context_entity_col,
            "reference_observer_col": self.reference_observer_col,
            "task_instance_col": self.task_instance_col,
            "train_sheet": self.train_sheet,
            "context_free_sheet": self.context_free_sheet,
            "correct_context_sheet": self.correct_context_sheet,
            "context_free_suffix": self.context_free_suffix,
            "correct_context_suffix": self.correct_context_suffix,
            "expected_reference_observer_count": self.expected_reference_observer_count,
        }
