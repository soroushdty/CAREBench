"""Paired-context dataset adapter — workbook loading, validation, and canonical records.

Provides the primary dataset loading boundary for paired-context
datasets. Downstream tracks do not parse the dataset workbook directly; they
receive validated DataFrames through this adapter.

Canonical path: adapters/paired_context/dataset_adapter.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context import labels as paired_context_labels
from adapters.paired_context.reference_adapter import PairedContextReferenceAdapter
from shared.io.excel import load_workbook_sheets


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetValidationError(Exception):
    """Raised when the paired-context dataset fails structural validation."""


# ---------------------------------------------------------------------------
# PairedContextDatasetAdapter
# ---------------------------------------------------------------------------


class PairedContextDatasetAdapter:
    """Dataset adapter for workbook loading and validation.

    Uses ``shared.io.excel.load_workbook_sheets`` for raw I/O, then applies
    dataset-specific dtype enforcement, column validation, and triplet
    correspondence checks.

    Satisfies: ``shared.adapters.base.DatasetAdapter``

    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset workbook (.xlsx).
    class_cols : list[str]
        Output-dimension column names (display names as they appear in the
        workbook, e.g. ``["Behavioral health", "Diagnoses", ...]``).
    column_map : PairedContextColumnMap | None
        Column mapping. Defaults to ``PairedContextColumnMap()`` if not provided.
    reference_adapter : PairedContextReferenceAdapter | None
        Reference aggregation adapter. Defaults to a new instance if not
        provided.
    mismatch_error : bool
        Passed to reference adapter if one is created internally.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        class_cols: list[str],
        column_map: PairedContextColumnMap | None = None,
        reference_adapter: PairedContextReferenceAdapter | None = None,
        *,
        mismatch_error: bool = False,
    ) -> None:
        self._dataset_path = Path(dataset_path).resolve()
        self._class_cols = list(class_cols)
        self._column_map = column_map or PairedContextColumnMap()
        self._reference_adapter = reference_adapter or PairedContextReferenceAdapter(
            column_map=self._column_map, mismatch_error=mismatch_error
        )

    # ------------------------------------------------------------------
    # Protocol: DatasetAdapter
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return "paired_context_dataset"

    def manifest(self) -> dict[str, Any]:
        """Return reproducibility manifest."""
        return {
            "adapter": self.name,
            "dataset_path": str(self._dataset_path),
            "column_map": self._column_map.to_dict(),
            "label_mapping": paired_context_labels.label_manifest(),
            "reference_aggregation": self._reference_adapter.manifest(),
            "class_cols": self._class_cols,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_sheets(self) -> dict[str, pd.DataFrame]:
        """Load all configured sheets from the workbook.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of sheet name → raw DataFrame.

        Raises
        ------
        DatasetValidationError
            If required sheets are missing.
        FileNotFoundError
            If the dataset file does not exist.
        """
        dtype_map = self._build_dtype_map()
        try:
            sheets = load_workbook_sheets(
                self._dataset_path,
                self._column_map.sheet_names(),
                dtype_map=dtype_map,
            )
        except Exception as exc:
            raise DatasetValidationError(
                f"Failed to load workbook dataset sheets: {exc}"
            ) from exc

        # Validate required columns in each sheet
        self._validate_columns(sheets)
        return sheets

    def load_and_aggregate(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load sheets, validate, and aggregate reference observers.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, context_free_aggregated, correct_context_aggregated)
            Train is returned raw; context_free and correct_context sheets
            have physician pairs aggregated via the reference adapter.

        Raises
        ------
        DatasetValidationError
            If validation (columns, triplets) fails.
        """
        sheets = self.load_sheets()
        cmap = self._column_map

        train_df = sheets[cmap.train_sheet]
        context_free_raw = sheets[cmap.context_free_sheet]
        correct_context_raw = sheets[cmap.correct_context_sheet]

        # Validate triplet correspondence between context-free and correct-context
        self._verify_triplet_correspondence(context_free_raw, correct_context_raw)

        # Aggregate reference observers
        context_free_agg = self._reference_adapter.aggregate(
            context_free_raw, class_cols=self._class_cols
        )
        correct_context_agg = self._reference_adapter.aggregate(
            correct_context_raw, class_cols=self._class_cols
        )

        return train_df, context_free_agg, correct_context_agg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dtype_map(self) -> dict[str, str]:
        """Build dtype mapping for pandas read_excel."""
        cmap = self._column_map
        dtype_map: dict[str, str] = {
            cmap.context_entity_col: "string",
            cmap.reference_observer_col: "string",
            cmap.task_instance_col: "string",
        }
        for col in self._class_cols:
            dtype_map[col] = "Float64"
        return dtype_map

    def _validate_columns(self, sheets: dict[str, pd.DataFrame]) -> None:
        """Validate that required configured columns exist in loaded sheets."""
        cmap = self._column_map
        required_cols = [
            cmap.context_entity_col,
            cmap.reference_observer_col,
            cmap.task_instance_col,
        ]

        for sheet_name, df in sheets.items():
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise DatasetValidationError(
                    f"Sheet '{sheet_name}' is missing required columns: "
                    f"{', '.join(sorted(missing))}. "
                    f"Available columns: {', '.join(sorted(df.columns))}"
                )

    def _verify_triplet_correspondence(
        self,
        context_free_df: pd.DataFrame,
        correct_context_df: pd.DataFrame,
    ) -> None:
        """Verify (entity, observer, task_instance) triplets match exactly."""
        cmap = self._column_map
        key_cols = [
            cmap.context_entity_col,
            cmap.reference_observer_col,
            cmap.task_instance_col,
        ]

        # Check columns exist
        for col in key_cols:
            if col not in context_free_df.columns:
                raise DatasetValidationError(
                    f"Column '{col}' missing from context-free sheet."
                )
            if col not in correct_context_df.columns:
                raise DatasetValidationError(
                    f"Column '{col}' missing from correct-context sheet."
                )

        def _triplet_set(df: pd.DataFrame) -> set[tuple[Any, ...]]:
            return set(
                tuple(row)
                for row in df[key_cols].itertuples(index=False, name=None)
            )

        cf_triplets = _triplet_set(context_free_df)
        cc_triplets = _triplet_set(correct_context_df)

        only_in_cf = cf_triplets - cc_triplets
        only_in_cc = cc_triplets - cf_triplets

        if only_in_cf or only_in_cc:
            parts: list[str] = []
            if only_in_cf:
                parts.append(
                    f"Triplets in context-free but not correct-context "
                    f"({len(only_in_cf)}): {sorted(only_in_cf)[:5]}"
                )
            if only_in_cc:
                parts.append(
                    f"Triplets in correct-context but not context-free "
                    f"({len(only_in_cc)}): {sorted(only_in_cc)[:5]}"
                )
            raise DatasetValidationError(
                "Triplet correspondence mismatch between context-free and "
                "correct-context sheets. " + " | ".join(parts)
            )
