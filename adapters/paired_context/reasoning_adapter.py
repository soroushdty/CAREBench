"""Paired-context reasoning adapter — composes dataset, context, reference, and labels.

Provides the complete reasoning-ready dataset loading pipeline for Track 3.
This is the single dataset-specific reasoning loader outside tests/configs.

Canonical path: adapters/paired_context/reasoning_adapter.py
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context.context_adapter import PairedContextContextAdapter
from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter
from adapters.paired_context.reference_adapter import PairedContextReferenceAdapter
from shared.label_space import LabelSpace

# ---------------------------------------------------------------------------
# PairedDataset DTO (reasoning-ready records)
# ---------------------------------------------------------------------------


@dataclass
class PairedDataset:
    """Paired reference-observer consensus dataset for reasoning evaluation.

    This is the canonical DTO shape consumed by tracks/reasoning.

    Attributes
    ----------
    patient_ids : np.ndarray
        Shape (N,). Context entity identifier for each row.
    item_texts : np.ndarray
        Shape (N,). Task instance text for each row.
    survey_consensus : np.ndarray
        Shape (N, C). Context-free reference consensus labels.
    interview_consensus : np.ndarray
        Shape (N, C). Correct-context reference consensus labels.
    delta_physician : np.ndarray
        Shape (N, C). Delta = correct_context - context_free.
    category_names : list[str]
        Output-dimension names (display names) in column order.
    label_space : LabelSpace | None
        Keys, display names and definitions of the output dimensions. When
        omitted it is built from ``category_names`` with
        :meth:`LabelSpace.from_config`.
    """

    patient_ids: np.ndarray
    item_texts: np.ndarray
    survey_consensus: np.ndarray
    interview_consensus: np.ndarray
    delta_physician: np.ndarray
    category_names: list[str]
    label_space: LabelSpace | None = field(default=None)

    def __post_init__(self) -> None:
        if self.label_space is None:
            self.label_space = LabelSpace.from_config(self.category_names)
        elif self.label_space.display_names() != list(self.category_names):
            raise ValueError(
                "label_space display names "
                f"{self.label_space.display_names()} do not match category_names "
                f"{list(self.category_names)}."
            )

    @property
    def canonical_category_names(self) -> list[str]:
        """Output-dimension keys in the same order as :attr:`category_names`.

        For example ``"Behavioral health"`` → ``"behavioral_health"``.
        """
        return self.label_space.keys()


# ---------------------------------------------------------------------------
# PairedContextReasoningAdapter
# ---------------------------------------------------------------------------


class PairedContextReasoningAdapter:
    """Paired-context reasoning adapter — full pipeline for Track 3.

    Composes ``PairedContextDatasetAdapter``, ``PairedContextContextAdapter``,
    ``PairedContextReferenceAdapter``, ``PairedContextColumnMap``, and the default label
    space to produce a ``PairedDataset`` DTO.

    Satisfies: ``shared.adapters.base.ReasoningDatasetAdapter``

    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset workbook (.xlsx).
    patient_summaries_path : str | Path | None
        Path to the patient_summaries.json file. Optional: only needed when
        the context adapter is used (context building), not for loading
        reference labels.
    class_cols : list[str]
        Output-dimension column names as they appear in the workbook
        (display names).
    class_definitions : Mapping[str, Any] | None
        Optional keys and prompt definitions per class, as accepted by
        :meth:`shared.label_space.LabelSpace.from_config`.
    column_map : PairedContextColumnMap | None
        Column mapping. Defaults to ``PairedContextColumnMap()``.
    mismatch_error : bool
        Whether to raise on observer count mismatches. Default: False.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        patient_summaries_path: str | Path | None,
        class_cols: list[str],
        column_map: PairedContextColumnMap | None = None,
        *,
        mismatch_error: bool = False,
        class_definitions: Mapping[str, Any] | None = None,
    ) -> None:
        self._column_map = column_map or PairedContextColumnMap()
        self._class_cols = list(class_cols)
        self._label_space = LabelSpace.from_config(self._class_cols, class_definitions)
        self._mismatch_error = mismatch_error

        self._reference_adapter = PairedContextReferenceAdapter(
            column_map=self._column_map,
            mismatch_error=mismatch_error,
        )
        self._dataset_adapter = PairedContextDatasetAdapter(
            dataset_path=dataset_path,
            class_cols=self._class_cols,
            column_map=self._column_map,
            reference_adapter=self._reference_adapter,
            mismatch_error=mismatch_error,
            label_space=self._label_space,
        )
        # Context records are only needed for context building, not for
        # loading reference labels, so the summaries file is optional here.
        self._context_adapter = (
            PairedContextContextAdapter(summaries_path=patient_summaries_path)
            if patient_summaries_path is not None
            else None
        )

    # ------------------------------------------------------------------
    # Protocol: ReasoningDatasetAdapter
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return "paired_context_reasoning"

    def manifest(self) -> dict[str, Any]:
        """Return reproducibility manifest."""
        return {
            "adapter": self.name,
            "dataset": self._dataset_adapter.manifest(),
            "context": (
                self._context_adapter.manifest()
                if self._context_adapter is not None
                else None
            ),
            "reference": self._reference_adapter.manifest(),
            "label_mapping": self._label_space.manifest(),
            "column_map": self._column_map.to_dict(),
        }

    def load_reasoning_dataset(self) -> PairedDataset:
        """Load and return a reasoning-ready PairedDataset.

        Steps
        -----
        1. Load and aggregate the dataset via PairedContextDatasetAdapter.
        2. Align context-free and correct-context aggregated frames.
        3. Compute delta (correct_context - context_free).
        4. Return PairedDataset DTO.

        Returns
        -------
        PairedDataset
            The reasoning-ready dataset with consensus arrays.
        """
        cmap = self._column_map

        # Step 1: Load and aggregate
        _train_df, context_free_agg, correct_context_agg = (
            self._dataset_adapter.load_and_aggregate()
        )

        # Step 2: Align frames by (entity, task_instance) keys
        context_free_aligned, correct_context_aligned = self._align_frames(
            context_free_agg, correct_context_agg
        )

        # Step 3: Extract arrays and compute delta
        survey_matrix = context_free_aligned[self._class_cols].to_numpy(dtype=float)
        interview_matrix = correct_context_aligned[self._class_cols].to_numpy(dtype=float)
        delta_matrix = interview_matrix - survey_matrix

        patient_ids = context_free_aligned[cmap.context_entity_col].to_numpy()
        item_texts = context_free_aligned[cmap.task_instance_col].to_numpy(dtype=str)

        # Step 4: Return DTO
        return PairedDataset(
            patient_ids=patient_ids,
            item_texts=item_texts,
            survey_consensus=survey_matrix,
            interview_consensus=interview_matrix,
            delta_physician=delta_matrix,
            category_names=list(self._class_cols),
            label_space=self._label_space,
        )

    # ------------------------------------------------------------------
    # Accessors for composed adapters
    # ------------------------------------------------------------------

    @property
    def context_adapter(self) -> PairedContextContextAdapter:
        """Access the underlying context adapter (for context building)."""
        if self._context_adapter is None:
            raise ValueError(
                "No patient_summaries_path was configured, so no context adapter is available."
            )
        return self._context_adapter

    @property
    def dataset_adapter(self) -> PairedContextDatasetAdapter:
        """Access the underlying dataset adapter."""
        return self._dataset_adapter

    @property
    def reference_adapter(self) -> PairedContextReferenceAdapter:
        """Access the underlying reference adapter."""
        return self._reference_adapter

    @property
    def label_space(self) -> LabelSpace:
        """The output dimensions this adapter loads."""
        return self._label_space

    @property
    def column_map(self) -> PairedContextColumnMap:
        """Access the active column map."""
        return self._column_map

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _align_frames(
        self,
        context_free_agg: pd.DataFrame,
        correct_context_agg: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sort both aggregated frames by (entity, task_instance) for alignment."""
        cmap = self._column_map
        key_cols = [cmap.context_entity_col, cmap.task_instance_col]

        cf_sorted = (
            context_free_agg.sort_values(by=key_cols, kind="mergesort")
            .reset_index(drop=True)
        )
        cc_sorted = (
            correct_context_agg.sort_values(by=key_cols, kind="mergesort")
            .reset_index(drop=True)
        )

        # Sanity check: keys should align after triplet verification
        cf_keys = cf_sorted[key_cols].reset_index(drop=True)
        cc_keys = cc_sorted[key_cols].reset_index(drop=True)

        if not cf_keys.equals(cc_keys):
            from adapters.paired_context.dataset_adapter import DatasetValidationError
            raise DatasetValidationError(
                "Aggregated context-free and correct-context frames have different "
                "key sets after sorting. This indicates a post-aggregation mismatch."
            )

        return cf_sorted, cc_sorted
