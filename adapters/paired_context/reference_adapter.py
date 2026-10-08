"""Paired-context reference observer aggregation adapter.

Isolates the two-physician paired-mean aggregation assumption behind the
ReferenceAdapter protocol. The shared aggregation function
(shared.reference.aggregation.paired_reference_mean._aggregate_physicians)
remains dataset-neutral; this adapter supplies the configured column names
and observer count.

Canonical path: adapters/paired_context/reference_adapter.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from adapters.paired_context.column_map import PairedContextColumnMap
from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians


class PairedContextReferenceAdapter:
    """Reference observer aggregation for the paired-context format.

    Wraps the shared ``_aggregate_physicians`` function with configured column
    names and the expected two-observer count from ``PairedContextColumnMap``.

    Satisfies: ``shared.adapters.base.ReferenceAdapter``

    Parameters
    ----------
    column_map : PairedContextColumnMap | None
        Column mapping to use. Defaults to ``PairedContextColumnMap()`` if not provided.
    mismatch_error : bool
        If True, raise on unresolvable observer count mismatches rather than
        skipping rows. Default: False.
    """

    def __init__(
        self,
        column_map: PairedContextColumnMap | None = None,
        *,
        mismatch_error: bool = False,
    ) -> None:
        self._column_map = column_map or PairedContextColumnMap()
        self._mismatch_error = mismatch_error

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return "paired_context_reference"

    def manifest(self) -> dict[str, Any]:
        """Return reproducibility manifest for reference aggregation."""
        return {
            "adapter": self.name,
            "aggregation_policy": "paired_reference_mean",
            "expected_observer_count": self._column_map.expected_reference_observer_count,
            "context_entity_col": self._column_map.context_entity_col,
            "reference_observer_col": self._column_map.reference_observer_col,
            "task_instance_col": self._column_map.task_instance_col,
            "mismatch_error": self._mismatch_error,
        }

    def aggregate(self, df: pd.DataFrame, *, class_cols: list[str]) -> pd.DataFrame:
        """Aggregate per-observer rows into consensus rows.

        Delegates to the shared ``_aggregate_physicians`` function with
        the configured column names and expected observer count.

        Parameters
        ----------
        df : pd.DataFrame
            Raw per-observer records with columns for context entity,
            reference observer, task instance, and class labels.
        class_cols : list[str]
            Label/class column names to aggregate (mean).

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with one row per (entity, task_instance).
        """
        return _aggregate_physicians(
            df,
            patient_col=self._column_map.context_entity_col,
            physician_col=self._column_map.reference_observer_col,
            item_col=self._column_map.task_instance_col,
            class_cols=class_cols,
            physician_count=self._column_map.expected_reference_observer_count,
            mismatch_error=self._mismatch_error,
        )
