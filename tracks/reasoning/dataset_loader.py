"""
Dataset loader for the reasoning track (Track 3).

Loads and returns paired consensus datasets through the adapter boundary.
All dataset parsing flows through the configured adapter
(by default adapters.paired_context.reasoning_adapter).

Canonical path: tracks/reasoning/dataset_loader.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context.reasoning_adapter import (
    PairedContextReasoningAdapter,
    PairedDataset,
)
from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians


# ---------------------------------------------------------------------------
# Re-export PairedDataset for backward compatibility
# ---------------------------------------------------------------------------

# PairedDataset is re-exported from adapters.paired_context.reasoning_adapter
__all__ = ["DatasetLoader", "PairedDataset", "ValidationError"]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when dataset validation fails."""


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type] = {
    "paired_context": PairedContextReasoningAdapter,
}


# ---------------------------------------------------------------------------
# DatasetLoader (adapter-backed)
# ---------------------------------------------------------------------------


class DatasetLoader:
    """Adapter-backed dataset loader for the reasoning track.

    Delegates all dataset loading to the configured adapter. Currently
    supports the paired-context adapter; future adapters can be registered in
    ``_ADAPTER_REGISTRY``.

    Parameters
    ----------
    cfg : dict
        Validated assay/reasoning configuration dictionary.
        Expected structure: ``cfg["data"]`` with keys for dataset_path,
        patient_summaries_path, train_sheet, test_sheet, interview_sheet,
        patient_col, physician_col, item_col, classes.
    dataset_path : str | Path | None
        Override path to dataset file. If ``None``, resolved from config.
    adapter_name : str | None
        Name of the adapter to use. If ``None``, read from ``cfg["data"]["adapter"]``,
        falling back to ``"paired_context"``.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        dataset_path: str | Path | None = None,
        adapter_name: str | None = None,
    ) -> None:
        self._cfg = cfg
        data_cfg: dict[str, Any] = cfg["data"]
        if adapter_name is None:
            adapter_name = data_cfg.get("adapter", "paired_context")

        # Resolve adapter
        if adapter_name not in _ADAPTER_REGISTRY:
            available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
            raise ValidationError(
                f"Unknown adapter '{adapter_name}'. "
                f"Available adapters: {available}"
            )

        # Resolve dataset path
        if dataset_path is not None:
            resolved_path = Path(dataset_path).resolve()
        else:
            resolved_path = Path(data_cfg["dataset_path"]).resolve()

        # Resolve patient summaries path
        # Optional: only context building needs the patient summaries.
        summaries_raw = data_cfg.get("patient_summaries_path")
        summaries_path = Path(summaries_raw).resolve() if summaries_raw else None

        # Column names for backward compat
        self._patient_col: str = data_cfg["patient_col"]
        self._physician_col: str = data_cfg["physician_col"]
        self._item_col: str = data_cfg["item_col"]
        self._class_cols: list[str] = list(data_cfg["classes"])

        # Sheet names for backward compat
        self._train_sheet: str = data_cfg["train_sheet"]
        self._test_sheet: str = data_cfg["test_sheet"]
        self._interview_sheet: str = data_cfg["interview_sheet"]

        # Build the adapter
        adapter_cls = _ADAPTER_REGISTRY[adapter_name]
        self._adapter = adapter_cls(
            dataset_path=resolved_path,
            patient_summaries_path=summaries_path,
            class_cols=self._class_cols,
            column_map=PairedContextColumnMap.from_config(data_cfg),
            mismatch_error=bool(data_cfg.get("mismatch_error", False)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> PairedDataset:
        """Load the dataset through the adapter boundary.

        Returns
        -------
        PairedDataset
            The reasoning-ready paired consensus dataset.

        Raises
        ------
        ValidationError
            If any validation fails during loading.
        """
        try:
            return self._adapter.load_reasoning_dataset()
        except Exception as exc:
            raise ValidationError(str(exc)) from exc

    def manifest(self) -> dict[str, Any]:
        """Return the reproducibility manifest of the underlying adapter."""
        return self._adapter.manifest()

    @property
    def adapter_name(self) -> str:
        """Name of the underlying adapter."""
        return self._adapter.name

    @property
    def adapter_class(self) -> str:
        """Fully qualified class path of the underlying adapter."""
        cls = type(self._adapter)
        return f"{cls.__module__}.{cls.__qualname__}"

    # ------------------------------------------------------------------
    # Backward-compatible helpers
    # ------------------------------------------------------------------

    def _verify_triplet_correspondence(self, test_df, interview_df) -> None:
        """Verify triplet correspondence (delegated to adapter internals).

        Maintained for backward compatibility with existing tests that call
        this method directly. Adapter validation errors are re-raised as
        :class:`ValidationError`, matching :meth:`load`.
        """
        try:
            self._adapter.dataset_adapter._verify_triplet_correspondence(
                test_df, interview_df
            )
        except Exception as exc:
            raise ValidationError(str(exc)) from exc
