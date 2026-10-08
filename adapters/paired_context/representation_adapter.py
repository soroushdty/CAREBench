"""Paired-context representation adapter — full Track 1 dataset loading.

Loads a paired-context dataset and constructs a ``RepresentationDataset``
with all fields populated: embeddings, labels, train/test splits, the
item-standardization state needed for fuzzy fallback, and the per-entity
context records used by Stage 2.

Workbook sheets are read through ``PairedContextDatasetAdapter`` and context
records through ``PairedContextContextAdapter`` — the same loaders the
reasoning track uses — and then run through the shared preprocessing pipeline
(item standardization, reference-observer aggregation, summaries).

This module MAY know dataset sheet names, columns, and patient-summary shapes.
The Track 1 framework (``framework.py``) does NOT import this module.

Canonical path: adapters/paired_context/representation_adapter.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from shared.adapters.base import RepresentationDataset

logger = logging.getLogger(__name__)


class PairedContextRepresentationAdapter:
    """Paired-context dataset adapter for Track 1.

    Knows dataset sheet names, column names, patient-summary shapes, and
    label ordering. Produces a ``RepresentationDataset`` with all fields
    populated.

    Uses
    ----
    - ``adapters.paired_context.column_map.PairedContextColumnMap`` for column constants
    - ``adapters.paired_context.dataset_adapter.PairedContextDatasetAdapter`` for workbook loading
    - ``adapters.paired_context.context_adapter.PairedContextContextAdapter`` for context records
    - ``shared.preprocessing.preprocessing.preprocess`` for standardization and aggregation
    - ``shared.embeddings.compute_embeddings`` for embedding computation
    """

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return "paired_context_representation"

    def load_representation_dataset(
        self, config: dict[str, Any]
    ) -> RepresentationDataset:
        """Load paired-context data and return a RepresentationDataset.

        Parameters
        ----------
        config : dict[str, Any]
            Resolved Track 1 configuration (``main_config.yaml`` merged with
            the training and summary configs). Keys read:

            - ``DIR_DATASET``, ``TRAIN_SHEET``, ``TEST_SHEET``,
              ``INTERVIEW_SHEET``: workbook path and sheet names
            - ``patient_col``, ``physician_col``, ``item_col``,
              ``physician_count``: see ``PairedContextColumnMap.from_config``
            - ``classes``: output-dimension column names
            - ``llm``, ``batch_size``: item embedding model and batch size
            - ``DIR_CONTEXT`` (optional): patient summaries JSON for Stage 2
            - ``PROJECT_ROOT`` (optional): base for relative paths
            - plus every key the shared preprocessing pipeline reads
              (``DIR_JSON_MAP``, ``ENABLE_SUMMARY``, ``DIR_SUMMARY``, ...)

        Returns
        -------
        RepresentationDataset
            Fully populated data container ready for strategy injection.
        """
        from adapters.paired_context.column_map import PairedContextColumnMap
        from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter
        from adapters.paired_context import labels as paired_context_labels
        from shared.embeddings.compute_embeddings import compute_embeddings
        from shared.preprocessing.preprocessing import preprocess

        project_root = Path(config.get("PROJECT_ROOT", ".")).resolve()
        dataset_path = _resolve(project_root, config["DIR_DATASET"])
        output_dimensions = list(config["classes"])
        column_map = PairedContextColumnMap.from_config(
            {
                "patient_col": config.get("patient_col"),
                "physician_col": config.get("physician_col"),
                "item_col": config.get("item_col"),
                "train_sheet": config.get("TRAIN_SHEET"),
                "test_sheet": config.get("TEST_SHEET"),
                "interview_sheet": config.get("INTERVIEW_SHEET"),
                "physician_count": config.get("physician_count"),
            }
        )

        # --- Context records (Stage 2) ---
        context_records = self._load_context_records(config, project_root)

        # --- Load sheets and preprocess ---
        sheets = PairedContextDatasetAdapter(
            dataset_path=dataset_path,
            class_cols=output_dimensions,
            column_map=column_map,
        ).load_sheets()

        logger.info("Starting preprocessing...")
        df = preprocess(config, dfs=sheets)

        # --- Embeddings ---
        item_col = column_map.task_instance_col
        patient_col = column_map.context_entity_col

        logger.info("Computing embeddings with model: %s", config["llm"])
        embedding_map = compute_embeddings(
            config["llm"],
            data=df[item_col],
            schema="pandas",
            batch_size=config.get("batch_size", 32),
            cfg=config,
        )

        # --- Build arrays ---
        context_free_cols = [
            f"{c}{column_map.context_free_suffix}" for c in output_dimensions
        ]
        correct_context_cols = [
            f"{c}{column_map.correct_context_suffix}" for c in output_dimensions
        ]

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        X_train = np.vstack(
            [embedding_map[item.casefold()] for item in train_df[item_col]]
        )
        X_test = np.vstack(
            [embedding_map[item.casefold()] for item in test_df[item_col]]
        )

        Y_train = train_df[context_free_cols].to_numpy(dtype=np.float32)
        Y_test = test_df[context_free_cols].to_numpy(dtype=np.float32)
        Y_test_context_free = (
            test_df[context_free_cols].to_numpy(dtype=np.float64).astype(np.float32)
        )
        Y_test_correct_context = (
            test_df[correct_context_cols].to_numpy(dtype=np.float64).astype(np.float32)
        )

        logger.info(
            "Paired-context dataset loaded: X_train=%s, X_test=%s, %d output dimensions",
            X_train.shape,
            X_test.shape,
            len(output_dimensions),
        )

        return RepresentationDataset(
            X_train=X_train,
            Y_train=Y_train,
            patient_ids_train=train_df[patient_col].to_numpy(),
            X_test=X_test,
            Y_test=Y_test,
            patient_ids_test=test_df[patient_col].to_numpy(),
            output_dimensions=output_dimensions,
            item_strings_train=train_df[item_col].to_numpy(),
            item_strings_test=test_df[item_col].to_numpy(),
            Y_test_context_free=Y_test_context_free,
            Y_test_correct_context=Y_test_correct_context,
            unresolved_mask_train=(~train_df["item_json_resolved"]).to_numpy(),
            unresolved_mask_test=(~test_df["item_json_resolved"]).to_numpy(),
            embedding_cache=embedding_map,
            context_records=context_records,
            metadata={
                "adapter": self.name,
                "dataset_path": str(dataset_path),
                "sheet_names": {
                    "train": column_map.train_sheet,
                    "test": column_map.context_free_sheet,
                    "interview": column_map.correct_context_sheet,
                },
                "label_space": {
                    paired_context_labels.normalize_output_dimension(name): name
                    for name in output_dimensions
                },
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_context_records(
        config: dict[str, Any], project_root: Path
    ) -> dict[str, Any] | None:
        """Load per-entity context records, or None when Stage 2 has no context."""
        from adapters.paired_context.context_adapter import PairedContextContextAdapter

        if not config.get("DIR_CONTEXT"):
            return None
        summaries_path = _resolve(project_root, config["DIR_CONTEXT"])
        if not summaries_path.exists():
            logger.warning(
                "DIR_CONTEXT path not found: %s — Stage 2 fusion disabled.",
                summaries_path,
            )
            return None
        return PairedContextContextAdapter(
            summaries_path=summaries_path
        ).load_context_records()


def _resolve(project_root: Path, raw_path: Any) -> Path:
    """Resolve a config path against the project root."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path
