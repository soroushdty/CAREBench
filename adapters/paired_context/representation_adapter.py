"""Paired-context representation adapter — full Track 1 dataset loading.

Loads a paired-context dataset and constructs a
``RepresentationDataset`` with all fields populated: embeddings, labels,
train/test splits, optional context vectors for Stage 2.

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
    - ``adapters.paired_context.labels`` for output dimension ordering
    - ``adapters.paired_context.dataset_adapter.PairedContextDatasetAdapter`` for workbook loading
    - ``shared.embeddings.compute_embeddings`` for embedding computation
    - ``adapters.paired_context.context_adapter.PairedContextContextAdapter`` for Stage 2 context
    """

    def load_representation_dataset(
        self, config: dict[str, Any]
    ) -> RepresentationDataset:
        """Load paired-context data and return a RepresentationDataset.

        Parameters
        ----------
        config : dict[str, Any]
            Pipeline configuration. Expected keys:
            - ``dataset_path``: path to the workbook Excel file
            - ``patient_summaries_path`` (required when ``stage2_enabled``):
              path to patient_summaries.json
            - ``classes`` (optional): output-dimension column names; defaults
              to the default label space display names
            - ``patient_col`` / ``physician_col`` / ``item_col`` and sheet
              names (optional): see ``PairedContextColumnMap.from_config``
            - ``embedding_model`` (optional): embedding model name
            - ``stage2_enabled`` (optional): whether to build context vectors

        Returns
        -------
        RepresentationDataset
            Fully populated data container ready for strategy injection.
        """
        from adapters.paired_context.column_map import PairedContextColumnMap
        from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter
        from adapters.paired_context import labels as paired_context_labels

        import pandas as pd

        # --- Resolve paths and config ---
        dataset_path = Path(config["dataset_path"])
        column_map = PairedContextColumnMap.from_config(config)
        output_dimensions = list(config.get("classes") or paired_context_labels.display_names())

        # --- Load and aggregate dataset ---
        adapter = PairedContextDatasetAdapter(
            dataset_path=dataset_path,
            class_cols=output_dimensions,
            column_map=column_map,
        )
        train_df, context_free_agg, correct_context_agg = adapter.load_and_aggregate()

        # --- Build item strings ---
        item_col = column_map.task_instance_col
        patient_col = column_map.context_entity_col

        item_strings_train = train_df[item_col].values.astype(str)
        item_strings_test = context_free_agg[item_col].values.astype(str)

        # --- Build patient IDs ---
        patient_ids_train = train_df[patient_col].values.astype(str)
        patient_ids_test = context_free_agg[patient_col].values.astype(str)

        # --- Build label matrices ---
        Y_train = train_df[output_dimensions].values.astype(np.float32)
        Y_test_context_free = context_free_agg[output_dimensions].values.astype(
            np.float32
        )
        Y_test_correct_context = correct_context_agg[output_dimensions].values.astype(
            np.float32
        )
        # Default Y_test uses context-free labels (consistent with existing pipeline)
        Y_test = Y_test_context_free.copy()

        # --- Build embeddings ---
        X_train, X_test = self._build_embeddings(
            item_strings_train, item_strings_test, config
        )

        # --- Optionally build context vectors for Stage 2 ---
        context_vectors: dict[str, np.ndarray] | None = None
        if config.get("stage2_enabled", False):
            context_vectors = self._build_context_vectors(config, patient_ids_test)

        logger.info(
            "Paired-context dataset loaded: X_train=%s, X_test=%s, %d output dimensions",
            X_train.shape,
            X_test.shape,
            len(output_dimensions),
        )

        return RepresentationDataset(
            X_train=X_train,
            Y_train=Y_train,
            patient_ids_train=patient_ids_train,
            X_test=X_test,
            Y_test=Y_test,
            patient_ids_test=patient_ids_test,
            output_dimensions=output_dimensions,
            item_strings_train=item_strings_train,
            item_strings_test=item_strings_test,
            Y_test_context_free=Y_test_context_free,
            Y_test_correct_context=Y_test_correct_context,
            context_vectors=context_vectors,
            metadata={
                "adapter": "paired_context_representation",
                "dataset_path": str(dataset_path),
                "label_space": {
                    paired_context_labels.normalize_output_dimension(name): name
                    for name in output_dimensions
                },
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embeddings(
        self,
        item_strings_train: np.ndarray,
        item_strings_test: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute or load embeddings for train and test item strings.

        Uses the shared embedding utility when available. Falls back to
        loading precomputed embeddings from config paths.
        """
        # Check for precomputed embedding paths first
        train_emb_path = config.get("embeddings_train_path")
        test_emb_path = config.get("embeddings_test_path")

        if train_emb_path and test_emb_path:
            logger.info("Loading precomputed embeddings from paths...")
            X_train = np.load(train_emb_path).astype(np.float32)
            X_test = np.load(test_emb_path).astype(np.float32)
            return X_train, X_test

        # Fall back to computing embeddings via shared utility
        from shared.embeddings.compute_embeddings import compute_embeddings

        all_strings = np.concatenate([item_strings_train, item_strings_test])
        unique_strings = list(set(all_strings.tolist()))

        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        logger.info(
            "Computing embeddings for %d unique items with model=%s",
            len(unique_strings),
            embedding_model,
        )

        import pandas as pd

        # compute_embeddings keys its result by the stripped string.
        embeddings_dict = compute_embeddings(
            embedding_model, pd.Series(unique_strings), schema="pandas", cfg=config
        )

        X_train = np.array(
            [embeddings_dict[s.strip()] for s in item_strings_train.tolist()],
            dtype=np.float32,
        )
        X_test = np.array(
            [embeddings_dict[s.strip()] for s in item_strings_test.tolist()],
            dtype=np.float32,
        )
        return X_train, X_test

    def _build_context_vectors(
        self,
        config: dict[str, Any],
        patient_ids_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build context vectors for Stage 2 fusion.

        Uses the paired-context context adapter to load patient summaries and
        build context vector representations.
        """
        from adapters.paired_context.context_adapter import PairedContextContextAdapter

        summaries_path = config["patient_summaries_path"]
        context_adapter = PairedContextContextAdapter(summaries_path=summaries_path)

        # Load context records for all test patients
        unique_patients = list(set(patient_ids_test.tolist()))
        context_vectors: dict[str, np.ndarray] = {}

        # Check for precomputed context vectors
        context_vectors_path = config.get("context_vectors_path")
        if context_vectors_path:
            import json

            with open(context_vectors_path, "r") as f:
                raw = json.load(f)
            for pid, vec in raw.items():
                context_vectors[str(pid)] = np.array(vec, dtype=np.float32)
            return context_vectors

        # Build context vectors from patient summaries using embeddings
        import pandas as pd

        from shared.embeddings.compute_embeddings import compute_embeddings

        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")

        for pid in unique_patients:
            try:
                record = context_adapter.get_record(str(pid))
                context_text = context_adapter.format_context(record)
                emb = compute_embeddings(
                    embedding_model, pd.Series([context_text]), schema="pandas", cfg=config
                )
                context_vectors[str(pid)] = np.array(
                    emb[context_text.strip()], dtype=np.float32
                )
            except KeyError:
                logger.warning(
                    "No context record for patient %s; skipping context vector.",
                    pid,
                )

        return context_vectors
