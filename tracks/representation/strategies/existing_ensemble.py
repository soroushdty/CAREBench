"""Existing ensemble training strategy wrapper.

Wraps the existing LOPO-CV ensemble training pipeline as a TrainingStrategy
implementation. Delegates all training logic to ``train_ensemble_pipeline()``
and handles only the field-name translation.

Canonical path: tracks/representation/strategies/existing_ensemble.py
"""

from __future__ import annotations

from typing import Any

from .base import RepresentationDataset


class ExistingEnsembleTrainingStrategy:
    """Wraps the existing LOPO-CV ensemble training pipeline as a TrainingStrategy.

    Delegates all training logic to ``train_ensemble_pipeline()``. This wrapper
    handles only the translation between ``RepresentationDataset`` fields and
    the legacy function signature.

    Field mapping
    -------------
    dataset.X_train               -> X_train
    dataset.Y_train               -> Y_train
    dataset.patient_ids_train     -> patient_ids_train
    dataset.X_test                -> X_test
    dataset.Y_test                -> Y_test
    dataset.patient_ids_test      -> patient_ids_test
    dataset.item_strings_train    -> item_strings_train
    dataset.item_strings_test     -> item_strings_test
    dataset.Y_test_context_free   -> Y_test_survey
    dataset.Y_test_correct_context -> Y_test_interview
    dataset.context_vectors       -> context_vectors
    """

    def fit_and_evaluate(
        self, dataset: RepresentationDataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate dataset fields and delegate to train_ensemble_pipeline.

        Parameters
        ----------
        dataset : RepresentationDataset
            The canonical data container produced by an adapter.
        config : dict[str, Any]
            Pipeline configuration dict (passed as ``cfg``).

        Returns
        -------
        dict[str, Any]
            The result dict from ``train_ensemble_pipeline``.
        """
        # Deferred import to avoid heavyweight dependencies at module load time
        from tracks.representation.training.orchestrator.train_ensemble_pipeline import (
            train_ensemble_pipeline,
        )

        result = train_ensemble_pipeline(
            X_train=dataset.X_train,
            Y_train=dataset.Y_train,
            patient_ids_train=dataset.patient_ids_train,
            X_test=dataset.X_test,
            Y_test=dataset.Y_test,
            cfg=config,
            item_strings_train=dataset.item_strings_train,
            item_strings_test=dataset.item_strings_test,
            patient_ids_test=dataset.patient_ids_test,
            Y_test_interview=dataset.Y_test_correct_context,
            Y_test_survey=dataset.Y_test_context_free,
            context_vectors=dataset.context_vectors,
        )
        return result
