"""Leave-One-Patient-Out cross-validation split generator."""

from __future__ import annotations

import numpy as np


def lopo_splits(
    patient_ids: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate leave-one-patient-out (LOPO) cross-validation splits.

    For each unique patient ID, one fold is produced where that patient's
    rows form the validation set and all remaining rows form the training set.
    The order of folds follows the sorted order of unique patient IDs.

    Args:
        patient_ids: 1-D array-like of patient IDs aligned with the data rows.

    Returns:
        A list of ``(train_indices, val_indices)`` pairs — one per unique
        patient ID.  Indices are integer arrays into the original row axis.

    Raises:
        ValueError: If ``patient_ids`` contains fewer than 2 unique patients,
            making a meaningful train/val split impossible.
    """
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError(
            f"patient_ids must be 1-D, got shape {patient_ids.shape}."
        )

    unique_patients = np.unique(patient_ids)
    if unique_patients.size < 2:
        raise ValueError(
            f"lopo_splits requires at least 2 unique patients, "
            f"got {unique_patients.size}."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for patient in unique_patients:
        val_mask = patient_ids == patient
        val_indices = np.where(val_mask)[0]
        train_indices = np.where(~val_mask)[0]
        splits.append((train_indices, val_indices))

    return splits
