"""Patient-grouped cross-validation splits.

Every split holds out whole patients: a patient's rows are either all in the
validation set or all in the training set, so no held-out patient's labels can
reach the model. Two schemes are supported:

- ``lopo``: leave one patient out. One fold per patient, in sorted patient order.
- ``group_kfold``: patients are shuffled with ``seed`` and dealt into
  ``n_splits`` folds of near-equal patient count. If ``n_splits`` is at least the
  number of patients, every fold holds out one patient (LOPO in shuffled order).

The scheme for each loop comes from config::

    cv:
      outer: {scheme: lopo, n_splits: null, seed: null}
      inner: {scheme: lopo, n_splits: null, seed: null}

A missing ``cv`` block, loop, or key means ``lopo``. For ``group_kfold`` a
missing ``seed`` falls back to ``global_seed``.

Canonical path: shared/cv/patient_splits.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SCHEMES = ("lopo", "group_kfold")
LOOPS = ("outer", "inner")
_KEYS = ("scheme", "n_splits", "seed")

Splits = list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CVScheme:
    """How to split patients into folds for one cross-validation loop."""

    scheme: str = "lopo"
    n_splits: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(
                f"cv scheme must be one of {SCHEMES}, got {self.scheme!r}."
            )
        if self.scheme == "lopo":
            if self.n_splits is not None or self.seed is not None:
                raise ValueError(
                    "cv scheme 'lopo' takes no n_splits or seed "
                    f"(got n_splits={self.n_splits!r}, seed={self.seed!r})."
                )
            return
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int):
            raise ValueError(
                f"cv scheme 'group_kfold' needs an integer n_splits, got {self.n_splits!r}."
            )
        if self.n_splits < 2:
            raise ValueError(
                f"cv n_splits must be at least 2, got {self.n_splits}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(
                f"cv scheme 'group_kfold' needs an integer seed, got {self.seed!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "n_splits": self.n_splits, "seed": self.seed}


def resolve_cv(cfg: Mapping[str, Any], loop: str) -> CVScheme:
    """Read the ``cv.<loop>`` block of ``cfg`` into a :class:`CVScheme`.

    Raises:
        ValueError: on an unknown loop name, an unknown key, or an invalid scheme.
    """
    if loop not in LOOPS:
        raise ValueError(f"cv loop must be one of {LOOPS}, got {loop!r}.")
    cv = cfg.get("cv") or {}
    unknown = set(cv) - set(LOOPS)
    if unknown:
        raise ValueError(
            f"Unknown cv loop(s) {sorted(unknown)}; expected {list(LOOPS)}."
        )
    block = cv.get(loop) or {}
    unknown = set(block) - set(_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in cv.{loop}; expected {list(_KEYS)}."
        )
    scheme = block.get("scheme") or "lopo"
    n_splits = block.get("n_splits")
    seed = block.get("seed")
    if scheme == "group_kfold" and seed is None:
        seed = cfg.get("global_seed")
    return CVScheme(scheme=scheme, n_splits=n_splits, seed=seed)


def _patient_groups(unique_patients: np.ndarray, cv: CVScheme) -> list[np.ndarray]:
    """Partition sorted unique patient IDs into held-out groups, one per fold."""
    if cv.scheme == "lopo":
        return [unique_patients[i : i + 1] for i in range(unique_patients.size)]
    shuffled = np.random.default_rng(cv.seed).permutation(unique_patients)
    n_splits = min(cv.n_splits, unique_patients.size)
    return [np.sort(group) for group in np.array_split(shuffled, n_splits)]


def patient_splits(patient_ids: np.ndarray, cv: CVScheme | None = None) -> Splits:
    """Generate patient-grouped cross-validation splits.

    Args:
        patient_ids: 1-D array of patient IDs aligned with the data rows.
        cv: The split scheme. ``None`` means leave one patient out.

    Returns:
        A list of ``(train_indices, val_indices)`` pairs, one per fold. Indices
        are sorted integer arrays into the original row axis.

    Raises:
        ValueError: If ``patient_ids`` is not 1-D or has fewer than 2 unique
            patients.
    """
    cv = cv or CVScheme()
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError(
            f"patient_ids must be 1-D, got shape {patient_ids.shape}."
        )

    unique_patients = np.unique(patient_ids)
    if unique_patients.size < 2:
        raise ValueError(
            f"Patient-grouped splits require at least 2 unique patients, "
            f"got {unique_patients.size}."
        )

    splits: Splits = []
    for group in _patient_groups(unique_patients, cv):
        val_mask = np.isin(patient_ids, group)
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


def lopo_splits(patient_ids: np.ndarray) -> Splits:
    """Leave-one-patient-out splits, one fold per patient in sorted order."""
    return patient_splits(patient_ids, CVScheme())


def held_out_patients(patient_ids: np.ndarray, splits: Splits) -> list[np.ndarray]:
    """Return the sorted patient IDs held out by each fold."""
    patient_ids = np.asarray(patient_ids)
    return [np.unique(patient_ids[val_ix]) for _, val_ix in splits]


def patient_to_fold(patient_ids: np.ndarray, splits: Splits) -> dict[Any, int]:
    """Map every held-out patient ID to the index of the fold that held it out."""
    return {
        patient.item() if isinstance(patient, np.generic) else patient: fold_idx
        for fold_idx, patients in enumerate(held_out_patients(patient_ids, splits))
        for patient in patients
    }
