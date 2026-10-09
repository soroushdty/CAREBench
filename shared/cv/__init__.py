"""
shared.cv — Patient-grouped cross-validation splits.

Canonical path: shared/cv/
"""

from .patient_splits import (  # noqa: F401
    CVScheme,
    held_out_patients,
    lopo_splits,
    patient_splits,
    patient_to_fold,
    resolve_cv,
)

__all__ = [
    "CVScheme",
    "held_out_patients",
    "lopo_splits",
    "patient_splits",
    "patient_to_fold",
    "resolve_cv",
]
