"""shared.adapters — Shared adapter protocol definitions.

Canonical path: shared/adapters/
"""

from shared.adapters.base import (
    ContextAdapter,
    DatasetAdapter,
    ReasoningDatasetAdapter,
    ReferenceAdapter,
)

__all__ = [
    "ContextAdapter",
    "DatasetAdapter",
    "ReasoningDatasetAdapter",
    "ReferenceAdapter",
]
