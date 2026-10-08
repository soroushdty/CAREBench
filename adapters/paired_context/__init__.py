"""adapters.paired_context — paired-context adapter package.

Loads datasets stored in the CAREBench paired-context data format
(see ``docs/data_format.md``): an Excel workbook with a training sheet and
two matched reference-condition sheets (context-free and correct-context),
plus a JSON file of per-entity context records.

All workbook column names, sheet names, label display names, context-record
assumptions, and reference-observer assumptions are isolated in this package
and are configurable through ``PairedContextColumnMap``.

Public API:
    - PairedContextColumnMap
    - labels (default label space helpers)
    - PairedContextDatasetAdapter
    - PairedContextContextAdapter
    - PairedContextReferenceAdapter
    - PairedContextReasoningAdapter (Track 3)
    - PairedContextRepresentationAdapter (Track 1)
"""

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context.labels import (
    display_for_key,
    display_names,
    key_for_display,
    keys,
    normalize_output_dimension,
)
from adapters.paired_context.context_adapter import PairedContextContextAdapter
from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter
from adapters.paired_context.reasoning_adapter import PairedContextReasoningAdapter
from adapters.paired_context.reference_adapter import PairedContextReferenceAdapter
from adapters.paired_context.representation_adapter import PairedContextRepresentationAdapter

__all__ = [
    "PairedContextColumnMap",
    "PairedContextContextAdapter",
    "PairedContextDatasetAdapter",
    "PairedContextReasoningAdapter",
    "PairedContextReferenceAdapter",
    "PairedContextRepresentationAdapter",
    "display_for_key",
    "display_names",
    "key_for_display",
    "keys",
    "normalize_output_dimension",
]
