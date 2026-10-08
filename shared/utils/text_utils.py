"""Text normalization helpers shared across the pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_for_matching(value: Any, *, strip: bool = True) -> Any:
    """Normalize a value for matching/mapping/overlap checks (casefold + optional strip)."""
    if pd.isna(value):
        return value
    text = str(value)
    if strip:
        text = text.strip()
    return text.casefold()


def trim_item_value(value: Any) -> Any:
    """Strip leading/trailing whitespace from a string value; pass through NaN."""
    if pd.isna(value):
        return value
    return str(value).strip()
