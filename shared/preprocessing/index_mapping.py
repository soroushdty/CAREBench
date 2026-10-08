from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union
import pandas as pd
from shared.utils.mapping_utils import _build_reverse_lookup, _load_grouped_mapping
from shared.utils.text_utils import normalize_for_matching

JsonPath = Union[str, Path]
GroupedMapping = Mapping[str, Iterable[str]]
IndexMapping = Dict[int, Any]

def index_mapping(
    df: pd.DataFrame,
    mapping_input: Union[GroupedMapping, JsonPath],
    cfg: Mapping[str, Any],
    item_col: str = "items",
    mapping_has_unique_values: bool = False,
    remove_spaces: bool = True,
    _return_resolved_set: bool = False,
) -> "IndexMapping | tuple[IndexMapping, frozenset]":
    """Build an index map from row-index to standardized item value.

    Parameters
    ----------
    _return_resolved_set : bool, default False
        When True, also return a frozenset of row indices whose item was
        found in the JSON reverse lookup (i.e. JSON actually covered that
        row).  Rows whose normalized item happens to equal the canonical
        are correctly counted as resolved regardless of value equality.
    """
    if item_col not in df.columns:
        raise KeyError(f"Column '{item_col}' not found in DataFrame.")

    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a dictionary-like mapping object with configuration parameters.")

    grouped = _load_grouped_mapping(mapping_input, remove_spaces=remove_spaces)
    reverse_lookup = _build_reverse_lookup(grouped)

    index_map: IndexMapping = {}
    resolved_indices: set = set()

    for idx, raw_val in df[item_col].items():
        if pd.isna(raw_val):
            index_map[idx] = raw_val
            continue

        val = str(raw_val).strip() if remove_spaces else str(raw_val)
        val_norm = normalize_for_matching(val, strip=remove_spaces)

        if val_norm in reverse_lookup:
            index_map[idx] = reverse_lookup[val_norm]
            resolved_indices.add(idx)
        else:
            if mapping_has_unique_values:
                raise ValueError(
                    f"Value '{raw_val}' at index {idx} is not found in grouped mapping "
                    "while mapping_has_unique_values=True."
                )
            index_map[idx] = val

    if _return_resolved_set:
        return index_map, frozenset(resolved_indices)
    return index_map
