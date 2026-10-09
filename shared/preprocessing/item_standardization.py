from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

import pandas as pd
from pandas.api.types import is_integer_dtype

from shared.preprocessing.index_mapping import index_mapping

JsonPath = Union[str, Path]
GroupedMapping = Mapping[str, Iterable[str]]
IndexMapping = Dict[int, str]


def _is_canonical_int_string(value: Any) -> bool:
    """
    Return True only for strings that are unambiguously integer literals,
    such as '0', '-3', '42'.

    Rejects strings like:
      - '01'
      - ' 5 '
      - '1.0'
      - 'abc'
    """
    if not isinstance(value, str) or value == "":
        return False

    if value[0] == "-":
        digits = value[1:]
        if digits == "":
            return False
        return digits.isdigit() and (digits == "0" or not digits.startswith("0"))

    return value.isdigit() and (value == "0" or not value.startswith("0"))


def _normalize_index_map_keys(
    df_index: pd.Index,
    index_map: Mapping[Union[int, str], Any],
) -> Dict[Any, Any]:
    """
    Normalize mapping keys to match `df_index` safely.

    Rules
    -----
    1. Exact key matches are always accepted.
    2. String-to-int coercion is allowed only when:
       - df_index is a non-MultiIndex integer index, and
       - the key is a canonical integer string, and
       - the coerced integer exists in df_index.
    3. No other coercion is attempted.

    This avoids brittle broad try/except casting and prevents silent
    mismatches for string, object, or MultiIndex-based DataFrames.
    """
    normalized: Dict[Any, Any] = {}

    # Fast path for exact matches
    for key, value in index_map.items():
        if key in df_index:
            normalized[key] = value

    # Controlled support for JSON-loaded stringified integer keys
    if isinstance(df_index, pd.MultiIndex) or not is_integer_dtype(df_index.dtype):
        return normalized

    for key, value in index_map.items():
        if key in normalized:
            continue

        if _is_canonical_int_string(key):
            coerced_key = int(key)
            if coerced_key in df_index:
                normalized[coerced_key] = value

    return normalized


def _item_standardizer(
    df: pd.DataFrame,
    index_map: Mapping[Union[int, str], Any],
    item_col: str = "items",
) -> pd.DataFrame:
    """
    Return a new DataFrame where `item_col` values are replaced using
    index-based mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    index_map : Mapping[int | str, Any]
        Output from `index_mapping`, mapping row index -> standardized value.
        Stringified integer keys are accepted only for integer-based indexes
        (e.g. keys loaded from JSON).
    item_col : str, default="items"
        Column to replace.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with standardized item column.
    """
    if item_col not in df.columns:
        raise KeyError(f"Column '{item_col}' not found in DataFrame.")

    normalized_map = _normalize_index_map_keys(df.index, index_map)

    out_df = df.copy()

    if not normalized_map:
        return out_df

    # Vectorized alignment by index value rather than row-by-row assignment.
    index_series = pd.Series(out_df.index, index=out_df.index)
    mask = index_series.isin(normalized_map)

    if mask.any():
        out_df.loc[mask, item_col] = index_series[mask].map(normalized_map).to_numpy()

    return out_df


def standardize_items(
    df: pd.DataFrame,
    *,
    item_col: str,
    mapping_json: Union[GroupedMapping, JsonPath, None] = None,
    cfg: dict,
    remove_spaces: bool = True,
    return_index_map: bool = False,
    return_resolved_mask: bool = False,
) -> "pd.DataFrame | tuple":
    """
    Standardize item column using a precomputed grouped mapping JSON/dict.

    If ``mapping_json`` is None the DataFrame is returned unchanged with an
    all-False resolved mask (no items were standardized).  Fuzzy fallback is
    NOT performed here; it is a separate train-only runtime stage.

    Parameters
    ----------
    mapping_json : grouped mapping dict/path or None
        When None, the DataFrame is returned as-is (no standardization).
    return_index_map : bool
        When True, also return the IndexMapping dict.
    return_resolved_mask : bool
        When True, also return a boolean pd.Series (index-aligned) where
        True means the row's item was found in the JSON mapping.
    """
    if item_col not in df.columns:
        raise KeyError(f"Column '{item_col}' not found in DataFrame.")

    if mapping_json is None:
        index_map: IndexMapping = {idx: val for idx, val in df[item_col].items()}
        resolved_mask = pd.Series(False, index=df.index, dtype=bool)
        result: list = [df.copy()]
        if return_index_map:
            result.append(index_map)
        if return_resolved_mask:
            result.append(resolved_mask)
        return tuple(result) if len(result) > 1 else result[0]

    # JSON mapping path
    index_map, resolved_set = index_mapping(
        df=df,
        mapping_input=mapping_json,
        item_col=item_col,
        mapping_has_unique_values=False,
        cfg=cfg,
        remove_spaces=remove_spaces,
        _return_resolved_set=True,
    )

    standardized_df = _item_standardizer(df=df, index_map=index_map, item_col=item_col)

    resolved_mask = pd.Series(
        [idx in resolved_set for idx in df.index],
        index=df.index,
        dtype=bool,
    )

    result = [standardized_df]
    if return_index_map:
        result.append(index_map)
    if return_resolved_mask:
        result.append(resolved_mask)
    return tuple(result) if len(result) > 1 else result[0]
