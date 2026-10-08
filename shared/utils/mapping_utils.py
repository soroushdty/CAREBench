"""Grouped mapping and reverse-lookup helpers shared across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Union

from shared.utils.text_utils import normalize_for_matching
from shared.utils.json_utils import load_json_from_source, JsonPath

GroupedMapping = Mapping[str, list[str] | tuple[str, ...] | set[str]]


def is_valid_grouped_mapping_obj(data: object) -> bool:
    """Return ``True`` if ``data`` is a valid grouped-mapping object.

    A valid grouped mapping is a ``dict`` where every key is a ``str`` and
    every value is a ``list``, ``tuple``, or ``set`` of ``str`` elements.
    Plain string values are **not** accepted as values (they must be wrapped
    in a sequence).

    Args:
        data: Object to validate.

    Returns:
        bool: ``True`` if ``data`` satisfies the grouped-mapping contract,
            ``False`` otherwise (no exception is raised).
    """
    if not isinstance(data, dict):
        return False
    for key, value in data.items():
        if not isinstance(key, str):
            return False
        if isinstance(value, str) or not isinstance(value, (list, tuple, set)):
            return False
        if not all(isinstance(v, str) for v in value):
            return False
    return True


def _load_grouped_mapping(
    mapping_input: Union[GroupedMapping, JsonPath],
    *,
    remove_spaces: bool = True,
) -> Dict[str, list[str]]:
    """Load a grouped mapping from a dict or JSON file path.

    Returns a normalized dict with casefolded/stripped keys and aliases.
    """
    if isinstance(mapping_input, (str, Path)):
        raw = load_json_from_source(mapping_input, allow_json_payload=False)
    elif isinstance(mapping_input, Mapping):
        raw = dict(mapping_input)
    else:
        raise TypeError(
            "mapping_input must be either a dict-like grouped mapping or a JSON path (str/Path)."
        )
    if not isinstance(raw, dict):
        raise ValueError("Grouped mapping must be a dictionary at top level.")
    normalized: Dict[str, list[str]] = {}
    for standard, aliases in raw.items():
        if not isinstance(standard, str):
            raise ValueError("All grouped mapping keys (standardized items) must be strings.")
        if not isinstance(aliases, (list, tuple, set)):
            raise ValueError(
                f"Grouped mapping value for '{standard}' must be a list/tuple/set of aliases."
            )
        std_norm = normalize_for_matching(standard, strip=remove_spaces)
        alias_norm: list[str] = []
        for a in aliases:
            if not isinstance(a, str):
                raise ValueError(f"Alias '{a}' under '{standard}' must be a string.")
            alias_norm.append(normalize_for_matching(a, strip=remove_spaces))
        normalized[std_norm] = alias_norm
    return normalized


def _build_reverse_lookup(grouped_mapping: Dict[str, list[str]]) -> Dict[str, str]:
    """Build alias->standardized lookup from a grouped mapping.

    Each standardized key also maps to itself.
    """
    reverse: Dict[str, str] = {}
    for std, aliases in grouped_mapping.items():
        if std in reverse and reverse[std] != std:
            raise ValueError(f"Conflicting mapping for standardized item '{std}'.")
        reverse[std] = std
        for alias in aliases:
            if alias in reverse and reverse[alias] != std:
                raise ValueError(
                    f"Alias '{alias}' appears under multiple standardized items: "
                    f"'{reverse[alias]}' and '{std}'."
                )
            reverse[alias] = std
    return reverse
