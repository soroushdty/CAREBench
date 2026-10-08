"""Default output-dimension label space.

Defines the default list of 10 sensitive-data output dimensions used by
CAREBench (the taxonomy used in the SHARES project). Provides helpers for
normalization, display↔key lookups, and manifest generation.

Canonical path: adapters/paired_context/labels.py

The canonical machine keys use snake_case. The display names use the
human-readable form as it appears in reports and configs.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Authoritative label registry: (canonical_key, display_name)
# ---------------------------------------------------------------------------

_LABEL_REGISTRY: tuple[tuple[str, str], ...] = (
    ("behavioral_health", "Behavioral health"),
    ("diagnoses", "Diagnoses"),
    ("disabilities", "Disabilities"),
    ("infectious_diseases", "Infectious diseases"),
    ("genetics", "Genetics"),
    ("medications", "Medications"),
    ("sexual_reproductive_health", "Sexual and reproductive health"),
    ("social_determinants_of_health", "Social determinants of health"),
    ("violence", "Violence"),
    ("other", "Other"),
)

# Pre-computed lookup dicts (built at import time from the immutable registry)
_KEY_TO_DISPLAY: dict[str, str] = {k: d for k, d in _LABEL_REGISTRY}
_DISPLAY_TO_KEY: dict[str, str] = {d: k for k, d in _LABEL_REGISTRY}
# Normalized lookup: lowercased display names → key (for fuzzy matching)
_DISPLAY_LOWER_TO_KEY: dict[str, str] = {d.lower(): k for k, d in _LABEL_REGISTRY}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownOutputDimensionError(ValueError):
    """Raised when an unrecognized output dimension is supplied."""

    def __init__(self, value: str) -> None:
        valid_keys = ", ".join(k for k, _ in _LABEL_REGISTRY)
        valid_displays = ", ".join(d for _, d in _LABEL_REGISTRY)
        super().__init__(
            f"Unknown output dimension: {value!r}. "
            f"Valid keys: [{valid_keys}]. "
            f"Valid display names: [{valid_displays}]."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def keys() -> list[str]:
    """Return canonical machine keys in registry order."""
    return [k for k, _ in _LABEL_REGISTRY]


def display_names() -> list[str]:
    """Return human-readable display names in registry order."""
    return [d for _, d in _LABEL_REGISTRY]


def key_for_display(display_name: str) -> str:
    """Return the canonical key for a given display name.

    Parameters
    ----------
    display_name : str
        A valid display name (case-sensitive).

    Returns
    -------
    str
        The canonical machine key.

    Raises
    ------
    UnknownOutputDimensionError
        If the display name is not recognized.
    """
    result = _DISPLAY_TO_KEY.get(display_name)
    if result is None:
        raise UnknownOutputDimensionError(display_name)
    return result


def display_for_key(key: str) -> str:
    """Return the display name for a given canonical key.

    Parameters
    ----------
    key : str
        A valid canonical machine key.

    Returns
    -------
    str
        The human-readable display name.

    Raises
    ------
    UnknownOutputDimensionError
        If the key is not recognized.
    """
    result = _KEY_TO_DISPLAY.get(key)
    if result is None:
        raise UnknownOutputDimensionError(key)
    return result


def normalize_output_dimension(value: str) -> str:
    """Normalize a value (key or display name) to the canonical key.

    Accepts either a canonical key or a display name (case-insensitive for
    display names) and returns the canonical machine key.

    Parameters
    ----------
    value : str
        Either a canonical key or a display name.

    Returns
    -------
    str
        The canonical machine key.

    Raises
    ------
    UnknownOutputDimensionError
        If the value is neither a valid key nor a valid display name.
    """
    # Direct key match
    if value in _KEY_TO_DISPLAY:
        return value

    # Direct display name match (case-sensitive)
    if value in _DISPLAY_TO_KEY:
        return _DISPLAY_TO_KEY[value]

    # Case-insensitive display name match
    lower_val = value.lower()
    if lower_val in _DISPLAY_LOWER_TO_KEY:
        return _DISPLAY_LOWER_TO_KEY[lower_val]

    raise UnknownOutputDimensionError(value)


def label_manifest() -> dict[str, str]:
    """Return a key→display mapping suitable for inclusion in run manifests."""
    return dict(_KEY_TO_DISPLAY)
