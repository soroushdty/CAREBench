"""Default output-dimension label space.

Thin wrapper around :data:`shared.label_space.DEFAULT_LABEL_SPACE`, the ten
sensitive-data categories used in the SHARES project. Provides helpers for
normalization, display<->key lookups, and manifest generation.

Datasets with a different taxonomy configure it with ``classes`` and
``class_definitions`` (see ``docs/adapters.md``) and use a
:class:`shared.label_space.LabelSpace` built from config instead.

Canonical path: adapters/paired_context/labels.py

The canonical machine keys use snake_case. The display names use the
human-readable form as it appears in reports and configs.
"""

from __future__ import annotations

from shared.label_space import (
    DEFAULT_LABEL_SPACE,
    LabelSpace,
    UnknownOutputDimensionError,
)

__all__ = [
    "DEFAULT_LABEL_SPACE",
    "LabelSpace",
    "UnknownOutputDimensionError",
    "display_for_key",
    "display_names",
    "key_for_display",
    "keys",
    "label_manifest",
    "normalize_output_dimension",
]


def keys() -> list[str]:
    """Return canonical machine keys in registry order."""
    return DEFAULT_LABEL_SPACE.keys()


def display_names() -> list[str]:
    """Return human-readable display names in registry order."""
    return DEFAULT_LABEL_SPACE.display_names()


def key_for_display(display_name: str) -> str:
    """Return the canonical key for a (case-sensitive) display name.

    Raises
    ------
    UnknownOutputDimensionError
        If the display name is not recognized.
    """
    return DEFAULT_LABEL_SPACE.key_for_display(display_name)


def display_for_key(key: str) -> str:
    """Return the display name for a canonical key.

    Raises
    ------
    UnknownOutputDimensionError
        If the key is not recognized.
    """
    return DEFAULT_LABEL_SPACE.display_for_key(key)


def normalize_output_dimension(value: str) -> str:
    """Normalize a key or display name (case-insensitive) to the canonical key.

    Raises
    ------
    UnknownOutputDimensionError
        If the value is neither a valid key nor a valid display name.
    """
    return DEFAULT_LABEL_SPACE.normalize(value)


def label_manifest() -> dict[str, str]:
    """Return a key->display mapping suitable for inclusion in run manifests."""
    return DEFAULT_LABEL_SPACE.manifest()
