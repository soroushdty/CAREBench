"""
Tests for canonical key normalization in schema_validator.

Covers:
- Display label → canonical key mapping
- Canonical key pass-through
- Unknown name raises KeyError
- Case sensitivity
- DISPLAY_TO_CANONICAL / CANONICAL_TO_DISPLAY consistency
"""

from __future__ import annotations

import pytest

from tracks.reasoning.schema_validator import (
    CANONICAL_TO_DISPLAY,
    DISPLAY_TO_CANONICAL,
    REQUIRED_KEYS,
    normalize_category_key,
)


# ---------------------------------------------------------------------------
# Display label → canonical key
# ---------------------------------------------------------------------------


class TestNormalizeCategoryKey:
    """Tests for normalize_category_key()."""

    @pytest.mark.parametrize(
        "display_label,expected_canonical",
        list(DISPLAY_TO_CANONICAL.items()),
    )
    def test_display_labels_normalize_to_canonical(
        self, display_label: str, expected_canonical: str
    ) -> None:
        """All 10 display labels normalize to the correct canonical key."""
        assert normalize_category_key(display_label) == expected_canonical

    @pytest.mark.parametrize("canonical_key", REQUIRED_KEYS)
    def test_canonical_keys_pass_through(self, canonical_key: str) -> None:
        """All 10 canonical keys pass through unchanged."""
        assert normalize_category_key(canonical_key) == canonical_key

    def test_unknown_name_raises_key_error(self) -> None:
        """Unknown names raise KeyError with descriptive message."""
        with pytest.raises(KeyError, match="Unknown output-dimension name"):
            normalize_category_key("not_a_real_category")

    def test_empty_string_raises_key_error(self) -> None:
        """Empty string raises KeyError."""
        with pytest.raises(KeyError):
            normalize_category_key("")

    @pytest.mark.parametrize(
        "wrong_case",
        [
            "Behavioral_Health",
            "BEHAVIORAL_HEALTH",
            "behavioral health",
            "Behavioral Health",
            "DIAGNOSES",
            "other ",
            " other",
        ],
    )
    def test_case_sensitivity_raises_key_error(self, wrong_case: str) -> None:
        """Incorrect casing or extra whitespace raises KeyError."""
        with pytest.raises(KeyError):
            normalize_category_key(wrong_case)


# ---------------------------------------------------------------------------
# Mapping consistency
# ---------------------------------------------------------------------------


class TestMappingConsistency:
    """Tests for DISPLAY_TO_CANONICAL / CANONICAL_TO_DISPLAY consistency."""

    def test_display_to_canonical_has_10_entries(self) -> None:
        assert len(DISPLAY_TO_CANONICAL) == 10

    def test_canonical_to_display_has_10_entries(self) -> None:
        assert len(CANONICAL_TO_DISPLAY) == 10

    def test_reverse_mapping_is_correct(self) -> None:
        """CANONICAL_TO_DISPLAY is the exact inverse of DISPLAY_TO_CANONICAL."""
        for display, canonical in DISPLAY_TO_CANONICAL.items():
            assert CANONICAL_TO_DISPLAY[canonical] == display

    def test_canonical_values_match_required_keys(self) -> None:
        """DISPLAY_TO_CANONICAL values are exactly REQUIRED_KEYS."""
        assert set(DISPLAY_TO_CANONICAL.values()) == set(REQUIRED_KEYS)

    def test_canonical_to_display_keys_match_required_keys(self) -> None:
        """CANONICAL_TO_DISPLAY keys are exactly REQUIRED_KEYS."""
        assert set(CANONICAL_TO_DISPLAY.keys()) == set(REQUIRED_KEYS)
