"""Tests for adapters/paired_context/labels.py — Output dimension label space.

Verifies display↔key mappings, normalization, and error on unknown dimension.
"""

from __future__ import annotations

import pytest

from adapters.paired_context.labels import (
    UnknownOutputDimensionError,
    display_for_key,
    display_names,
    key_for_display,
    keys,
    label_manifest,
    normalize_output_dimension,
)


class TestKeysAndDisplayNames:
    """Verify basic registry accessors."""

    def test_keys_returns_10_items(self):
        assert len(keys()) == 10

    def test_display_names_returns_10_items(self):
        assert len(display_names()) == 10

    def test_keys_are_snake_case(self):
        for k in keys():
            assert k == k.lower()
            assert " " not in k

    def test_display_names_are_readable(self):
        for d in display_names():
            # Display names should have at least one uppercase letter
            assert any(c.isupper() for c in d) or d[0].isupper()

    def test_keys_and_display_names_same_order(self):
        k = keys()
        d = display_names()
        # First key should correspond to first display name
        assert key_for_display(d[0]) == k[0]
        assert display_for_key(k[0]) == d[0]


class TestKeyForDisplay:
    """Verify display-to-key lookups."""

    def test_behavioral_health(self):
        assert key_for_display("Behavioral health") == "behavioral_health"

    def test_diagnoses(self):
        assert key_for_display("Diagnoses") == "diagnoses"

    def test_sexual_reproductive_health(self):
        assert key_for_display("Sexual and reproductive health") == "sexual_reproductive_health"

    def test_social_determinants(self):
        assert key_for_display("Social determinants of health") == "social_determinants_of_health"

    def test_unknown_raises(self):
        with pytest.raises(UnknownOutputDimensionError):
            key_for_display("Not a real category")


class TestDisplayForKey:
    """Verify key-to-display lookups."""

    def test_behavioral_health(self):
        assert display_for_key("behavioral_health") == "Behavioral health"

    def test_infectious_diseases(self):
        assert display_for_key("infectious_diseases") == "Infectious diseases"

    def test_other(self):
        assert display_for_key("other") == "Other"

    def test_unknown_raises(self):
        with pytest.raises(UnknownOutputDimensionError):
            display_for_key("not_a_key")


class TestNormalizeOutputDimension:
    """Verify normalization from either key or display name to canonical key."""

    def test_key_passes_through(self):
        assert normalize_output_dimension("behavioral_health") == "behavioral_health"

    def test_display_name_maps_to_key(self):
        assert normalize_output_dimension("Behavioral health") == "behavioral_health"

    def test_case_insensitive_display(self):
        assert normalize_output_dimension("behavioral health") == "behavioral_health"
        assert normalize_output_dimension("BEHAVIORAL HEALTH") == "behavioral_health"

    def test_all_keys_normalize_to_self(self):
        for k in keys():
            assert normalize_output_dimension(k) == k

    def test_all_display_names_normalize_to_key(self):
        for d, k in zip(display_names(), keys()):
            assert normalize_output_dimension(d) == k

    def test_unknown_raises(self):
        with pytest.raises(UnknownOutputDimensionError):
            normalize_output_dimension("bogus_category")


class TestLabelManifest:
    """Verify manifest generation."""

    def test_manifest_is_dict(self):
        m = label_manifest()
        assert isinstance(m, dict)

    def test_manifest_has_10_entries(self):
        assert len(label_manifest()) == 10

    def test_manifest_keys_are_canonical(self):
        m = label_manifest()
        for k in keys():
            assert k in m
