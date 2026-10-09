"""Tests for shared/label_space.py."""
from __future__ import annotations

import pytest

from adapters.paired_context import labels as paired_context_labels
from shared.label_space import (
    DEFAULT_LABEL_SPACE,
    LabelSpace,
    OutputDimension,
    UnknownOutputDimensionError,
    slugify_key,
)


class TestDefaultLabelSpace:
    def test_ten_dimensions_with_definitions(self):
        assert len(DEFAULT_LABEL_SPACE) == 10
        assert all(d.definition for d in DEFAULT_LABEL_SPACE)

    def test_adapter_helpers_delegate_to_default(self):
        assert paired_context_labels.keys() == DEFAULT_LABEL_SPACE.keys()
        assert paired_context_labels.display_names() == DEFAULT_LABEL_SPACE.display_names()
        assert paired_context_labels.label_manifest() == DEFAULT_LABEL_SPACE.manifest()

    def test_from_config_with_default_names_is_default(self):
        space = LabelSpace.from_config(DEFAULT_LABEL_SPACE.display_names())
        assert space == DEFAULT_LABEL_SPACE


class TestFromConfig:
    def test_custom_names_get_slug_keys_and_no_definition(self):
        space = LabelSpace.from_config(["Mood & anxiety", "Housing"])
        assert space.keys() == ["mood_anxiety", "housing"]
        assert space.display_names() == ["Mood & anxiety", "Housing"]
        assert space.definitions() == {"mood_anxiety": None, "housing": None}

    def test_definition_string(self):
        space = LabelSpace.from_config(["Housing"], {"Housing": "Housing instability"})
        assert space.definitions() == {"housing": "Housing instability"}

    def test_definition_mapping_with_key(self):
        space = LabelSpace.from_config(
            ["Drug therapy"],
            {"Drug therapy": {"key": "medications", "definition": "Any drug"}},
        )
        assert space.keys() == ["medications"]
        assert space.display_for_key("medications") == "Drug therapy"
        assert space.definitions() == {"medications": "Any drug"}

    def test_default_name_keeps_default_key_and_definition(self):
        space = LabelSpace.from_config(["Genetics", "Housing"])
        assert space.keys() == ["genetics", "housing"]
        assert space.definitions()["genetics"] == DEFAULT_LABEL_SPACE.definitions()["genetics"]

    def test_default_definition_can_be_overridden(self):
        space = LabelSpace.from_config(["Genetics"], {"Genetics": "Germline variants"})
        assert space.definitions() == {"genetics": "Germline variants"}

    def test_definition_for_unknown_class_rejected(self):
        with pytest.raises(ValueError, match="not in 'classes'"):
            LabelSpace.from_config(["Housing"], {"Hosuing": "typo"})

    def test_unknown_mapping_field_rejected(self):
        with pytest.raises(ValueError, match="unknown field"):
            LabelSpace.from_config(["Housing"], {"Housing": {"desc": "x"}})

    def test_classes_must_be_a_list(self):
        with pytest.raises(ValueError, match="list"):
            LabelSpace.from_config("Housing")

    def test_duplicate_keys_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            LabelSpace.from_config(["Mood/anxiety", "Mood anxiety"])

    def test_invalid_key_rejected(self):
        with pytest.raises(ValueError, match="snake_case"):
            LabelSpace.from_config(["Housing"], {"Housing": {"key": "Housing-Status"}})

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            LabelSpace.from_config([])


class TestLookups:
    space = LabelSpace([
        OutputDimension("mood", "Mood"),
        OutputDimension("housing", "Housing status"),
    ])

    def test_normalize_accepts_key_display_and_case_insensitive_display(self):
        assert self.space.normalize("mood") == "mood"
        assert self.space.normalize("Housing status") == "housing"
        assert self.space.normalize("HOUSING STATUS") == "housing"

    def test_normalize_unknown_raises(self):
        with pytest.raises(UnknownOutputDimensionError, match="Valid keys"):
            self.space.normalize("diagnoses")

    def test_key_for_display_is_case_sensitive(self):
        with pytest.raises(UnknownOutputDimensionError):
            self.space.key_for_display("mood")

    def test_manifest(self):
        assert self.space.manifest() == {"mood": "Mood", "housing": "Housing status"}


@pytest.mark.parametrize(
    "name, key",
    [
        ("Mood & anxiety", "mood_anxiety"),
        ("  Misc. sensitive ", "misc_sensitive"),
        ("2nd opinion", "c_2nd_opinion"),
    ],
)
def test_slugify_key(name, key):
    assert slugify_key(name) == key
