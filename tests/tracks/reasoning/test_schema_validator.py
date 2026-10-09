"""
Tests for tracks/reasoning/schema_validator.py.

Covers:
  - Valid JSON responses pass schema validation (14.2)
  - Responses with missing keys raise MissingKeyError (14.3)
  - Responses with out-of-range values raise RangeError (14.4)
  - Malformed JSON raises JSONParseError (14.5)
"""

from __future__ import annotations

import json

import pytest

from tracks.reasoning.schema_validator import (
    REQUIRED_KEYS,
    JSONParseError,
    MissingKeyError,
    RangeError,
    validate_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_dict(value: float = 0.5) -> dict[str, float]:
    """Return a valid score dict with all 10 required keys."""
    return {key: value for key in REQUIRED_KEYS}


def _make_valid_json(value: float = 0.5) -> str:
    return json.dumps(_make_valid_dict(value))


# ---------------------------------------------------------------------------
# 14.2 — Valid responses pass validation
# ---------------------------------------------------------------------------


class TestValidResponsePasses:
    def test_all_keys_mid_range(self):
        """Well-formed JSON with all 10 keys and values of 0.5 passes."""
        result = validate_response(_make_valid_json(0.5))
        assert set(result.keys()) == set(REQUIRED_KEYS)
        assert all(v == 0.5 for v in result.values())

    def test_boundary_zero(self):
        """Value of exactly 0.0 is valid (inclusive lower bound)."""
        result = validate_response(_make_valid_json(0.0))
        assert all(v == 0.0 for v in result.values())

    def test_boundary_one(self):
        """Value of exactly 1.0 is valid (inclusive upper bound)."""
        result = validate_response(_make_valid_json(1.0))
        assert all(v == 1.0 for v in result.values())

    def test_returns_float_values(self):
        """All returned values are Python floats."""
        result = validate_response(_make_valid_json(0.75))
        assert all(isinstance(v, float) for v in result.values())

    def test_integer_values_cast_to_float(self):
        """Integer values (0, 1) are accepted and cast to float."""
        d = {key: 0 for key in REQUIRED_KEYS}
        d["diagnoses"] = 1
        result = validate_response(json.dumps(d))
        assert isinstance(result["diagnoses"], float)
        assert result["diagnoses"] == 1.0

    def test_exactly_ten_keys_returned(self):
        """Result contains exactly 10 keys."""
        result = validate_response(_make_valid_json())
        assert len(result) == 10

    def test_extra_keys_in_response_ignored(self):
        """Extra keys beyond the 10 required ones are silently ignored."""
        d = _make_valid_dict()
        d["extra_key"] = 0.9
        result = validate_response(json.dumps(d))
        assert "extra_key" not in result
        assert len(result) == 10

    def test_strips_leading_trailing_whitespace(self):
        """Leading/trailing whitespace around the JSON is handled."""
        raw = "   \n" + _make_valid_json() + "\n   "
        result = validate_response(raw)
        assert len(result) == 10

    def test_strips_markdown_json_fence(self):
        """Markdown ```json ... ``` code fence is stripped."""
        raw = "```json\n" + _make_valid_json() + "\n```"
        result = validate_response(raw)
        assert len(result) == 10

    def test_strips_plain_markdown_fence(self):
        """Markdown ``` ... ``` code fence (no language tag) is stripped."""
        raw = "```\n" + _make_valid_json() + "\n```"
        result = validate_response(raw)
        assert len(result) == 10

    def test_extracts_json_from_surrounding_text(self):
        """JSON object embedded in surrounding prose is extracted."""
        raw = (
            "Here is my response:\n"
            + _make_valid_json()
            + "\nThat's all."
        )
        result = validate_response(raw)
        assert len(result) == 10

    def test_required_keys_order_preserved(self):
        """Returned dict contains all REQUIRED_KEYS."""
        result = validate_response(_make_valid_json())
        for key in REQUIRED_KEYS:
            assert key in result


# ---------------------------------------------------------------------------
# 14.3 — Missing keys raise MissingKeyError
# ---------------------------------------------------------------------------


class TestMissingKeyFails:
    def test_single_missing_key(self):
        """Omitting one required key raises MissingKeyError."""
        d = _make_valid_dict()
        del d["genetics"]
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response(json.dumps(d))
        assert "genetics" in exc_info.value.missing_keys

    def test_multiple_missing_keys(self):
        """Omitting several required keys raises MissingKeyError listing all."""
        d = _make_valid_dict()
        del d["genetics"]
        del d["violence"]
        del d["other"]
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response(json.dumps(d))
        assert set(exc_info.value.missing_keys) == {"genetics", "violence", "other"}

    def test_all_keys_missing(self):
        """An empty JSON object raises MissingKeyError with all 10 keys."""
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response("{}")
        assert len(exc_info.value.missing_keys) == 10

    def test_missing_keys_attribute_is_list(self):
        """missing_keys attribute is a list."""
        d = _make_valid_dict()
        del d["diagnoses"]
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response(json.dumps(d))
        assert isinstance(exc_info.value.missing_keys, list)

    def test_wrong_key_name_treated_as_missing(self):
        """A key with a typo is treated as missing."""
        d = _make_valid_dict()
        del d["behavioral_health"]
        d["behavioural_health"] = 0.5  # British spelling — wrong key
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response(json.dumps(d))
        assert "behavioral_health" in exc_info.value.missing_keys


# ---------------------------------------------------------------------------
# 14.4 — Out-of-range values raise RangeError
# ---------------------------------------------------------------------------


class TestOutOfRangeFails:
    def test_value_above_one(self):
        """A value of 1.1 raises RangeError."""
        d = _make_valid_dict()
        d["diagnoses"] = 1.1
        with pytest.raises(RangeError) as exc_info:
            validate_response(json.dumps(d))
        assert exc_info.value.key == "diagnoses"
        assert exc_info.value.value == pytest.approx(1.1)

    def test_value_below_zero(self):
        """A value of -0.1 raises RangeError."""
        d = _make_valid_dict()
        d["violence"] = -0.1
        with pytest.raises(RangeError) as exc_info:
            validate_response(json.dumps(d))
        assert exc_info.value.key == "violence"

    def test_range_error_stores_key_and_value(self):
        """RangeError stores the offending key and value as attributes."""
        d = _make_valid_dict()
        d["other"] = 2.0
        with pytest.raises(RangeError) as exc_info:
            validate_response(json.dumps(d))
        assert exc_info.value.key == "other"
        assert exc_info.value.value == pytest.approx(2.0)

    def test_non_numeric_value_raises_range_error(self):
        """A string value raises RangeError (not numeric)."""
        d = _make_valid_dict()
        d["medications"] = "high"
        with pytest.raises(RangeError):
            validate_response(json.dumps(d))

    def test_boolean_value_raises_range_error(self):
        """A boolean value (True/False) raises RangeError."""
        d = _make_valid_dict()
        d["genetics"] = True  # bool is subclass of int in Python
        with pytest.raises(RangeError):
            validate_response(json.dumps(d))

    def test_null_value_raises_range_error(self):
        """A null/None value raises RangeError."""
        d = _make_valid_dict()
        d["disabilities"] = None
        with pytest.raises(RangeError):
            validate_response(json.dumps(d))


# ---------------------------------------------------------------------------
# 14.5 — Malformed JSON raises JSONParseError
# ---------------------------------------------------------------------------


class TestMalformedJsonFails:
    def test_plain_text_raises_json_parse_error(self):
        """Plain English text raises JSONParseError."""
        with pytest.raises(JSONParseError):
            validate_response("I cannot classify this item.")

    def test_truncated_json_raises_json_parse_error(self):
        """Truncated JSON raises JSONParseError."""
        with pytest.raises(JSONParseError):
            validate_response('{"behavioral_health": 0.5, "diagnoses":')

    def test_empty_string_raises_json_parse_error(self):
        """Empty string raises JSONParseError."""
        with pytest.raises(JSONParseError):
            validate_response("")

    def test_whitespace_only_raises_json_parse_error(self):
        """Whitespace-only string raises JSONParseError."""
        with pytest.raises(JSONParseError):
            validate_response("   \n\t  ")

    def test_json_array_raises_json_parse_error(self):
        """A JSON array (not object) raises JSONParseError."""
        with pytest.raises(JSONParseError):
            validate_response("[0.5, 0.3, 0.1]")

    def test_json_parse_error_stores_raw_text(self):
        """JSONParseError stores the original raw_text as an attribute."""
        raw = "not valid json at all"
        with pytest.raises(JSONParseError) as exc_info:
            validate_response(raw)
        assert exc_info.value.raw_text == raw

    def test_single_quotes_raises_json_parse_error(self):
        """JSON with single quotes (invalid JSON) raises JSONParseError."""
        raw = "{'behavioral_health': 0.5}"
        with pytest.raises(JSONParseError):
            validate_response(raw)


# ---------------------------------------------------------------------------
# Thinking-trace prefix stripping
# ---------------------------------------------------------------------------


class TestThinkingTraceStripping:
    """Thinking-trace prefixes emitted by reasoning models are stripped."""

    def test_unused94_truncated_no_json_raises_json_parse_error(self):
        """Truncated <unused94> trace with no JSON raises JSONParseError."""
        raw = "<unused94>thought\nThe user wants me to classify"
        with pytest.raises(JSONParseError):
            validate_response(raw)

    def test_unused94_with_valid_json_parses_correctly(self):
        """<unused94> trace followed by valid JSON parses to a full 10-key dict."""
        valid_json = _make_valid_json(0.7)
        raw = f"<unused94>thought\nSome long reasoning...\n{valid_json}"
        result = validate_response(raw)
        assert set(result.keys()) == set(REQUIRED_KEYS)
        assert all(v == pytest.approx(0.7) for v in result.values())

    def test_think_tag_with_valid_json_parses_correctly(self):
        """<think>...</think> prefix followed by valid JSON parses correctly."""
        valid_json = _make_valid_json(0.3)
        raw = f"<think>I need to classify this item carefully.</think>{valid_json}"
        result = validate_response(raw)
        assert set(result.keys()) == set(REQUIRED_KEYS)
        assert all(v == pytest.approx(0.3) for v in result.values())

    def test_plain_json_still_works(self):
        """Plain JSON without any thinking trace still passes (regression)."""
        result = validate_response(_make_valid_json(0.5))
        assert len(result) == 10

    def test_fenced_json_still_works(self):
        """Fenced ```json ... ``` response still passes (regression)."""
        raw = "```json\n" + _make_valid_json(0.5) + "\n```"
        result = validate_response(raw)
        assert len(result) == 10

    def test_thinking_prefix_missing_key_raises_missing_key_error(self):
        """Thinking trace with JSON missing a required key raises MissingKeyError."""
        d = _make_valid_dict()
        del d["genetics"]
        raw = "<think>reasoning</think>" + json.dumps(d)
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response(raw)
        assert "genetics" in exc_info.value.missing_keys

    def test_thinking_prefix_out_of_range_raises_range_error(self):
        """Thinking trace with a value of 1.5 raises RangeError."""
        d = _make_valid_dict()
        d["diagnoses"] = 1.5
        raw = "<unused94>thought\nreasoning...\n" + json.dumps(d)
        with pytest.raises(RangeError) as exc_info:
            validate_response(raw)
        assert exc_info.value.key == "diagnoses"
