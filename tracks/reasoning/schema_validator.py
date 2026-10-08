"""
Schema Validator for the LLM Context-Shift Assay.

Validates LLM JSON responses against the strict privacy category schema.
Handles common LLM output artifacts (markdown code fences, surrounding text).

Raises:
    JSONParseError  — if json.loads fails
    MissingKeyError — if any of the 10 required keys are absent
    RangeError      — if any value is outside [0.0, 1.0]

Returns a dict[str, float] with exactly 10 keys on success.
"""

from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# Required keys (canonical Privacy_Category names, glossary order)
# ---------------------------------------------------------------------------

REQUIRED_KEYS: list[str] = [
    "behavioral_health",
    "diagnoses",
    "disabilities",
    "infectious_diseases",
    "genetics",
    "medications",
    "sexual_reproductive_health",
    "social_determinants_of_health",
    "violence",
    "other",
]

# Set form for O(1) membership checks
_REQUIRED_KEYS_SET: set[str] = set(REQUIRED_KEYS)


# ---------------------------------------------------------------------------
# Display-label ↔ canonical-key mappings
# ---------------------------------------------------------------------------

DISPLAY_TO_CANONICAL: dict[str, str] = {
    "Behavioral health": "behavioral_health",
    "Diagnoses": "diagnoses",
    "Disabilities": "disabilities",
    "Infectious diseases": "infectious_diseases",
    "Genetics": "genetics",
    "Medications": "medications",
    "Sexual and reproductive health": "sexual_reproductive_health",
    "Social determinants of health": "social_determinants_of_health",
    "Violence": "violence",
    "Other": "other",
}

CANONICAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_CANONICAL.items()}


def normalize_category_key(name: str) -> str:
    """Normalize a display label or canonical key to canonical snake_case form.

    Parameters
    ----------
    name : str
        Either a human-readable display label (e.g. ``"Behavioral health"``)
        or a canonical key (e.g. ``"behavioral_health"``).

    Returns
    -------
    str
        The canonical snake_case key.

    Raises
    ------
    KeyError
        If *name* is not a recognized display label or canonical key.
    """
    if name in DISPLAY_TO_CANONICAL:
        return DISPLAY_TO_CANONICAL[name]
    if name in _REQUIRED_KEYS_SET:
        return name
    raise KeyError(
        f"Unknown output-dimension name: {name!r}. "
        f"Expected one of the canonical keys {REQUIRED_KEYS} "
        f"or a display label from {list(DISPLAY_TO_CANONICAL.keys())}."
    )


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class JSONParseError(ValueError):
    """Raised when the LLM response cannot be parsed as JSON.

    Attributes
    ----------
    raw_text : str
        The original raw response text that failed to parse.
    """

    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        super().__init__(
            f"Failed to parse LLM response as JSON. "
            f"Raw text (first 200 chars): {raw_text[:200]!r}"
        )


class MissingKeyError(ValueError):
    """Raised when one or more required Privacy_Category keys are absent.

    Attributes
    ----------
    missing_keys : list[str]
        Names of the keys that were not found in the parsed JSON.
    """

    def __init__(self, missing_keys: list[str]) -> None:
        self.missing_keys = missing_keys
        super().__init__(
            f"LLM response is missing required keys: {missing_keys}"
        )


class RangeError(ValueError):
    """Raised when a Privacy_Category value is outside [0.0, 1.0].

    Attributes
    ----------
    key : str
        The Privacy_Category key whose value is out of range.
    value : float
        The out-of-range value.
    """

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value
        super().__init__(
            f"Value for key {key!r} is out of range [0.0, 1.0]: {value}"
        )


# ---------------------------------------------------------------------------
# Public validation function
# ---------------------------------------------------------------------------


def validate_response(raw_text: str) -> dict[str, float]:
    """Parse and validate an LLM JSON response.

    Processing steps:
    1. Strip leading/trailing whitespace.
    2. Strip markdown code fences (```json ... ``` or ``` ... ```).
    3. Extract the first ``{...}`` block if the response has surrounding text.
    4. Attempt ``json.loads``; raise :class:`JSONParseError` on failure.
    5. Check all 10 required keys are present; raise :class:`MissingKeyError`
       with the list of missing keys if any are absent.
    6. Check all values are numeric (int or float) and in [0.0, 1.0]; raise
       :class:`RangeError` on the first violation found.
    7. Return ``dict[str, float]`` with exactly the 10 required keys (values
       cast to float).

    Parameters
    ----------
    raw_text : str
        The raw text returned by the LLM.

    Returns
    -------
    dict[str, float]
        Validated score dictionary with exactly 10 Privacy_Category keys.

    Raises
    ------
    JSONParseError
        If the text cannot be parsed as JSON after cleaning.
    MissingKeyError
        If any of the 10 required keys are absent from the parsed object.
    RangeError
        If any value is not numeric or is outside [0.0, 1.0].
    """
    # Step 1: strip leading/trailing whitespace
    text = raw_text.strip()

    # Step 2: strip markdown code fences
    # Handles ```json ... ```, ``` ... ```, and variants with optional language tag
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Step 2.5: strip thinking-trace prefixes emitted by reasoning models
    # (e.g. MedGemma 1.5 uses <unused94>thought\n... before its JSON answer)
    _THINKING_MARKERS = ("<unused94>", "<think>", "<thinking>")
    _earliest_marker_pos = -1
    for _marker in _THINKING_MARKERS:
        _pos = text.find(_marker)
        if _pos != -1 and (_earliest_marker_pos == -1 or _pos < _earliest_marker_pos):
            _earliest_marker_pos = _pos
    if _earliest_marker_pos != -1:
        _brace_pos = text.find("{", _earliest_marker_pos)
        if _brace_pos != -1:
            text = text[_brace_pos:]

    # Step 3: extract the first {...} block if surrounding text is present
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if brace_match:
        # Use a more robust extraction: find the outermost balanced braces
        text = _extract_first_json_object(text)

    # Step 4: attempt json.loads
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        raise JSONParseError(raw_text)

    if not isinstance(parsed, dict):
        raise JSONParseError(raw_text)

    # Step 5: check all required keys are present
    missing_keys = [key for key in REQUIRED_KEYS if key not in parsed]
    if missing_keys:
        raise MissingKeyError(missing_keys)

    # Step 6: check all values are numeric and in [0.0, 1.0]
    for key in REQUIRED_KEYS:
        value = parsed[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RangeError(key, value)
        float_value = float(value)
        if float_value < 0.0 or float_value > 1.0:
            raise RangeError(key, float_value)

    # Step 7: return validated dict[str, float] with exactly the 10 required keys
    return {key: float(parsed[key]) for key in REQUIRED_KEYS}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced ``{...}`` block from *text*.

    Walks the string character by character to find the outermost balanced
    brace pair, handling nested objects correctly.

    Parameters
    ----------
    text : str
        Input string that may contain a JSON object surrounded by other text.

    Returns
    -------
    str
        The substring from the first ``{`` to its matching ``}``, inclusive.
        If no balanced block is found, returns *text* unchanged so that the
        caller's ``json.loads`` will raise a descriptive error.
    """
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]

    # No balanced block found — return as-is
    return text
