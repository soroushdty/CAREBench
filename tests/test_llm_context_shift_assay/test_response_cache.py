"""
Tests for tracks/reasoning/response_cache.py — ResponseCache class.

Covers:
  - 8.1 Directory structure: {cache_dir}/{model_id_slug}/{condition}/{patient_id}__{item_hash}.json
  - 8.2 get() returns cached LLMResponse or None
  - 8.3 put() writes JSON to the correct cache file
  - 8.4 get_status() returns completed, failed, and pending lists
  - 8.5 Cache files include all LLMResponse fields
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tracks.reasoning.llm_client import LLMClient, LLMResponse
from tracks.reasoning.response_cache import ResponseCache, _item_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for the cache."""
    return tmp_path / "cache"


def make_response(
    model_id: str = "org/model",
    condition: str = "context_free",
    patient_id: str = "P001",
    item_text: str = "patient has hypertension",
    error: str | None = None,
) -> LLMResponse:
    """Build a minimal LLMResponse for testing."""
    return LLMResponse(
        raw_text='{"behavioral_health": 0.5}' if error is None else "",
        model_id=model_id,
        condition=condition,
        patient_id=patient_id,
        item_text=item_text,
        prompt_hash="abcdef1234567890",
        timestamp="2024-01-01T00:00:00.000000Z",
        backend="dry_run",
        token_usage=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# 8.1 Directory structure
# ---------------------------------------------------------------------------


class TestDirectoryStructure:
    """Tests that cache files are stored in the correct directory structure."""

    def test_cache_file_path_uses_model_id_slug(self, cache_dir):
        """Cache file is stored under {model_id_slug}/ directory."""
        cache = ResponseCache(cache_dir)
        response = make_response(model_id="org/model-name")
        cache.put(response)

        slug = LLMClient.model_id_slug("org/model-name")
        assert (cache_dir / slug).is_dir()

    def test_cache_file_path_uses_condition(self, cache_dir):
        """Cache file is stored under {condition}/ subdirectory."""
        cache = ResponseCache(cache_dir)
        response = make_response(model_id="org/model", condition="correct_context")
        cache.put(response)

        slug = LLMClient.model_id_slug("org/model")
        assert (cache_dir / slug / "correct_context").is_dir()

    def test_cache_file_name_uses_patient_id_and_item_hash(self, cache_dir):
        """Cache filename is {patient_id}__{item_hash}.json."""
        cache = ResponseCache(cache_dir)
        item_text = "patient has hypertension"
        response = make_response(
            model_id="org/model",
            condition="context_free",
            patient_id="P001",
            item_text=item_text,
        )
        cache.put(response)

        slug = LLMClient.model_id_slug("org/model")
        ih = _item_hash(item_text)
        expected_file = cache_dir / slug / "context_free" / f"P001__{ih}.json"
        assert expected_file.exists()

    def test_item_hash_is_12_char_sha256(self):
        """_item_hash returns first 12 chars of SHA-256 hex digest."""
        text = "some item text"
        expected = hashlib.sha256(text.encode()).hexdigest()[:12]
        assert _item_hash(text) == expected
        assert len(_item_hash(text)) == 12

    def test_full_path_structure(self, cache_dir):
        """Full path matches {cache_dir}/{slug}/{condition}/{patient_id}__{item_hash}.json."""
        cache = ResponseCache(cache_dir)
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        condition = "shuffled_context"
        patient_id = "P007"
        item_text = "allergy to penicillin"

        response = make_response(
            model_id=model_id,
            condition=condition,
            patient_id=patient_id,
            item_text=item_text,
        )
        cache.put(response)

        slug = LLMClient.model_id_slug(model_id)
        ih = _item_hash(item_text)
        expected = cache_dir / slug / condition / f"{patient_id}__{ih}.json"
        assert expected.exists()


# ---------------------------------------------------------------------------
# 8.2 get() method
# ---------------------------------------------------------------------------


class TestGet:
    """Tests for ResponseCache.get()."""

    def test_get_returns_none_when_not_cached(self, cache_dir):
        """get() returns None when no cache file exists."""
        cache = ResponseCache(cache_dir)
        result = cache.get("org/model", "context_free", "P001", "some item")
        assert result is None

    def test_get_returns_llm_response_when_cached(self, cache_dir):
        """get() returns an LLMResponse when the cache file exists."""
        cache = ResponseCache(cache_dir)
        response = make_response()
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result is not None
        assert isinstance(result, LLMResponse)

    def test_get_returns_correct_raw_text(self, cache_dir):
        """get() returns the correct raw_text from the cached response."""
        cache = ResponseCache(cache_dir)
        response = make_response()
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result.raw_text == response.raw_text

    def test_get_returns_correct_all_fields(self, cache_dir):
        """get() reconstructs all LLMResponse fields correctly."""
        cache = ResponseCache(cache_dir)
        response = LLMResponse(
            raw_text='{"behavioral_health": 0.9}',
            model_id="org/model",
            condition="correct_context",
            patient_id="P003",
            item_text="diabetes diagnosis",
            prompt_hash="1234567890abcdef",
            timestamp="2024-06-15T12:30:00.000000Z",
            backend="huggingface",
            token_usage={"generated_tokens": 128, "prefill_tokens": 64},
            error=None,
        )
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result.raw_text == response.raw_text
        assert result.model_id == response.model_id
        assert result.condition == response.condition
        assert result.patient_id == response.patient_id
        assert result.item_text == response.item_text
        assert result.prompt_hash == response.prompt_hash
        assert result.timestamp == response.timestamp
        assert result.backend == response.backend
        assert result.token_usage == response.token_usage
        assert result.error == response.error

    def test_get_returns_none_for_different_condition(self, cache_dir):
        """get() returns None when condition differs from cached entry."""
        cache = ResponseCache(cache_dir)
        response = make_response(condition="context_free")
        cache.put(response)

        result = cache.get(
            response.model_id,
            "correct_context",  # different condition
            response.patient_id,
            response.item_text,
        )
        assert result is None

    def test_get_returns_none_for_different_patient(self, cache_dir):
        """get() returns None when patient_id differs from cached entry."""
        cache = ResponseCache(cache_dir)
        response = make_response(patient_id="P001")
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            "P002",  # different patient
            response.item_text,
        )
        assert result is None

    def test_get_returns_none_for_different_item_text(self, cache_dir):
        """get() returns None when item_text differs (different hash)."""
        cache = ResponseCache(cache_dir)
        response = make_response(item_text="item one")
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            "item two",  # different item
        )
        assert result is None

    def test_get_returns_none_for_corrupted_file(self, cache_dir, tmp_path):
        """get() returns None when the cache file is corrupted JSON."""
        cache = ResponseCache(cache_dir)
        response = make_response()
        cache.put(response)

        # Corrupt the file
        slug = LLMClient.model_id_slug(response.model_id)
        ih = _item_hash(response.item_text)
        path = cache_dir / slug / response.condition / f"{response.patient_id}__{ih}.json"
        path.write_text("not valid json", encoding="utf-8")

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result is None

    def test_get_returns_failed_response(self, cache_dir):
        """get() returns a failed LLMResponse (error is not None) from cache."""
        cache = ResponseCache(cache_dir)
        response = make_response(error="API timeout after 3 retries")
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result is not None
        assert result.error == "API timeout after 3 retries"


# ---------------------------------------------------------------------------
# 8.3 put() method
# ---------------------------------------------------------------------------


class TestPut:
    """Tests for ResponseCache.put()."""

    def test_put_creates_cache_file(self, cache_dir):
        """put() creates the cache JSON file."""
        cache = ResponseCache(cache_dir)
        response = make_response()
        cache.put(response)

        slug = LLMClient.model_id_slug(response.model_id)
        ih = _item_hash(response.item_text)
        path = cache_dir / slug / response.condition / f"{response.patient_id}__{ih}.json"
        assert path.exists()

    def test_put_creates_intermediate_directories(self, cache_dir):
        """put() creates all intermediate directories."""
        cache = ResponseCache(cache_dir)
        response = make_response(model_id="new-org/new-model", condition="shuffled_context")
        cache.put(response)

        slug = LLMClient.model_id_slug("new-org/new-model")
        assert (cache_dir / slug / "shuffled_context").is_dir()

    def test_put_writes_valid_json(self, cache_dir):
        """put() writes valid JSON to the cache file."""
        cache = ResponseCache(cache_dir)
        response = make_response()
        cache.put(response)

        slug = LLMClient.model_id_slug(response.model_id)
        ih = _item_hash(response.item_text)
        path = cache_dir / slug / response.condition / f"{response.patient_id}__{ih}.json"

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_put_overwrites_existing_file(self, cache_dir):
        """put() overwrites an existing cache file with new data."""
        cache = ResponseCache(cache_dir)
        response1 = make_response()
        response1_modified = LLMResponse(
            raw_text='{"behavioral_health": 0.9}',
            model_id=response1.model_id,
            condition=response1.condition,
            patient_id=response1.patient_id,
            item_text=response1.item_text,
            prompt_hash="newprompt1234567",
            timestamp="2024-12-01T00:00:00.000000Z",
            backend="dry_run",
            token_usage=None,
            error=None,
        )
        cache.put(response1)
        cache.put(response1_modified)

        result = cache.get(
            response1.model_id,
            response1.condition,
            response1.patient_id,
            response1.item_text,
        )
        assert result.raw_text == '{"behavioral_health": 0.9}'
        assert result.prompt_hash == "newprompt1234567"

    def test_put_different_conditions_stored_separately(self, cache_dir):
        """put() stores different conditions in separate files."""
        cache = ResponseCache(cache_dir)
        r_free = make_response(condition="context_free")
        r_correct = make_response(condition="correct_context")
        r_shuffled = make_response(condition="shuffled_context")

        cache.put(r_free)
        cache.put(r_correct)
        cache.put(r_shuffled)

        slug = LLMClient.model_id_slug(r_free.model_id)
        ih = _item_hash(r_free.item_text)
        pid = r_free.patient_id

        assert (cache_dir / slug / "context_free" / f"{pid}__{ih}.json").exists()
        assert (cache_dir / slug / "correct_context" / f"{pid}__{ih}.json").exists()
        assert (cache_dir / slug / "shuffled_context" / f"{pid}__{ih}.json").exists()


# ---------------------------------------------------------------------------
# 8.5 All LLMResponse fields in cache file
# ---------------------------------------------------------------------------


class TestCacheFileFields:
    """Tests that cache files include all required LLMResponse fields."""

    REQUIRED_FIELDS = [
        "raw_text", "model_id", "condition", "patient_id", "item_text",
        "prompt_hash", "timestamp", "backend", "token_usage", "error",
    ]

    def test_all_fields_present_in_cache_file(self, cache_dir):
        """Cache JSON file contains all 10 required LLMResponse fields."""
        cache = ResponseCache(cache_dir)
        response = make_response()
        cache.put(response)

        slug = LLMClient.model_id_slug(response.model_id)
        ih = _item_hash(response.item_text)
        path = cache_dir / slug / response.condition / f"{response.patient_id}__{ih}.json"

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing field: {field!r}"

    def test_token_usage_dict_serialized_correctly(self, cache_dir):
        """token_usage dict is serialized and deserialized correctly."""
        cache = ResponseCache(cache_dir)
        response = LLMResponse(
            raw_text="{}",
            model_id="org/model",
            condition="context_free",
            patient_id="P001",
            item_text="test item",
            prompt_hash="abcdef1234567890",
            timestamp="2024-01-01T00:00:00.000000Z",
            backend="huggingface",
            token_usage={"generated_tokens": 256, "prefill_tokens": 32},
            error=None,
        )
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result.token_usage == {"generated_tokens": 256, "prefill_tokens": 32}

    def test_null_error_serialized_as_none(self, cache_dir):
        """error=None is stored as JSON null and deserialized as None."""
        cache = ResponseCache(cache_dir)
        response = make_response(error=None)
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result.error is None

    def test_error_string_serialized_correctly(self, cache_dir):
        """error string is stored and deserialized correctly."""
        cache = ResponseCache(cache_dir)
        response = make_response(error="Connection timeout")
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result.error == "Connection timeout"

    def test_null_token_usage_serialized_as_none(self, cache_dir):
        """token_usage=None is stored as JSON null and deserialized as None."""
        cache = ResponseCache(cache_dir)
        response = make_response()  # token_usage=None by default
        cache.put(response)

        result = cache.get(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        assert result.token_usage is None


# ---------------------------------------------------------------------------
# 8.4 get_status() method
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for ResponseCache.get_status()."""

    def test_get_status_empty_cache_no_expected(self, cache_dir):
        """get_status() returns empty lists when cache is empty and no expected calls."""
        cache = ResponseCache(cache_dir)
        status = cache.get_status()
        assert status == {"completed": [], "failed": [], "pending": []}

    def test_get_status_completed_entry(self, cache_dir):
        """Successful response (error=None) appears in completed list."""
        cache = ResponseCache(cache_dir)
        response = make_response(error=None)
        cache.put(response)

        status = cache.get_status()
        key = (response.model_id, response.condition, response.patient_id, response.item_text)
        assert key in status["completed"]
        assert key not in status["failed"]

    def test_get_status_failed_entry(self, cache_dir):
        """Failed response (error is not None) appears in failed list."""
        cache = ResponseCache(cache_dir)
        response = make_response(error="API error")
        cache.put(response)

        status = cache.get_status()
        key = (response.model_id, response.condition, response.patient_id, response.item_text)
        assert key in status["failed"]
        assert key not in status["completed"]

    def test_get_status_pending_computed_from_expected(self, cache_dir):
        """Pending = expected_calls minus completed and failed."""
        expected = [
            ("org/model", "context_free", "P001", "item one"),
            ("org/model", "context_free", "P002", "item two"),
            ("org/model", "correct_context", "P001", "item one"),
        ]
        cache = ResponseCache(cache_dir, expected_calls=expected)

        # Cache only the first call
        response = make_response(
            model_id="org/model",
            condition="context_free",
            patient_id="P001",
            item_text="item one",
        )
        cache.put(response)

        status = cache.get_status()
        assert ("org/model", "context_free", "P001", "item one") in status["completed"]
        assert ("org/model", "context_free", "P002", "item two") in status["pending"]
        assert ("org/model", "correct_context", "P001", "item one") in status["pending"]

    def test_get_status_no_pending_when_all_done(self, cache_dir):
        """Pending is empty when all expected calls are completed or failed."""
        expected = [
            ("org/model", "context_free", "P001", "item one"),
            ("org/model", "context_free", "P002", "item two"),
        ]
        cache = ResponseCache(cache_dir, expected_calls=expected)

        cache.put(make_response(model_id="org/model", condition="context_free",
                                patient_id="P001", item_text="item one"))
        cache.put(make_response(model_id="org/model", condition="context_free",
                                patient_id="P002", item_text="item two", error="fail"))

        status = cache.get_status()
        assert status["pending"] == []
        assert len(status["completed"]) == 1
        assert len(status["failed"]) == 1

    def test_get_status_pending_empty_when_no_expected(self, cache_dir):
        """Pending is always empty when no expected_calls provided."""
        cache = ResponseCache(cache_dir)
        cache.put(make_response())

        status = cache.get_status()
        assert status["pending"] == []

    def test_get_status_multiple_models(self, cache_dir):
        """get_status() correctly handles responses from multiple models."""
        cache = ResponseCache(cache_dir)
        r1 = make_response(model_id="org/model-a", condition="context_free",
                           patient_id="P001", item_text="item one")
        r2 = make_response(model_id="org/model-b", condition="context_free",
                           patient_id="P001", item_text="item one")
        cache.put(r1)
        cache.put(r2)

        status = cache.get_status()
        assert len(status["completed"]) == 2
        assert ("org/model-a", "context_free", "P001", "item one") in status["completed"]
        assert ("org/model-b", "context_free", "P001", "item one") in status["completed"]

    def test_get_status_nonexistent_cache_dir(self, tmp_path):
        """get_status() returns empty lists when cache_dir does not exist."""
        cache = ResponseCache(tmp_path / "nonexistent")
        status = cache.get_status()
        assert status == {"completed": [], "failed": [], "pending": []}

    def test_get_status_failed_not_in_pending(self, cache_dir):
        """Failed calls are not included in pending even if in expected_calls."""
        expected = [("org/model", "context_free", "P001", "item one")]
        cache = ResponseCache(cache_dir, expected_calls=expected)

        # Cache a failed response matching the expected call
        cache.put(make_response(
            model_id="org/model",
            condition="context_free",
            patient_id="P001",
            item_text="item one",
            error="timeout",
        ))

        status = cache.get_status()
        assert ("org/model", "context_free", "P001", "item one") not in status["pending"]
        assert ("org/model", "context_free", "P001", "item one") in status["failed"]
