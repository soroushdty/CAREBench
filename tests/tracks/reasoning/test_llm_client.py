"""
Tests for tracks/reasoning/llm_client.py — LLMClient class.

Covers:
  - dry_run backend returns deterministic mock JSON with all 10 categories = 0.5
  - LLMResponse dataclass has all required metadata fields
  - Retry loop records FailedCall on exhaustion (huggingface backend, mocked)
  - Failed calls are never silently dropped
  - model_id_slug replaces '/' with '__'
  - prompt_hash is a 16-character hex string
  - Timestamp is ISO 8601 format
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from tracks.reasoning.llm_client import FailedCall, LLMClient, LLMResponse
from tracks.reasoning.schema_validator import REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dry_run_cfg() -> dict:
    """Minimal config for dry_run backend."""
    return {
        "backend": "dry_run",
        "retry_limit": 3,
        "temperature": 0.0,
        "max_tokens": 512,
        "top_p": 1.0,
        "seed": 42,
    }


@pytest.fixture
def hf_cfg() -> dict:
    """Minimal config for huggingface backend."""
    return {
        "backend": "huggingface",
        "retry_limit": 3,
        "temperature": 0.0,
        "max_tokens": 512,
        "top_p": 1.0,
        "seed": 42,
        "hf_token": "test-token",
    }


@pytest.fixture
def sample_prompt() -> str:
    return "Classify this EHR item: patient has hypertension."


@pytest.fixture
def call_kwargs() -> dict:
    return {
        "model_id": "test-org/test-model",
        "condition": "context_free",
        "patient_id": "P001",
        "item_text": "patient has hypertension",
    }


# ---------------------------------------------------------------------------
# 7.3 dry_run backend
# ---------------------------------------------------------------------------


class TestDryRunBackend:
    """Tests for the dry_run backend."""

    def test_dry_run_returns_llm_response(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run call returns an LLMResponse instance."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert isinstance(response, LLMResponse)

    def test_dry_run_raw_text_is_valid_json(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run raw_text is valid JSON."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        parsed = json.loads(response.raw_text)
        assert isinstance(parsed, dict)

    def test_dry_run_all_ten_categories_set_to_0_5(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run response has all 10 required categories set to 0.5."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        parsed = json.loads(response.raw_text)
        assert set(parsed.keys()) == set(REQUIRED_KEYS)
        for key in REQUIRED_KEYS:
            assert parsed[key] == 0.5, f"Expected 0.5 for {key!r}, got {parsed[key]}"

    def test_dry_run_is_deterministic(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run returns the same raw_text on repeated calls."""
        client = LLMClient(dry_run_cfg)
        r1 = client.call(sample_prompt, **call_kwargs)
        r2 = client.call(sample_prompt, **call_kwargs)
        assert r1.raw_text == r2.raw_text

    def test_dry_run_no_error(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run response has error=None."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.error is None

    def test_dry_run_backend_field(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run response has backend='dry_run'."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.backend == "dry_run"

    def test_dry_run_token_usage_is_none(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run response has token_usage=None (no external call)."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.token_usage is None

    def test_dry_run_no_failed_calls(self, dry_run_cfg, sample_prompt, call_kwargs):
        """dry_run never records failed calls."""
        client = LLMClient(dry_run_cfg)
        client.call(sample_prompt, **call_kwargs)
        assert client.get_failed_calls() == []


# ---------------------------------------------------------------------------
# 7.5 LLMResponse metadata fields
# ---------------------------------------------------------------------------


class TestLLMResponseMetadata:
    """Tests that LLMResponse contains all required metadata fields."""

    def test_response_has_model_id(self, dry_run_cfg, sample_prompt, call_kwargs):
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.model_id == call_kwargs["model_id"]

    def test_response_has_condition(self, dry_run_cfg, sample_prompt, call_kwargs):
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.condition == call_kwargs["condition"]

    def test_response_has_patient_id(self, dry_run_cfg, sample_prompt, call_kwargs):
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.patient_id == call_kwargs["patient_id"]

    def test_response_has_item_text(self, dry_run_cfg, sample_prompt, call_kwargs):
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.item_text == call_kwargs["item_text"]

    def test_response_has_prompt_hash_16_chars(self, dry_run_cfg, sample_prompt, call_kwargs):
        """prompt_hash is a 16-character hex string."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert len(response.prompt_hash) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", response.prompt_hash)

    def test_response_has_iso8601_timestamp(self, dry_run_cfg, sample_prompt, call_kwargs):
        """timestamp is ISO 8601 format ending with 'Z'."""
        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        assert response.timestamp.endswith("Z")
        # Basic ISO 8601 pattern: YYYY-MM-DDTHH:MM:SS...Z
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", response.timestamp)

    def test_response_prompt_hash_matches_prompt(self, dry_run_cfg, sample_prompt, call_kwargs):
        """prompt_hash is consistent with PromptTemplate.get_prompt_hash."""
        from tracks.reasoning.prompt_template import PromptTemplate

        client = LLMClient(dry_run_cfg)
        response = client.call(sample_prompt, **call_kwargs)
        expected_hash = PromptTemplate.get_prompt_hash(sample_prompt)
        assert response.prompt_hash == expected_hash

    def test_different_prompts_produce_different_hashes(self, dry_run_cfg, call_kwargs):
        """Different prompts produce different prompt_hash values."""
        client = LLMClient(dry_run_cfg)
        r1 = client.call("prompt one", **call_kwargs)
        r2 = client.call("prompt two", **call_kwargs)
        assert r1.prompt_hash != r2.prompt_hash


# ---------------------------------------------------------------------------
# 7.4 Retry loop and FailedCall recording
# ---------------------------------------------------------------------------


class TestRetryAndFailedCalls:
    """Tests for retry logic and FailedCall recording."""

    def test_failed_call_recorded_on_exhaustion(self, hf_cfg, sample_prompt, call_kwargs):
        """After retry_limit failures, a FailedCall is recorded."""
        hf_cfg["retry_limit"] = 2

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.side_effect = RuntimeError("API error")

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)), \
             patch("time.sleep"):  # skip actual sleep
            client = LLMClient(hf_cfg)
            response = client.call(sample_prompt, **call_kwargs)

        failed = client.get_failed_calls()
        assert len(failed) == 1
        fc = failed[0]
        assert isinstance(fc, FailedCall)
        assert fc.model_id == call_kwargs["model_id"]
        assert fc.patient_id == call_kwargs["patient_id"]
        assert fc.condition == call_kwargs["condition"]
        assert fc.item_id == call_kwargs["item_text"]
        assert len(fc.prompt_hash) == 16
        assert "API error" in fc.error_reason

    def test_failed_response_has_empty_raw_text(self, hf_cfg, sample_prompt, call_kwargs):
        """On exhaustion, LLMResponse.raw_text is empty string."""
        hf_cfg["retry_limit"] = 1

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.side_effect = RuntimeError("timeout")

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)), \
             patch("time.sleep"):
            client = LLMClient(hf_cfg)
            response = client.call(sample_prompt, **call_kwargs)

        assert response.raw_text == ""
        assert response.error is not None
        assert "timeout" in response.error

    def test_failed_response_has_error_field(self, hf_cfg, sample_prompt, call_kwargs):
        """On exhaustion, LLMResponse.error contains the error message."""
        hf_cfg["retry_limit"] = 1

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.side_effect = RuntimeError("connection refused")

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)), \
             patch("time.sleep"):
            client = LLMClient(hf_cfg)
            response = client.call(sample_prompt, **call_kwargs)

        assert "connection refused" in response.error

    def test_retry_attempts_match_retry_limit(self, hf_cfg, sample_prompt, call_kwargs):
        """text_generation is called exactly retry_limit times on persistent failure."""
        hf_cfg["retry_limit"] = 3

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.side_effect = RuntimeError("fail")

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)), \
             patch("time.sleep"):
            client = LLMClient(hf_cfg)
            client.call(sample_prompt, **call_kwargs)

        assert mock_client_instance.text_generation.call_count == 3

    def test_no_failed_calls_on_success(self, hf_cfg, sample_prompt, call_kwargs):
        """No FailedCall is recorded when the call succeeds on first attempt."""
        valid_json = json.dumps({key: 0.1 for key in REQUIRED_KEYS})
        mock_result = MagicMock()
        mock_result.generated_text = valid_json
        mock_result.details = None

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.return_value = mock_result

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)):
            client = LLMClient(hf_cfg)
            response = client.call(sample_prompt, **call_kwargs)

        assert client.get_failed_calls() == []
        assert response.error is None
        assert response.raw_text == valid_json

    def test_multiple_failed_calls_all_recorded(self, hf_cfg, sample_prompt):
        """Multiple failed calls are all recorded, none silently dropped."""
        hf_cfg["retry_limit"] = 1

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.side_effect = RuntimeError("fail")

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)), \
             patch("time.sleep"):
            client = LLMClient(hf_cfg)
            client.call(sample_prompt, model_id="m1", condition="context_free",
                        patient_id="P001", item_text="item1")
            client.call(sample_prompt, model_id="m1", condition="correct_context",
                        patient_id="P002", item_text="item2")

        failed = client.get_failed_calls()
        assert len(failed) == 2

    def test_exponential_backoff_sleep_called(self, hf_cfg, sample_prompt, call_kwargs):
        """time.sleep is called with exponential backoff between retries."""
        hf_cfg["retry_limit"] = 3

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.side_effect = RuntimeError("fail")

        sleep_calls = []
        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)), \
             patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            client = LLMClient(hf_cfg)
            client.call(sample_prompt, **call_kwargs)

        # With retry_limit=3: attempt 0 sleeps 2^0=1, attempt 1 sleeps 2^1=2
        # attempt 2 is the last, no sleep after
        assert sleep_calls == [1, 2]


# ---------------------------------------------------------------------------
# 7.1 model_id_slug utility
# ---------------------------------------------------------------------------


class TestModelIdSlug:
    """Tests for the model_id_slug static method."""

    def test_slash_replaced_with_double_underscore(self):
        assert LLMClient.model_id_slug("org/model-name") == "org__model-name"

    def test_multiple_slashes(self):
        assert LLMClient.model_id_slug("a/b/c") == "a__b__c"

    def test_no_slash_unchanged(self):
        assert LLMClient.model_id_slug("local-model") == "local-model"

    def test_meta_llama_example(self):
        assert LLMClient.model_id_slug("meta-llama/Llama-3.1-8B-Instruct") == \
               "meta-llama__Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# 7.2 huggingface backend — successful call
# ---------------------------------------------------------------------------


class TestHuggingFaceBackendSuccess:
    """Tests for successful huggingface backend calls."""

    def test_hf_response_has_correct_backend_field(self, hf_cfg, sample_prompt, call_kwargs):
        """Successful HF call returns backend='huggingface'."""
        mock_result = MagicMock()
        mock_result.generated_text = '{"behavioral_health": 0.3}'
        mock_result.details = None

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.return_value = mock_result

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)):
            client = LLMClient(hf_cfg)
            response = client.call(sample_prompt, **call_kwargs)

        assert response.backend == "huggingface"

    def test_hf_response_raw_text_from_generated_text(self, hf_cfg, sample_prompt, call_kwargs):
        """raw_text is taken from result.generated_text."""
        expected_text = '{"behavioral_health": 0.7}'
        mock_result = MagicMock()
        mock_result.generated_text = expected_text
        mock_result.details = None

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.return_value = mock_result

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)):
            client = LLMClient(hf_cfg)
            response = client.call(sample_prompt, **call_kwargs)

        assert response.raw_text == expected_text

    def test_hf_passes_correct_parameters(self, hf_cfg, sample_prompt, call_kwargs):
        """text_generation is called with the configured parameters."""
        mock_result = MagicMock()
        mock_result.generated_text = "{}"
        mock_result.details = None

        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.return_value = mock_result

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)):
            client = LLMClient(hf_cfg)
            client.call(sample_prompt, **call_kwargs)

        mock_client_instance.text_generation.assert_called_once_with(
            sample_prompt,
            model=call_kwargs["model_id"],
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            details=True,
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# batch_call
# ---------------------------------------------------------------------------


class TestBatchCall:
    """Tests for the batch_call public method."""

    @pytest.fixture
    def dry_run_cfg(self) -> dict:
        return {
            "backend": "dry_run",
            "retry_limit": 3,
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1.0,
            "seed": 42,
        }

    @pytest.fixture
    def sample_items(self) -> list[dict]:
        return [
            {
                "prompt": "Classify this item: hypertension.",
                "model_id": "test-org/test-model",
                "condition": "context_free",
                "patient_id": "P001",
                "item_text": "hypertension",
            },
            {
                "prompt": "Classify this item: diabetes.",
                "model_id": "test-org/test-model",
                "condition": "correct_context",
                "patient_id": "P002",
                "item_text": "diabetes",
            },
        ]

    def test_batch_call_empty_returns_empty(self, dry_run_cfg):
        """batch_call with an empty list returns an empty list."""
        client = LLMClient(dry_run_cfg)
        assert client.batch_call([]) == []

    def test_batch_call_dry_run_returns_one_response_per_item(self, dry_run_cfg, sample_items):
        """batch_call returns exactly one LLMResponse per input item."""
        client = LLMClient(dry_run_cfg)
        responses = client.batch_call(sample_items)
        assert len(responses) == len(sample_items)
        assert all(isinstance(r, LLMResponse) for r in responses)

    def test_batch_call_dry_run_preserves_order(self, dry_run_cfg, sample_items):
        """Responses are returned in the same order as the input items."""
        client = LLMClient(dry_run_cfg)
        responses = client.batch_call(sample_items)
        for item, response in zip(sample_items, responses):
            assert response.patient_id == item["patient_id"]
            assert response.condition == item["condition"]
            assert response.item_text == item["item_text"]

    def test_batch_call_dry_run_no_errors(self, dry_run_cfg, sample_items):
        """dry_run batch responses have error=None."""
        client = LLMClient(dry_run_cfg)
        responses = client.batch_call(sample_items)
        assert all(r.error is None for r in responses)

    def test_batch_call_dry_run_valid_json_raw_text(self, dry_run_cfg, sample_items):
        """dry_run batch responses contain parseable JSON raw_text."""
        client = LLMClient(dry_run_cfg)
        responses = client.batch_call(sample_items)
        for r in responses:
            parsed = json.loads(r.raw_text)
            assert isinstance(parsed, dict)

    def test_batch_call_hf_backend_falls_back_to_individual_calls(self, hf_cfg, sample_items):
        """For the huggingface backend, batch_call issues one call() per item."""
        valid_json = json.dumps({key: 0.1 for key in REQUIRED_KEYS})
        mock_result = MagicMock()
        mock_result.generated_text = valid_json
        mock_result.details = None
        mock_client_instance = MagicMock()
        mock_client_instance.text_generation.return_value = mock_result

        with patch("tracks.reasoning.llm_client.LLMClient._resolve_hf_token", return_value="tok"), \
             patch("builtins.__import__", side_effect=_make_import_mock(mock_client_instance)):
            client = LLMClient(hf_cfg)
            responses = client.batch_call(sample_items)

        assert len(responses) == len(sample_items)
        assert mock_client_instance.text_generation.call_count == len(sample_items)

    def test_batch_call_local_transformers_uses_single_generate(self):
        """local_transformers batch_call triggers exactly one model.generate() call."""
        cfg = {
            "backend": "local_transformers",
            "retry_limit": 1,
            "temperature": 0.0,
            "max_tokens": 16,
            "top_p": 1.0,
            "seed": 42,
        }
        valid_json = json.dumps({key: 0.5 for key in REQUIRED_KEYS})

        mock_tokenizer = MagicMock()
        mock_tokenizer.padding_side = "right"
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        # apply_chat_template returns a formatted string
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        # Batch tokenization returns input_ids + attention_mask tensors
        import torch
        fake_input_ids = torch.zeros((2, 10), dtype=torch.long)
        fake_attn_mask = torch.ones((2, 10), dtype=torch.long)
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: fake_input_ids if k == "input_ids" else fake_attn_mask
        mock_inputs["input_ids"] = fake_input_ids
        mock_inputs["attention_mask"] = fake_attn_mask
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        # generate returns (batch_size, input_len + new_tokens)
        fake_output = torch.zeros((2, 20), dtype=torch.long)
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = fake_output
        mock_tokenizer.decode.return_value = valid_json

        items = [
            {"prompt": "p1", "model_id": "m", "condition": "context_free", "patient_id": "P1", "item_text": "i1"},
            {"prompt": "p2", "model_id": "m", "condition": "context_free", "patient_id": "P2", "item_text": "i2"},
        ]

        client = LLMClient(cfg)
        client._local_model = mock_model
        client._local_tokenizer = mock_tokenizer

        responses = client.batch_call(items)

        mock_model.generate.assert_called_once()
        assert len(responses) == 2


# ---------------------------------------------------------------------------
# Helper: mock import for huggingface_hub
# ---------------------------------------------------------------------------


def _make_import_mock(mock_client_instance: MagicMock):
    """Return a side_effect function that intercepts huggingface_hub imports."""
    import builtins
    real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            mock_module = MagicMock()
            mock_module.InferenceClient.return_value = mock_client_instance
            return mock_module
        return real_import(name, *args, **kwargs)

    return _mock_import
