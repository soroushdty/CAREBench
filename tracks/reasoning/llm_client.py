"""
LLM Client for the LLM Context-Shift Assay.

Supports three backend modes:
  - huggingface:      Uses huggingface_hub.InferenceClient for real inference
  - local_transformers: Runs the model locally via AutoModelForCausalLM
  - dry_run:          Returns deterministic mock responses without external calls

Implements retry logic with exponential backoff and records all failures
(never silently drops failed calls).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from shared.label_space import DEFAULT_LABEL_SPACE, LabelSpace
from tracks.reasoning.prompt_template import PromptTemplate
from tracks.reasoning.schema_validator import (
    JSONParseError,
    MissingKeyError,
    RangeError,
    validate_response,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Holds the result of a single LLM call, successful or failed.

    Attributes
    ----------
    raw_text : str
        The raw text returned by the LLM, or "" on failure.
    model_id : str
        The model identifier used for this call.
    condition : str
        One of "context_free", "correct_context", or "shuffled_context".
    patient_id : str
        The patient identifier for this call.
    item_text : str
        The EHR item text being classified.
    prompt_hash : str
        16-character SHA-256 hex digest of the prompt.
    timestamp : str
        ISO 8601 timestamp of the call (UTC).
    backend : str
        The backend used: "huggingface" or "dry_run".
    token_usage : dict | None
        Token usage metadata if available, otherwise None.
    error : str | None
        Error message if the call failed, otherwise None.
    """

    raw_text: str
    model_id: str
    condition: str
    patient_id: str
    item_text: str
    prompt_hash: str
    timestamp: str
    backend: str
    token_usage: dict[str, Any] | None
    error: str | None


@dataclass
class FailedCall:
    """Records metadata for a call that exhausted all retry attempts.

    Attributes
    ----------
    model_id : str
        The model identifier.
    patient_id : str
        The patient identifier.
    item_id : str
        The item text (used as item identifier).
    condition : str
        The assay condition.
    prompt_hash : str
        16-character SHA-256 hex digest of the prompt.
    error_reason : str
        The error message from the final failed attempt.
    """

    model_id: str
    patient_id: str
    item_id: str
    condition: str
    prompt_hash: str
    error_reason: str


# ---------------------------------------------------------------------------
# Dry-run mock response
# ---------------------------------------------------------------------------

def _build_dry_run_response(label_space: LabelSpace | None = None) -> str:
    """Return a deterministic JSON string with every category set to 0.5."""
    mock = {d.key: 0.5 for d in (label_space or DEFAULT_LABEL_SPACE)}
    return json.dumps(mock)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Abstraction layer for LLM inference supporting huggingface and dry_run backends.

    Parameters
    ----------
    cfg : dict
        Validated configuration dictionary from :func:`assay.config_loader.load_config`.
        Expected keys used by this class:
          - ``backend``          : "huggingface" | "dry_run"
          - ``retry_limit``      : int, maximum number of attempts per call
          - ``temperature``      : float
          - ``max_tokens``       : int
          - ``top_p``            : float
          - ``seed``             : int
          - ``hf_token``         : str (optional, direct token)
          - ``hf_token_env``     : str (optional, env var name for token)
    label_space : LabelSpace, optional
        Output dimensions responses are validated against (and the dry-run
        mock returns). Defaults to the ten SHARES categories.
    """

    def __init__(
        self, cfg: dict[str, Any], label_space: LabelSpace | None = None
    ) -> None:
        self._cfg = cfg
        self._label_space: LabelSpace = label_space or DEFAULT_LABEL_SPACE
        self._backend: str = cfg["backend"]
        self._retry_limit: int = int(cfg["retry_limit"])
        self._temperature: float = float(cfg.get("temperature", 0.0))
        self._max_tokens: int = int(cfg.get("max_tokens", 512))
        self._top_p: float = float(cfg.get("top_p", 1.0))
        self._seed: int = int(cfg.get("seed", 42))
        self._failed_calls: list[FailedCall] = []

        # local_transformers backend configuration
        self._local_device_map: str = str(cfg.get("local_device_map", "auto"))
        _dtype_str: str = str(cfg.get("local_torch_dtype", "bfloat16"))
        _dtype_map: dict[str, Any] = {}  # populated lazily to avoid importing torch at init
        self._local_torch_dtype_str: str = _dtype_str
        if _dtype_str not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"Unknown local_torch_dtype '{_dtype_str}'. "
                "Must be one of: bfloat16, float16, float32."
            )
        self._local_model = None
        self._local_tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        model_id: str,
        condition: str,
        patient_id: str,
        item_text: str,
    ) -> LLMResponse:
        """Execute a single LLM call with retry logic.

        Parameters
        ----------
        prompt : str
            The full prompt string to send to the LLM.
        model_id : str
            The Hugging Face model ID (or any identifier for dry_run).
        condition : str
            One of "context_free", "correct_context", "shuffled_context".
        patient_id : str
            The patient identifier for this call.
        item_text : str
            The EHR item text being classified.

        Returns
        -------
        LLMResponse
            On success: ``raw_text`` contains the LLM output, ``error`` is None.
            On failure after all retries: ``raw_text`` is "", ``error`` contains
            the error message, and a :class:`FailedCall` is recorded internally.
        """
        prompt_hash = PromptTemplate.get_prompt_hash(prompt)
        timestamp = _utc_now_iso()

        if self._backend == "dry_run":
            return self._call_dry_run(
                prompt_hash=prompt_hash,
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                timestamp=timestamp,
            )
        elif self._backend == "huggingface":
            return self._call_huggingface(
                prompt=prompt,
                prompt_hash=prompt_hash,
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                timestamp=timestamp,
            )
        elif self._backend == "local_transformers":
            return self._call_local(
                prompt=prompt,
                prompt_hash=prompt_hash,
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                timestamp=timestamp,
            )
        else:
            raise ValueError(f"Unknown backend: {self._backend!r}")

    def get_failed_calls(self) -> list[FailedCall]:
        """Return all recorded failed calls (never silently dropped).

        Returns
        -------
        list[FailedCall]
            All calls that exhausted the retry limit.
        """
        return list(self._failed_calls)

    @staticmethod
    def model_id_slug(model_id: str) -> str:
        """Convert a model ID to a filesystem-safe slug.

        Replaces ``/`` with ``__``.

        Parameters
        ----------
        model_id : str
            The Hugging Face model ID, e.g. "meta-llama/Llama-3.1-8B-Instruct".

        Returns
        -------
        str
            Slug suitable for use as a directory name.
        """
        return model_id.replace("/", "__")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_call(
        self,
        items: list[dict[str, str]],
    ) -> list[LLMResponse]:
        """Execute LLM calls for a list of items, using batched GPU inference when possible.

        For the ``local_transformers`` backend all items are tokenized together
        and processed in a single ``model.generate()`` forward pass.  For other
        backends this falls back to individual :meth:`call` invocations.

        Parameters
        ----------
        items : list[dict]
            Each dict must contain the keys ``prompt``, ``model_id``,
            ``condition``, ``patient_id``, and ``item_text``.

        Returns
        -------
        list[LLMResponse]
            One response per input item, in the same order.
        """
        if not items:
            return []

        if self._backend != "local_transformers":
            return [
                self.call(
                    prompt=item["prompt"],
                    model_id=item["model_id"],
                    condition=item["condition"],
                    patient_id=item["patient_id"],
                    item_text=item["item_text"],
                )
                for item in items
            ]

        last_error: str = ""
        for attempt in range(self._retry_limit):
            try:
                return self._call_local_batch(items)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self._retry_limit - 1:
                    time.sleep(2 ** attempt)

        # All retries exhausted — return error responses for every item
        timestamp = _utc_now_iso()
        responses: list[LLMResponse] = []
        for item in items:
            prompt_hash = PromptTemplate.get_prompt_hash(item["prompt"])
            self._failed_calls.append(FailedCall(
                model_id=item["model_id"],
                patient_id=item["patient_id"],
                item_id=item["item_text"],
                condition=item["condition"],
                prompt_hash=prompt_hash,
                error_reason=last_error,
            ))
            responses.append(LLMResponse(
                raw_text="",
                model_id=item["model_id"],
                condition=item["condition"],
                patient_id=item["patient_id"],
                item_text=item["item_text"],
                prompt_hash=prompt_hash,
                timestamp=timestamp,
                backend="local_transformers",
                token_usage=None,
                error=last_error,
            ))
        return responses

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _call_dry_run(
        self,
        prompt_hash: str,
        model_id: str,
        condition: str,
        patient_id: str,
        item_text: str,
        timestamp: str,
    ) -> LLMResponse:
        """Return a deterministic mock response without any external calls."""
        raw_text = _build_dry_run_response(self._label_space)
        return LLMResponse(
            raw_text=raw_text,
            model_id=model_id,
            condition=condition,
            patient_id=patient_id,
            item_text=item_text,
            prompt_hash=prompt_hash,
            timestamp=timestamp,
            backend="dry_run",
            token_usage=None,
            error=None,
        )

    def _call_huggingface(
        self,
        prompt: str,
        prompt_hash: str,
        model_id: str,
        condition: str,
        patient_id: str,
        item_text: str,
        timestamp: str,
    ) -> LLMResponse:
        """Call the Hugging Face inference API with exponential backoff retry.

        The ``huggingface_hub`` import is deferred to avoid import errors when
        running in dry_run mode without the package installed.
        """
        # Lazy import to avoid errors in dry_run mode
        try:
            from huggingface_hub import InferenceClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for the huggingface backend. "
                "Install it with: pip install huggingface-hub"
            ) from exc

        token = self._resolve_hf_token()
        client = InferenceClient(token=token)

        last_error: str = ""
        raw_text: str = ""
        token_usage: dict[str, Any] | None = None
        succeeded: bool = False
        # Some providers (e.g. featherless-ai) only support the conversational
        # task for certain models. We start with text_generation and switch to
        # chat_completion on the first permanent "not supported" error.
        _use_chat: bool = False

        for attempt in range(self._retry_limit):
            try:
                if _use_chat:
                    cc_result = client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        seed=self._seed,
                    )
                    raw_text = cc_result.choices[0].message.content or ""
                    token_usage = None
                else:
                    tg_result = client.text_generation(
                        prompt,
                        model=model_id,
                        max_new_tokens=self._max_tokens,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        seed=self._seed,
                        details=True,
                    )
                    raw_text = tg_result.generated_text if hasattr(tg_result, "generated_text") else str(tg_result)
                    token_usage = None
                    if hasattr(tg_result, "details") and tg_result.details is not None:
                        details = tg_result.details
                        token_usage = {}
                        if hasattr(details, "generated_tokens"):
                            token_usage["generated_tokens"] = details.generated_tokens
                        if hasattr(details, "prefill"):
                            token_usage["prefill_tokens"] = len(details.prefill) if details.prefill else 0

                succeeded = True
                break

            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                # Permanent provider limitation — switch to chat_completion and
                # retry immediately (no sleep, don't count against retry budget).
                if not _use_chat and "not supported for task text-generation" in err_str:
                    _use_chat = True
                    continue
                last_error = err_str
                if attempt < self._retry_limit - 1:
                    wait_seconds = 2 ** attempt
                    time.sleep(wait_seconds)

        if not succeeded:
            # All retries exhausted — record the failure and return an error response
            self._failed_calls.append(FailedCall(
                model_id=model_id,
                patient_id=patient_id,
                item_id=item_text,
                condition=condition,
                prompt_hash=prompt_hash,
                error_reason=last_error,
            ))
            return LLMResponse(
                raw_text="",
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                prompt_hash=prompt_hash,
                timestamp=timestamp,
                backend="huggingface",
                token_usage=None,
                error=last_error,
            )

        # TODO: Investigate disabling MedGemma 1.5 thinking traces via
        # apply_chat_template kwargs (e.g. enable_thinking=False) — would
        # eliminate the need for an inflated max_tokens budget.
        try:
            validate_response(raw_text, self._label_space)
        except (JSONParseError, MissingKeyError, RangeError) as exc:
            err = f"SchemaValidationFailed: {exc}"
            self._failed_calls.append(FailedCall(
                model_id=model_id,
                patient_id=patient_id,
                item_id=item_text,
                condition=condition,
                prompt_hash=prompt_hash,
                error_reason=err,
            ))
            return LLMResponse(
                raw_text=raw_text,
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                prompt_hash=prompt_hash,
                timestamp=timestamp,
                backend="huggingface",
                token_usage=token_usage,
                error=err,
            )

        return LLMResponse(
            raw_text=raw_text,
            model_id=model_id,
            condition=condition,
            patient_id=patient_id,
            item_text=item_text,
            prompt_hash=prompt_hash,
            timestamp=timestamp,
            backend="huggingface",
            token_usage=token_usage,
            error=None,
        )

    def _ensure_local_model_loaded(self, model_id: str) -> None:
        """Lazy-load the model and tokenizer for the local_transformers backend.

        Sets padding_side='left' and ensures pad_token is defined — both
        required for correct batched generation with decoder-only models.
        """
        if self._local_model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import]
        import torch  # type: ignore[import]

        _dtype_lookup: dict[str, Any] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype = _dtype_lookup[self._local_torch_dtype_str]

        self._local_tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=self._resolve_hf_token()
        )
        # Decoder-only models need left-padding so all sequences in a batch
        # end at the same position before generation begins.
        self._local_tokenizer.padding_side = "left"
        if self._local_tokenizer.pad_token is None:
            self._local_tokenizer.pad_token = self._local_tokenizer.eos_token

        self._local_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            device_map=self._local_device_map,
            token=self._resolve_hf_token(),
        )
        self._local_model.eval()

    def _call_local(
        self,
        prompt: str,
        prompt_hash: str,
        model_id: str,
        condition: str,
        patient_id: str,
        item_text: str,
        timestamp: str,
    ) -> LLMResponse:
        """Run local inference via AutoModelForCausalLM with exponential backoff retry.

        Required packages (not top-level imports):
          transformers>=4.50.0
          accelerate
        """
        last_error: str = ""
        raw_text: str = ""
        token_usage: dict[str, Any] | None = None
        succeeded: bool = False

        for attempt in range(self._retry_limit):
            try:
                self._ensure_local_model_loaded(model_id)

                import torch  # type: ignore[import]  # noqa: PLC0415

                messages = [{"role": "user", "content": prompt}]

                inputs = self._local_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self._local_model.device)

                input_len: int = inputs["input_ids"].shape[-1]

                gen_kwargs: dict[str, Any] = dict(
                    max_new_tokens=self._max_tokens,
                    do_sample=self._temperature > 0,
                    pad_token_id=self._local_tokenizer.pad_token_id,
                )
                if self._temperature > 0:
                    gen_kwargs["temperature"] = self._temperature
                    gen_kwargs["top_p"] = self._top_p

                with torch.inference_mode():
                    output_ids = self._local_model.generate(**inputs, **gen_kwargs)

                new_tokens = output_ids[0][input_len:]
                raw_text = self._local_tokenizer.decode(new_tokens, skip_special_tokens=True)

                token_usage = {
                    "prompt_tokens": input_len,
                    "generated_tokens": len(new_tokens),
                }

                succeeded = True
                break

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self._retry_limit - 1:
                    wait_seconds = 2 ** attempt
                    time.sleep(wait_seconds)

        if not succeeded:
            self._failed_calls.append(FailedCall(
                model_id=model_id,
                patient_id=patient_id,
                item_id=item_text,
                condition=condition,
                prompt_hash=prompt_hash,
                error_reason=last_error,
            ))
            return LLMResponse(
                raw_text="",
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                prompt_hash=prompt_hash,
                timestamp=timestamp,
                backend="local_transformers",
                token_usage=None,
                error=last_error,
            )

        # TODO: Investigate disabling MedGemma 1.5 thinking traces via
        # apply_chat_template kwargs (e.g. enable_thinking=False) — would
        # eliminate the need for an inflated max_tokens budget.
        try:
            validate_response(raw_text, self._label_space)
        except (JSONParseError, MissingKeyError, RangeError) as exc:
            err = f"SchemaValidationFailed: {exc}"
            self._failed_calls.append(FailedCall(
                model_id=model_id,
                patient_id=patient_id,
                item_id=item_text,
                condition=condition,
                prompt_hash=prompt_hash,
                error_reason=err,
            ))
            return LLMResponse(
                raw_text=raw_text,
                model_id=model_id,
                condition=condition,
                patient_id=patient_id,
                item_text=item_text,
                prompt_hash=prompt_hash,
                timestamp=timestamp,
                backend="local_transformers",
                token_usage=token_usage,
                error=err,
            )

        return LLMResponse(
            raw_text=raw_text,
            model_id=model_id,
            condition=condition,
            patient_id=patient_id,
            item_text=item_text,
            prompt_hash=prompt_hash,
            timestamp=timestamp,
            backend="local_transformers",
            token_usage=token_usage,
            error=None,
        )

    def _call_local_batch(self, items: list[dict[str, str]]) -> list[LLMResponse]:
        """Run a single batched forward pass for all items.

        All prompts are formatted via the chat template, left-padded to a
        common length, and sent through ``model.generate()`` together.  New
        tokens start at the same column for every row because of left-padding,
        so slicing ``output_ids[:, input_len:]`` extracts the generated text
        for every item simultaneously.
        """
        import torch  # type: ignore[import]

        self._ensure_local_model_loaded(items[0]["model_id"])

        timestamp = _utc_now_iso()

        # Format each prompt with the chat template (produces a plain string)
        formatted: list[str] = []
        for item in items:
            text = self._local_tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            formatted.append(text)

        # Tokenize as a batch with left-padding.
        # add_special_tokens=False: the chat template already embeds them.
        inputs = self._local_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self._local_model.device)

        input_len: int = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self._max_tokens,
            do_sample=self._temperature > 0,
            pad_token_id=self._local_tokenizer.pad_token_id,
        )
        if self._temperature > 0:
            gen_kwargs["temperature"] = self._temperature
            gen_kwargs["top_p"] = self._top_p

        with torch.inference_mode():
            output_ids = self._local_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # All items share the same padded input length (left-padding), so new
        # tokens for every item start at the same offset.
        new_tokens_batch = output_ids[:, input_len:]

        responses: list[LLMResponse] = []
        for i, item in enumerate(items):
            new_tokens = new_tokens_batch[i]
            raw_text = self._local_tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_hash = PromptTemplate.get_prompt_hash(item["prompt"])
            token_usage: dict[str, Any] = {
                "prompt_tokens": int(inputs["attention_mask"][i].sum()),
                "generated_tokens": int(new_tokens.shape[0]),
            }

            try:
                validate_response(raw_text, self._label_space)
                error: str | None = None
            except (JSONParseError, MissingKeyError, RangeError) as exc:
                error = f"SchemaValidationFailed: {exc}"
                self._failed_calls.append(FailedCall(
                    model_id=item["model_id"],
                    patient_id=item["patient_id"],
                    item_id=item["item_text"],
                    condition=item["condition"],
                    prompt_hash=prompt_hash,
                    error_reason=error,
                ))

            responses.append(LLMResponse(
                raw_text=raw_text,
                model_id=item["model_id"],
                condition=item["condition"],
                patient_id=item["patient_id"],
                item_text=item["item_text"],
                prompt_hash=prompt_hash,
                timestamp=timestamp,
                backend="local_transformers",
                token_usage=token_usage,
                error=error,
            ))

        return responses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_hf_token(self) -> str | None:
        """Resolve the Hugging Face API token from config or environment.

        Returns
        -------
        str | None
            The token string, or None if not configured (relies on HF default
            credential resolution).
        """
        import os

        # Direct token in config takes priority
        direct_token: str | None = self._cfg.get("hf_token")
        if direct_token:
            return direct_token

        # Resolve from environment variable
        env_var: str | None = self._cfg.get("hf_token_env")
        if env_var:
            token_from_env = os.environ.get(env_var)
            if token_from_env:
                return token_from_env

        return None
