"""
Response Cache for Track 3 (reasoning).

Provides a file-based cache for raw LLM responses, enabling resumability
across interrupted runs. Cache files are stored in a structured directory:

    {cache_dir}/{model_id_slug}/{condition}/{patient_id}__{item_hash}.json

Where ``item_hash = hashlib.sha256(item_text.encode()).hexdigest()[:12]``.

The cache key is ``(model_id, condition, patient_id, item_text)``. The
``prompt_hash`` is stored with each response but is not part of the key;
the Track 3 runner compares it with the current prompt and runs the call again
when they differ, so a changed prompt never reuses an older answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tracks.reasoning.llm_client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_hash(item_text: str) -> str:
    """Return a 12-character SHA-256 hex digest of the item text.

    Parameters
    ----------
    item_text : str
        The EHR item text to hash.

    Returns
    -------
    str
        First 12 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(item_text.encode()).hexdigest()[:12]


def _response_to_dict(response: LLMResponse) -> dict[str, Any]:
    """Serialize an LLMResponse to a JSON-compatible dict.

    Parameters
    ----------
    response : LLMResponse
        The response to serialize.

    Returns
    -------
    dict[str, Any]
        Dictionary with all LLMResponse fields.
    """
    return {
        "raw_text": response.raw_text,
        "model_id": response.model_id,
        "condition": response.condition,
        "patient_id": response.patient_id,
        "item_text": response.item_text,
        "prompt_hash": response.prompt_hash,
        "timestamp": response.timestamp,
        "backend": response.backend,
        "token_usage": response.token_usage,
        "error": response.error,
    }


def _dict_to_response(data: dict[str, Any]) -> LLMResponse:
    """Reconstruct an LLMResponse from a deserialized JSON dict.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary loaded from a cache JSON file.

    Returns
    -------
    LLMResponse
        Reconstructed LLMResponse dataclass instance.
    """
    return LLMResponse(
        raw_text=data["raw_text"],
        model_id=data["model_id"],
        condition=data["condition"],
        patient_id=data["patient_id"],
        item_text=data["item_text"],
        prompt_hash=data["prompt_hash"],
        timestamp=data["timestamp"],
        backend=data["backend"],
        token_usage=data.get("token_usage"),
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


class ResponseCache:
    """File-based cache for raw LLM responses with resumability support.

    Directory structure::

        {cache_dir}/
        └── {model_id_slug}/
            └── {condition}/
                └── {patient_id}__{item_hash}.json

    Parameters
    ----------
    cache_dir : str or Path
        Root directory for the cache. Created on first write if absent.
    expected_calls : list of tuple, optional
        List of ``(model_id, condition, patient_id, item_text)`` tuples
        representing all expected LLM calls. Used by :meth:`get_status` to
        compute the ``pending`` list. If not provided, ``pending`` will be
        empty.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        expected_calls: list[tuple[str, str, str, str]] | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._expected_calls: list[tuple[str, str, str, str]] = (
            list(expected_calls) if expected_calls is not None else []
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        model_id: str,
        condition: str,
        patient_id: str,
        item_text: str,
    ) -> LLMResponse | None:
        """Return a cached LLMResponse, or None if not cached.

        Parameters
        ----------
        model_id : str
            The model identifier.
        condition : str
            One of "context_free", "correct_context", "shuffled_context".
        patient_id : str
            The patient identifier.
        item_text : str
            The EHR item text.

        Returns
        -------
        LLMResponse or None
            The cached response if it exists, otherwise None.
        """
        path = self._cache_path(model_id, condition, patient_id, item_text)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return _dict_to_response(data)
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupted or incomplete cache file — treat as a cache miss
            return None

    def put(self, response: LLMResponse) -> None:
        """Write an LLMResponse to the cache.

        Creates intermediate directories as needed. The write is performed
        atomically via a temporary file to avoid partial writes.

        Parameters
        ----------
        response : LLMResponse
            The response to cache. All fields are written to the JSON file.
        """
        path = self._cache_path(
            response.model_id,
            response.condition,
            response.patient_id,
            response.item_text,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        data = _response_to_dict(response)
        json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

        # Atomic write: write to a temp file in the same directory, then rename
        dir_path = path.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(json_bytes)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_status(self) -> dict[str, list[tuple[str, str, str, str]]]:
        """Return a resumability report for all cached and expected calls.

        Scans the cache directory for existing JSON files and classifies each
        as ``completed`` (``error`` is None) or ``failed`` (``error`` is not
        None). Computes ``pending`` as the set of expected calls minus
        completed and failed.

        Returns
        -------
        dict
            A dictionary with three keys:

            - ``"completed"``: list of ``(model_id, condition, patient_id, item_text)``
              tuples for successful cached responses.
            - ``"failed"``: list of ``(model_id, condition, patient_id, item_text)``
              tuples for cached responses that recorded an error.
            - ``"pending"``: list of ``(model_id, condition, patient_id, item_text)``
              tuples from ``expected_calls`` that are neither completed nor failed.
        """
        completed: list[tuple[str, str, str, str]] = []
        failed: list[tuple[str, str, str, str]] = []

        if self._cache_dir.exists():
            for json_file in self._cache_dir.rglob("*.json"):
                try:
                    with json_file.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    key = (
                        data["model_id"],
                        data["condition"],
                        data["patient_id"],
                        data["item_text"],
                    )
                    if data.get("error") is None:
                        completed.append(key)
                    else:
                        failed.append(key)
                except (json.JSONDecodeError, KeyError, OSError):
                    # Skip unreadable or malformed files
                    continue

        # Compute pending: expected calls not yet completed or failed
        done: set[tuple[str, str, str, str]] = set(completed) | set(failed)
        pending: list[tuple[str, str, str, str]] = [
            call for call in self._expected_calls if call not in done
        ]

        return {
            "completed": completed,
            "failed": failed,
            "pending": pending,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(
        self,
        model_id: str,
        condition: str,
        patient_id: str,
        item_text: str,
    ) -> Path:
        """Compute the cache file path for a given call key.

        Parameters
        ----------
        model_id : str
            The model identifier.
        condition : str
            The condition.
        patient_id : str
            The patient identifier.
        item_text : str
            The EHR item text.

        Returns
        -------
        Path
            Full path to the cache JSON file.
        """
        slug = LLMClient.model_id_slug(model_id)
        ih = _item_hash(item_text)
        filename = f"{patient_id}__{ih}.json"
        return self._cache_dir / slug / condition / filename
