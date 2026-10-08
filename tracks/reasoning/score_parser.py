"""
Score Parser for the LLM Context-Shift Assay.

Iterates over all cache files for a given model, calls
``Schema_Validator.validate_response()`` on each cached response, and writes
three score CSV files plus an errors CSV file:

    {scores_dir}/{model_id_slug}/
    ├── context_free_scores.csv
    ├── correct_context_scores.csv
    ├── shuffled_context_scores.csv
    └── errors.csv

Score CSV columns:
    patient_id, item_text, behavioral_health, diagnoses, disabilities,
    infectious_diseases, genetics, medications, sexual_reproductive_health,
    social_determinants_of_health, violence, other, model_id, timestamp

Errors CSV columns:
    patient_id, item_text, condition, model_id, error_type, error_detail
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tracks.reasoning.llm_client import LLMClient
from tracks.reasoning.response_cache import ResponseCache
from tracks.reasoning.schema_validator import (
    REQUIRED_KEYS,
    JSONParseError,
    MissingKeyError,
    RangeError,
    validate_response,
)


# ---------------------------------------------------------------------------
# Condition constants
# ---------------------------------------------------------------------------

_CONDITIONS = ("context_free", "correct_context", "shuffled_context")

_CONDITION_TO_FILENAME: dict[str, str] = {
    "context_free": "context_free_scores.csv",
    "correct_context": "correct_context_scores.csv",
    "shuffled_context": "shuffled_context_scores.csv",
}

# CSV column order for score files
_SCORE_COLUMNS: list[str] = (
    ["patient_id", "item_text"]
    + REQUIRED_KEYS
    + ["model_id", "timestamp"]
)

# CSV column order for errors file
_ERROR_COLUMNS: list[str] = [
    "patient_id",
    "item_text",
    "condition",
    "model_id",
    "error_type",
    "error_detail",
]


# ---------------------------------------------------------------------------
# ScoreParser
# ---------------------------------------------------------------------------


class ScoreParser:
    """Parses cached LLM responses and writes score and error CSV files.

    Parameters
    ----------
    cache_dir : str or Path
        Root directory of the response cache (same as used by
        :class:`assay.response_cache.ResponseCache`).
    scores_dir : str or Path
        Root directory where CSV output files will be written.
        Model-specific subdirectories are created automatically.
    paired_dataset : optional
        A :class:`assay.dataset_loader.PairedDataset` instance used to
        determine the ordering of rows in the output CSVs.  If provided,
        rows are sorted by the order of ``(patient_id, item_text)`` in the
        dataset.  If ``None``, rows are sorted alphabetically by
        ``(patient_id, item_text)``.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        scores_dir: str | Path,
        paired_dataset: Any | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._scores_dir = Path(scores_dir)
        self._paired_dataset = paired_dataset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_all(self, model_id: str) -> dict[str, Any]:
        """Parse all cached responses for *model_id* and write CSV files.

        Steps
        -----
        1. Scan the cache directory for all JSON files under
           ``{cache_dir}/{model_id_slug}/``.
        2. For each cached response:
           - If ``response.error`` is not None, record as ``LLMCallFailed``.
           - Otherwise, call :func:`assay.schema_validator.validate_response`
             on ``response.raw_text``.
           - On success, add the row to the appropriate condition's score list.
           - On validation failure, record in the errors list.
        3. Sort each condition's rows according to the paired dataset ordering
           (or alphabetically if no dataset is provided).
        4. Write the three score CSVs and the errors CSV to
           ``{scores_dir}/{model_id_slug}/``.

        Parameters
        ----------
        model_id : str
            The model identifier (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``).

        Returns
        -------
        dict
            A summary dict with keys:
            - ``"scores_written"``: dict mapping condition name → row count
            - ``"errors_written"``: int, number of error rows
            - ``"output_dir"``: Path to the model-specific output directory
        """
        slug = LLMClient.model_id_slug(model_id)
        model_cache_dir = self._cache_dir / slug

        # Accumulate rows per condition and errors
        scores: dict[str, list[dict[str, Any]]] = {
            cond: [] for cond in _CONDITIONS
        }
        errors: list[dict[str, Any]] = []

        # Scan all JSON files under the model's cache directory
        if model_cache_dir.exists():
            for json_file in sorted(model_cache_dir.rglob("*.json")):
                self._process_cache_file(json_file, scores, errors)

        # Sort rows
        order_map = self._build_order_map()
        for cond in _CONDITIONS:
            scores[cond] = self._sort_rows(scores[cond], order_map)

        # Write output CSVs
        output_dir = self._scores_dir / slug
        output_dir.mkdir(parents=True, exist_ok=True)

        scores_written: dict[str, int] = {}
        for cond in _CONDITIONS:
            filename = _CONDITION_TO_FILENAME[cond]
            out_path = output_dir / filename
            df = pd.DataFrame(scores[cond], columns=_SCORE_COLUMNS)
            df.to_csv(out_path, index=False)
            scores_written[cond] = len(df)

        errors_path = output_dir / "errors.csv"
        errors_df = pd.DataFrame(errors, columns=_ERROR_COLUMNS)
        errors_df.to_csv(errors_path, index=False)

        return {
            "scores_written": scores_written,
            "errors_written": len(errors_df),
            "output_dir": output_dir,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_cache_file(
        self,
        json_file: Path,
        scores: dict[str, list[dict[str, Any]]],
        errors: list[dict[str, Any]],
    ) -> None:
        """Load one cache file and route it to scores or errors.

        Parameters
        ----------
        json_file : Path
            Path to a single cache JSON file.
        scores : dict
            Mutable mapping from condition name to list of score row dicts.
        errors : list
            Mutable list of error row dicts.
        """
        import json

        try:
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Unreadable file — skip silently (cache integrity issue)
            return

        patient_id: str = data.get("patient_id", "")
        item_text: str = data.get("item_text", "")
        condition: str = data.get("condition", "")
        model_id: str = data.get("model_id", "")
        timestamp: str = data.get("timestamp", "")
        raw_text: str = data.get("raw_text", "")
        error_field: str | None = data.get("error")

        # Case 1: LLM call itself failed (error recorded by LLMClient)
        if error_field is not None:
            errors.append(
                {
                    "patient_id": patient_id,
                    "item_text": item_text,
                    "condition": condition,
                    "model_id": model_id,
                    "error_type": "LLMCallFailed",
                    "error_detail": error_field,
                }
            )
            return

        # Case 2: Attempt schema validation
        try:
            validated: dict[str, float] = validate_response(raw_text)
        except (JSONParseError, MissingKeyError, RangeError) as exc:
            errors.append(
                {
                    "patient_id": patient_id,
                    "item_text": item_text,
                    "condition": condition,
                    "model_id": model_id,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                }
            )
            return

        # Case 2b: Normalize keys to canonical form and detect duplicates.
        # validate_response() already returns canonical keys, but this guard
        # protects against future changes or alternative validators.
        from tracks.reasoning.schema_validator import normalize_category_key

        normalized: dict[str, float] = {}
        for key, value in validated.items():
            try:
                canon_key = normalize_category_key(key)
            except KeyError:
                # Unknown key from validator — treat as error
                errors.append(
                    {
                        "patient_id": patient_id,
                        "item_text": item_text,
                        "condition": condition,
                        "model_id": model_id,
                        "error_type": "UnknownDimensionKey",
                        "error_detail": (
                            f"Key '{key}' could not be normalized to a "
                            f"canonical output-dimension key."
                        ),
                    }
                )
                return
            if canon_key in normalized:
                raise ValueError(
                    f"Duplicate canonical key '{canon_key}' after "
                    f"normalization (original keys map to the same "
                    f"canonical key). This indicates a schema or "
                    f"validator bug. patient_id={patient_id!r}, "
                    f"item_text={item_text[:50]!r}, "
                    f"condition={condition!r}"
                )
            normalized[canon_key] = value

        validated = normalized

        # Case 3: Validation succeeded — build score row
        if condition not in scores:
            # Unknown condition — treat as error rather than silently drop
            errors.append(
                {
                    "patient_id": patient_id,
                    "item_text": item_text,
                    "condition": condition,
                    "model_id": model_id,
                    "error_type": "UnknownCondition",
                    "error_detail": f"Unrecognised condition: {condition!r}",
                }
            )
            return

        row: dict[str, Any] = {
            "patient_id": patient_id,
            "item_text": item_text,
            **validated,
            "model_id": model_id,
            "timestamp": timestamp,
        }
        scores[condition].append(row)

    def _build_order_map(self) -> dict[tuple[str, str], int] | None:
        """Build a ``{(patient_id, item_text): position}`` map for ordering.

        Returns ``None`` if no paired dataset is available (alphabetical sort
        will be used instead).
        """
        if self._paired_dataset is None:
            return None

        order_map: dict[tuple[str, str], int] = {}
        patient_ids = self._paired_dataset.patient_ids
        item_texts = self._paired_dataset.item_texts
        for idx, (pid, itxt) in enumerate(zip(patient_ids, item_texts)):
            key = (str(pid), str(itxt))
            if key not in order_map:
                order_map[key] = idx
        return order_map

    def _sort_rows(
        self,
        rows: list[dict[str, Any]],
        order_map: dict[tuple[str, str], int] | None,
    ) -> list[dict[str, Any]]:
        """Sort *rows* according to *order_map* or alphabetically.

        Parameters
        ----------
        rows : list of dict
            Score rows to sort.
        order_map : dict or None
            If provided, rows are sorted by their position in the paired
            dataset.  Unknown ``(patient_id, item_text)`` pairs are placed
            after all known pairs, sorted alphabetically among themselves.
            If ``None``, all rows are sorted alphabetically by
            ``(patient_id, item_text)``.

        Returns
        -------
        list of dict
            Sorted rows.
        """
        if order_map is None:
            return sorted(rows, key=lambda r: (r["patient_id"], r["item_text"]))

        max_pos = len(order_map)

        def _sort_key(row: dict[str, Any]) -> tuple:
            key = (str(row["patient_id"]), str(row["item_text"]))
            pos = order_map.get(key, max_pos)
            # Secondary sort by (patient_id, item_text) for unknown pairs
            return (pos, row["patient_id"], row["item_text"])

        return sorted(rows, key=_sort_key)
