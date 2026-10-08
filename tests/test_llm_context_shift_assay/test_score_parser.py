"""
Tests for tracks/reasoning/score_parser.py — ScoreParser class.

Covers:
  - 9.1 ScoreParser class iterates over cache files for a given model
  - 9.2 parse_all(model_id) calls Schema_Validator and collects validated scores
  - 9.3 Writes context_free_scores.csv, correct_context_scores.csv,
        shuffled_context_scores.csv with correct columns
  - 9.4 Writes errors.csv with correct columns for all failed validations
  - 9.5 Preserves Patient_Item_Pair ordering consistent with PairedDataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.llm_client import LLMClient, LLMResponse
from tracks.reasoning.response_cache import ResponseCache
from tracks.reasoning.schema_validator import REQUIRED_KEYS
from tracks.reasoning.score_parser import ScoreParser, _SCORE_COLUMNS, _ERROR_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SCORES: dict[str, float] = {k: 0.5 for k in REQUIRED_KEYS}
_VALID_RAW = json.dumps(_VALID_SCORES)


def _make_response(
    model_id: str = "org/model",
    condition: str = "context_free",
    patient_id: str = "P001",
    item_text: str = "patient has hypertension",
    raw_text: str = _VALID_RAW,
    error: str | None = None,
) -> LLMResponse:
    """Build a minimal LLMResponse for testing."""
    return LLMResponse(
        raw_text=raw_text if error is None else "",
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


def _write_cache(cache_dir: Path, response: LLMResponse) -> None:
    """Write a response to the cache using ResponseCache."""
    cache = ResponseCache(cache_dir)
    cache.put(response)


def _make_paired_dataset(
    patient_ids: list[str],
    item_texts: list[str],
) -> Any:
    """Build a minimal mock PairedDataset for ordering tests."""
    ds = MagicMock()
    ds.patient_ids = np.array(patient_ids)
    ds.item_texts = np.array(item_texts)
    return ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def scores_dir(tmp_path: Path) -> Path:
    return tmp_path / "scores"


# ---------------------------------------------------------------------------
# 9.1 ScoreParser iterates over cache files
# ---------------------------------------------------------------------------


class TestScoreParserIteration:
    """ScoreParser correctly scans cache files for a given model."""

    def test_parse_all_returns_summary_dict(self, cache_dir, scores_dir):
        """parse_all() returns a dict with scores_written, errors_written, output_dir."""
        _write_cache(cache_dir, _make_response())
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert "scores_written" in result
        assert "errors_written" in result
        assert "output_dir" in result

    def test_parse_all_empty_cache_produces_empty_csvs(self, cache_dir, scores_dir):
        """parse_all() with no cache files produces empty (header-only) CSVs."""
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        out_dir = scores_dir / slug

        for fname in [
            "context_free_scores.csv",
            "correct_context_scores.csv",
            "shuffled_context_scores.csv",
            "errors.csv",
        ]:
            assert (out_dir / fname).exists()

        assert result["scores_written"] == {
            "context_free": 0,
            "correct_context": 0,
            "shuffled_context": 0,
        }
        assert result["errors_written"] == 0

    def test_parse_all_only_scans_correct_model(self, cache_dir, scores_dir):
        """parse_all() only processes files for the specified model_id."""
        _write_cache(cache_dir, _make_response(model_id="org/model-a"))
        _write_cache(cache_dir, _make_response(model_id="org/model-b"))

        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model-a")

        # Only model-a's response should be counted
        assert result["scores_written"]["context_free"] == 1

        # model-b's output dir should not exist
        slug_b = LLMClient.model_id_slug("org/model-b")
        assert not (scores_dir / slug_b).exists()

    def test_parse_all_output_dir_uses_model_slug(self, cache_dir, scores_dir):
        """Output directory uses model_id_slug (/ replaced with __)."""
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("meta-llama/Llama-3.1-8B-Instruct")

        expected_slug = "meta-llama__Llama-3.1-8B-Instruct"
        assert result["output_dir"] == scores_dir / expected_slug


# ---------------------------------------------------------------------------
# 9.2 parse_all calls Schema_Validator and collects validated scores
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """parse_all() correctly validates responses and routes to scores or errors."""

    def test_valid_response_goes_to_scores(self, cache_dir, scores_dir):
        """A valid JSON response is added to the appropriate condition's CSV."""
        _write_cache(cache_dir, _make_response(condition="context_free"))
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["scores_written"]["context_free"] == 1
        assert result["errors_written"] == 0

    def test_invalid_json_goes_to_errors(self, cache_dir, scores_dir):
        """A response with invalid JSON is recorded in errors.csv."""
        _write_cache(
            cache_dir,
            _make_response(raw_text="not valid json at all"),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["scores_written"]["context_free"] == 0
        assert result["errors_written"] == 1

        slug = LLMClient.model_id_slug("org/model")
        errors_df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert errors_df.iloc[0]["error_type"] == "JSONParseError"

    def test_missing_key_goes_to_errors(self, cache_dir, scores_dir):
        """A response missing required keys is recorded in errors.csv."""
        incomplete = {"behavioral_health": 0.5}  # missing 9 keys
        _write_cache(
            cache_dir,
            _make_response(raw_text=json.dumps(incomplete)),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["errors_written"] == 1

        slug = LLMClient.model_id_slug("org/model")
        errors_df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert errors_df.iloc[0]["error_type"] == "MissingKeyError"

    def test_out_of_range_value_goes_to_errors(self, cache_dir, scores_dir):
        """A response with a value outside [0, 1] is recorded in errors.csv."""
        bad_scores = {k: 0.5 for k in REQUIRED_KEYS}
        bad_scores["behavioral_health"] = 1.5  # out of range
        _write_cache(
            cache_dir,
            _make_response(raw_text=json.dumps(bad_scores)),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["errors_written"] == 1

        slug = LLMClient.model_id_slug("org/model")
        errors_df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert errors_df.iloc[0]["error_type"] == "RangeError"

    def test_llm_call_failed_goes_to_errors(self, cache_dir, scores_dir):
        """A response with error field set is recorded as LLMCallFailed."""
        _write_cache(
            cache_dir,
            _make_response(error="API timeout after 3 retries"),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["scores_written"]["context_free"] == 0
        assert result["errors_written"] == 1

        slug = LLMClient.model_id_slug("org/model")
        errors_df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert errors_df.iloc[0]["error_type"] == "LLMCallFailed"
        assert "API timeout" in errors_df.iloc[0]["error_detail"]

    def test_no_silent_drops(self, cache_dir, scores_dir):
        """Every cached response appears in either scores or errors (never dropped)."""
        responses = [
            _make_response(patient_id="P001", item_text="item1"),
            _make_response(patient_id="P002", item_text="item2", raw_text="bad json"),
            _make_response(patient_id="P003", item_text="item3", error="fail"),
        ]
        for r in responses:
            _write_cache(cache_dir, r)

        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        total = result["scores_written"]["context_free"] + result["errors_written"]
        assert total == 3


# ---------------------------------------------------------------------------
# 9.3 CSV columns for score files
# ---------------------------------------------------------------------------


class TestScoreCSVColumns:
    """Score CSV files have the correct columns."""

    def test_context_free_scores_csv_columns(self, cache_dir, scores_dir):
        """context_free_scores.csv has all required columns."""
        _write_cache(cache_dir, _make_response(condition="context_free"))
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")
        assert list(df.columns) == _SCORE_COLUMNS

    def test_correct_context_scores_csv_columns(self, cache_dir, scores_dir):
        """correct_context_scores.csv has all required columns."""
        _write_cache(cache_dir, _make_response(condition="correct_context"))
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "correct_context_scores.csv")
        assert list(df.columns) == _SCORE_COLUMNS

    def test_shuffled_context_scores_csv_columns(self, cache_dir, scores_dir):
        """shuffled_context_scores.csv has all required columns."""
        _write_cache(cache_dir, _make_response(condition="shuffled_context"))
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "shuffled_context_scores.csv")
        assert list(df.columns) == _SCORE_COLUMNS

    def test_score_columns_include_all_ten_categories(self, cache_dir, scores_dir):
        """Score CSV columns include all ten Privacy_Category names."""
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")
        for key in REQUIRED_KEYS:
            assert key in df.columns

    def test_score_columns_include_patient_id_and_item_text(self, cache_dir, scores_dir):
        """Score CSV columns include patient_id and item_text."""
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")
        assert "patient_id" in df.columns
        assert "item_text" in df.columns

    def test_score_columns_include_model_id_and_timestamp(self, cache_dir, scores_dir):
        """Score CSV columns include model_id and timestamp."""
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")
        assert "model_id" in df.columns
        assert "timestamp" in df.columns

    def test_score_values_are_correct(self, cache_dir, scores_dir):
        """Score CSV contains the correct numeric values from the response."""
        scores = {k: round(i * 0.1, 1) for i, k in enumerate(REQUIRED_KEYS)}
        _write_cache(
            cache_dir,
            _make_response(
                condition="context_free",
                patient_id="P001",
                item_text="test item",
                raw_text=json.dumps(scores),
            ),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")
        assert len(df) == 1
        row = df.iloc[0]
        for key, expected_val in scores.items():
            assert abs(row[key] - expected_val) < 1e-9

    def test_model_id_and_timestamp_values_correct(self, cache_dir, scores_dir):
        """model_id and timestamp in CSV match the cached response values."""
        _write_cache(
            cache_dir,
            _make_response(
                model_id="org/model",
                condition="context_free",
                patient_id="P001",
                item_text="test item",
            ),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")
        assert df.iloc[0]["model_id"] == "org/model"
        assert df.iloc[0]["timestamp"] == "2024-01-01T00:00:00.000000Z"

    def test_three_conditions_written_to_separate_files(self, cache_dir, scores_dir):
        """Each condition's scores are written to a separate CSV file."""
        for cond in ("context_free", "correct_context", "shuffled_context"):
            _write_cache(
                cache_dir,
                _make_response(condition=cond, item_text=f"item_{cond}"),
            )

        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["scores_written"]["context_free"] == 1
        assert result["scores_written"]["correct_context"] == 1
        assert result["scores_written"]["shuffled_context"] == 1


# ---------------------------------------------------------------------------
# 9.4 Errors CSV columns
# ---------------------------------------------------------------------------


class TestErrorsCSV:
    """errors.csv has the correct columns and content."""

    def test_errors_csv_columns(self, cache_dir, scores_dir):
        """errors.csv has all required columns."""
        _write_cache(cache_dir, _make_response(error="fail"))
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert list(df.columns) == _ERROR_COLUMNS

    def test_errors_csv_empty_when_no_errors(self, cache_dir, scores_dir):
        """errors.csv has no data rows when all responses are valid."""
        _write_cache(cache_dir, _make_response())
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert len(df) == 0

    def test_errors_csv_records_patient_id_and_item_text(self, cache_dir, scores_dir):
        """errors.csv records patient_id and item_text for failed responses."""
        _write_cache(
            cache_dir,
            _make_response(
                patient_id="P007",
                item_text="specific item text",
                error="timeout",
            ),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert df.iloc[0]["patient_id"] == "P007"
        assert df.iloc[0]["item_text"] == "specific item text"

    def test_errors_csv_records_condition(self, cache_dir, scores_dir):
        """errors.csv records the condition for failed responses."""
        _write_cache(
            cache_dir,
            _make_response(condition="shuffled_context", error="fail"),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert df.iloc[0]["condition"] == "shuffled_context"

    def test_errors_csv_records_model_id(self, cache_dir, scores_dir):
        """errors.csv records the model_id for failed responses."""
        _write_cache(cache_dir, _make_response(error="fail"))
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert df.iloc[0]["model_id"] == "org/model"

    def test_errors_csv_records_error_type_and_detail(self, cache_dir, scores_dir):
        """errors.csv records error_type and error_detail."""
        _write_cache(
            cache_dir,
            _make_response(raw_text="not json"),
        )
        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        assert df.iloc[0]["error_type"] == "JSONParseError"
        assert len(str(df.iloc[0]["error_detail"])) > 0

    def test_errors_csv_multiple_error_types(self, cache_dir, scores_dir):
        """errors.csv records multiple different error types correctly."""
        responses = [
            _make_response(patient_id="P001", item_text="i1", error="LLM fail"),
            _make_response(patient_id="P002", item_text="i2", raw_text="bad json"),
            _make_response(
                patient_id="P003",
                item_text="i3",
                raw_text=json.dumps({"behavioral_health": 0.5}),  # missing keys
            ),
        ]
        for r in responses:
            _write_cache(cache_dir, r)

        parser = ScoreParser(cache_dir, scores_dir)
        result = parser.parse_all("org/model")

        assert result["errors_written"] == 3

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "errors.csv")
        error_types = set(df["error_type"].tolist())
        assert "LLMCallFailed" in error_types
        assert "JSONParseError" in error_types
        assert "MissingKeyError" in error_types


# ---------------------------------------------------------------------------
# 9.5 Patient_Item_Pair ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    """Rows in score CSVs are ordered consistently with PairedDataset."""

    def test_alphabetical_ordering_without_dataset(self, cache_dir, scores_dir):
        """Without a paired dataset, rows are sorted alphabetically by (patient_id, item_text)."""
        responses = [
            _make_response(patient_id="P003", item_text="item_z"),
            _make_response(patient_id="P001", item_text="item_a"),
            _make_response(patient_id="P002", item_text="item_m"),
        ]
        for r in responses:
            _write_cache(cache_dir, r)

        parser = ScoreParser(cache_dir, scores_dir)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")

        assert list(df["patient_id"]) == ["P001", "P002", "P003"]

    def test_dataset_ordering_respected(self, cache_dir, scores_dir):
        """With a paired dataset, rows follow the dataset's (patient_id, item_text) order."""
        # Dataset order: P003, P001, P002
        paired_dataset = _make_paired_dataset(
            patient_ids=["P003", "P001", "P002"],
            item_texts=["item_z", "item_a", "item_m"],
        )

        responses = [
            _make_response(patient_id="P001", item_text="item_a"),
            _make_response(patient_id="P002", item_text="item_m"),
            _make_response(patient_id="P003", item_text="item_z"),
        ]
        for r in responses:
            _write_cache(cache_dir, r)

        parser = ScoreParser(cache_dir, scores_dir, paired_dataset=paired_dataset)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")

        # Should follow dataset order: P003, P001, P002
        assert list(df["patient_id"]) == ["P003", "P001", "P002"]

    def test_unknown_pairs_appended_after_known(self, cache_dir, scores_dir):
        """Rows not in the paired dataset are appended after known rows."""
        paired_dataset = _make_paired_dataset(
            patient_ids=["P001"],
            item_texts=["item_a"],
        )

        responses = [
            _make_response(patient_id="P999", item_text="unknown_item"),
            _make_response(patient_id="P001", item_text="item_a"),
        ]
        for r in responses:
            _write_cache(cache_dir, r)

        parser = ScoreParser(cache_dir, scores_dir, paired_dataset=paired_dataset)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        df = pd.read_csv(scores_dir / slug / "context_free_scores.csv")

        # P001 (known) should come before P999 (unknown)
        assert df.iloc[0]["patient_id"] == "P001"
        assert df.iloc[1]["patient_id"] == "P999"

    def test_ordering_consistent_across_conditions(self, cache_dir, scores_dir):
        """All three condition CSVs use the same ordering."""
        paired_dataset = _make_paired_dataset(
            patient_ids=["P002", "P001"],
            item_texts=["item_b", "item_a"],
        )

        for cond in ("context_free", "correct_context", "shuffled_context"):
            for pid, itxt in [("P001", "item_a"), ("P002", "item_b")]:
                _write_cache(
                    cache_dir,
                    _make_response(condition=cond, patient_id=pid, item_text=itxt),
                )

        parser = ScoreParser(cache_dir, scores_dir, paired_dataset=paired_dataset)
        parser.parse_all("org/model")

        slug = LLMClient.model_id_slug("org/model")
        for fname in [
            "context_free_scores.csv",
            "correct_context_scores.csv",
            "shuffled_context_scores.csv",
        ]:
            df = pd.read_csv(scores_dir / slug / fname)
            assert list(df["patient_id"]) == ["P002", "P001"], f"Wrong order in {fname}"
