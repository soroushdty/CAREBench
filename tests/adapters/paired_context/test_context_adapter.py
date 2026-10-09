"""Tests for adapters/paired_context/context_adapter.py — PairedContextContextAdapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.paired_context.context_adapter import CONTEXT_FIELDS, PairedContextContextAdapter


class TestLoadContextRecords:
    """Verify context record loading."""

    def test_load_success(self, summaries_file: Path):
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        records = adapter.load_context_records()
        assert len(records) == 3
        assert "1" in records
        assert "2" in records
        assert "3" in records

    def test_keys_normalized_to_strings(self, tmp_path: Path):
        data = {1: {"summary": "test"}, 2: {"summary": "test2"}}
        path = tmp_path / "int_keys.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        adapter = PairedContextContextAdapter(summaries_path=path)
        records = adapter.load_context_records()
        assert "1" in records
        assert "2" in records

    def test_file_not_found(self, tmp_path: Path):
        adapter = PairedContextContextAdapter(summaries_path=tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            adapter.load_context_records()

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text("{}", encoding="utf-8")
        adapter = PairedContextContextAdapter(summaries_path=path)
        with pytest.raises(ValueError, match="empty"):
            adapter.load_context_records()

    def test_caching(self, summaries_file: Path):
        """Second call returns same object (cached)."""
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        r1 = adapter.load_context_records()
        r2 = adapter.load_context_records()
        assert r1 is r2


class TestGetRecord:
    """Verify single-record access."""

    def test_get_existing_record(self, summaries_file: Path):
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        record = adapter.get_record("1")
        assert "summary" in record

    def test_get_missing_record(self, summaries_file: Path):
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        with pytest.raises(KeyError, match="999"):
            adapter.get_record("999")


class TestGetEntityIds:
    """Verify entity ID listing."""

    def test_returns_all_ids(self, summaries_file: Path):
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        ids = adapter.get_entity_ids()
        assert set(ids) == {"1", "2", "3"}


class TestFormatContext:
    """Verify context formatting."""

    def test_format_includes_all_fields(self):
        fields = {f: f"value_{f}" for f in CONTEXT_FIELDS}
        text = PairedContextContextAdapter.format_context(fields)
        for field in CONTEXT_FIELDS:
            assert f"[{field.upper()}]" in text

    def test_format_handles_lists(self):
        fields = {"summary": "test", "medical_history": ["item1", "item2"]}
        text = PairedContextContextAdapter.format_context(fields)
        assert "- item1" in text
        assert "- item2" in text

    def test_format_handles_empty(self):
        fields = {}
        text = PairedContextContextAdapter.format_context(fields)
        # Should still produce headers for all fields
        for field in CONTEXT_FIELDS:
            assert f"[{field.upper()}]" in text


class TestManifest:
    """Verify manifest output."""

    def test_manifest_fields(self, summaries_file: Path):
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        m = adapter.manifest()
        assert m["adapter"] == "paired_context_context"
        assert "context_source_path" in m
        assert "parsing_policy" in m
        assert "context_fields" in m

    def test_name(self, summaries_file: Path):
        adapter = PairedContextContextAdapter(summaries_path=summaries_file)
        assert adapter.name == "paired_context_context"
