"""Test Track 3 artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


class TestTrack3ScoreCSVs:
    """Track 3 must produce three score CSVs with canonical columns."""

    _CONDITIONS = ("context_free", "correct_context", "shuffled_context")

    def test_all_score_csvs_exist(self, minimal_track3_run: Path):
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        for condition in self._CONDITIONS:
            role = f"track3.scores.{condition}"
            assert role in index, f"artifact_index missing role: {role}"
            csv_path = minimal_track3_run / index[role]
            assert csv_path.exists(), f"Score CSV not found: {csv_path}"
            assert csv_path.stat().st_size > 0, f"Score CSV is empty: {csv_path}"

    def test_score_csv_has_patient_id_column(self, minimal_track3_run: Path):
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        for condition in self._CONDITIONS:
            role = f"track3.scores.{condition}"
            csv_path = minimal_track3_run / index[role]
            df = pd.read_csv(csv_path)
            assert "patient_id" in df.columns, (
                f"{csv_path.name} missing 'patient_id' column"
            )

    def test_score_csv_has_item_text_column(self, minimal_track3_run: Path):
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        for condition in self._CONDITIONS:
            role = f"track3.scores.{condition}"
            csv_path = minimal_track3_run / index[role]
            df = pd.read_csv(csv_path)
            assert "item_text" in df.columns, (
                f"{csv_path.name} missing 'item_text' column"
            )

    def test_score_csv_uses_canonical_dimension_keys(
        self,
        minimal_track3_run: Path,
        expected_label_space: dict[str, str],
    ):
        """Score CSV columns for output dimensions must use canonical keys."""
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        canonical_keys = set(expected_label_space.keys())

        for condition in self._CONDITIONS:
            role = f"track3.scores.{condition}"
            csv_path = minimal_track3_run / index[role]
            df = pd.read_csv(csv_path)
            # Score columns = all columns except patient_id and item_text
            score_cols = set(df.columns) - {"patient_id", "item_text"}
            assert score_cols.issubset(canonical_keys), (
                f"{csv_path.name} has non-canonical columns: "
                f"{score_cols - canonical_keys}"
            )

    def test_score_csv_has_no_all_nan_columns(self, minimal_track3_run: Path):
        """No output dimension column should be entirely NaN."""
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        for condition in self._CONDITIONS:
            role = f"track3.scores.{condition}"
            csv_path = minimal_track3_run / index[role]
            df = pd.read_csv(csv_path)
            score_cols = [c for c in df.columns if c not in ("patient_id", "item_text")]
            for col in score_cols:
                assert not df[col].isna().all(), (
                    f"{csv_path.name}: column '{col}' is entirely NaN"
                )


class TestTrack3Report:
    """Track 3 must produce a hypothesis report."""

    def test_report_exists(self, minimal_track3_run: Path):
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        assert "track3.hypothesis_summary" in index
        report_path = minimal_track3_run / index["track3.hypothesis_summary"]
        assert report_path.exists()
        assert report_path.stat().st_size > 0

    def test_report_contains_display_labels(
        self,
        minimal_track3_run: Path,
        expected_label_space: dict[str, str],
    ):
        """Report should use at least one display label for readability."""
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        report_path = minimal_track3_run / index["track3.hypothesis_summary"]
        content = report_path.read_text(encoding="utf-8")
        # At least one display label should appear in the report
        display_labels = list(expected_label_space.values())
        found = any(label in content for label in display_labels)
        assert found, (
            "Report does not contain any display labels from label_space. "
            "Reports should use human-readable display labels."
        )


class TestTrack3DryRunMetadata:
    """Track 3 dry-run must include warning metadata."""

    def test_dry_run_flag_in_manifest(self, minimal_track3_run: Path):
        manifest = json.loads(
            (minimal_track3_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest.get("dry_run") is True

    def test_dry_run_warning_in_manifest(self, minimal_track3_run: Path):
        manifest = json.loads(
            (minimal_track3_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "dry_run_warning" in manifest
        assert "not scientific" in manifest["dry_run_warning"].lower()
