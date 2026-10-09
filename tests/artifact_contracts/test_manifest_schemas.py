"""Test manifest file schemas and required fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestRunManifestSchema:
    """run_manifest.json must have required keys."""

    _REQUIRED_KEYS = [
        "run_id",
        "timestamp_utc",
        "track",
        "entry_command",
        "protocol_version",
    ]

    def test_track1_run_manifest_keys(self, minimal_track1_run: Path):
        manifest = json.loads(
            (minimal_track1_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        for key in self._REQUIRED_KEYS:
            assert key in manifest, f"run_manifest.json missing key: {key}"
        # Track 1 should include strategy
        assert "strategy" in manifest, "Track 1 run_manifest.json should include strategy"

    def test_track3_run_manifest_keys(self, minimal_track3_run: Path):
        manifest = json.loads(
            (minimal_track3_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        for key in self._REQUIRED_KEYS:
            assert key in manifest, f"run_manifest.json missing key: {key}"
        # Track 3 dry-run should include dry_run flag
        assert manifest.get("dry_run") is True


class TestAdapterManifestSchema:
    """adapter_manifest.json must have required fields."""

    _REQUIRED_KEYS = [
        "adapter_name",
        "adapter_class",
        "label_space",
        "reference_aggregation_policy",
    ]

    def test_track1_adapter_manifest_keys(self, minimal_track1_run: Path):
        manifest = json.loads(
            (minimal_track1_run / "adapter_manifest.json").read_text(encoding="utf-8")
        )
        for key in self._REQUIRED_KEYS:
            assert key in manifest, f"adapter_manifest.json missing key: {key}"

    def test_track3_adapter_manifest_keys(self, minimal_track3_run: Path):
        manifest = json.loads(
            (minimal_track3_run / "adapter_manifest.json").read_text(encoding="utf-8")
        )
        for key in self._REQUIRED_KEYS:
            assert key in manifest, f"adapter_manifest.json missing key: {key}"

    def test_label_space_keys_are_canonical(
        self, minimal_track3_run: Path, expected_label_space: dict[str, str]
    ):
        manifest = json.loads(
            (minimal_track3_run / "adapter_manifest.json").read_text(encoding="utf-8")
        )
        label_space = manifest["label_space"]
        # All keys should be snake_case canonical keys
        for key in label_space:
            assert "_" in key or key.islower(), (
                f"Label space key '{key}' is not snake_case canonical"
            )
        # Should match expected label space
        assert set(label_space.keys()) == set(expected_label_space.keys())

    def test_label_space_values_are_display_labels(
        self, minimal_track3_run: Path, expected_label_space: dict[str, str]
    ):
        manifest = json.loads(
            (minimal_track3_run / "adapter_manifest.json").read_text(encoding="utf-8")
        )
        label_space = manifest["label_space"]
        for key, display in label_space.items():
            assert isinstance(display, str)
            assert len(display) > 0


class TestArtifactIndex:
    """artifact_index.json must map roles to existing paths."""

    def test_track3_artifact_index_paths_exist(self, minimal_track3_run: Path):
        index = json.loads(
            (minimal_track3_run / "artifact_index.json").read_text(encoding="utf-8")
        )
        assert len(index) > 0, "artifact_index.json is empty"
        for role, rel_path in index.items():
            full_path = minimal_track3_run / rel_path
            assert full_path.exists(), (
                f"artifact_index maps role '{role}' to '{rel_path}' but file does not exist"
            )
