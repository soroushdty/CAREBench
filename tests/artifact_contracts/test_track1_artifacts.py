"""Test Track 1 artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTrack1Artifacts:
    """Track 1 representation run must produce required artifacts or map them in artifact_index."""

    def test_artifact_index_or_files_exist(self, minimal_track1_run: Path):
        """At minimum, artifact_index.json must exist and be valid JSON."""
        index_path = minimal_track1_run / "artifact_index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert isinstance(index, dict)

    def test_run_manifest_has_strategy(self, minimal_track1_run: Path):
        """Track 1 run_manifest must declare the strategy used."""
        manifest = json.loads(
            (minimal_track1_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "strategy" in manifest
        assert isinstance(manifest["strategy"], str)
        assert len(manifest["strategy"]) > 0

    def test_adapter_manifest_has_label_space(self, minimal_track1_run: Path):
        """Track 1 adapter_manifest must include label_space with canonical keys."""
        manifest = json.loads(
            (minimal_track1_run / "adapter_manifest.json").read_text(encoding="utf-8")
        )
        assert "label_space" in manifest
        label_space = manifest["label_space"]
        assert isinstance(label_space, dict)
        assert len(label_space) == 10  # default label space has 10 output dimensions
        # All keys should be lowercase snake_case
        for key in label_space:
            assert key == key.lower(), f"Label key '{key}' is not lowercase"
