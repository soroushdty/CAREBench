"""Test that common artifacts are present for all track runs."""

from __future__ import annotations

from pathlib import Path

import pytest

_REQUIRED_COMMON = [
    "run_manifest.json",
    "adapter_manifest.json",
    "artifact_index.json",
    "resolved_config.yaml",
]


class TestCommonArtifactsTrack1:
    """Common artifacts for Track 1 runs."""

    def test_all_common_artifacts_present(self, minimal_track1_run: Path):
        for name in _REQUIRED_COMMON:
            path = minimal_track1_run / name
            assert path.exists(), f"Missing required common artifact: {name}"
            assert path.stat().st_size > 0, f"Common artifact is empty: {name}"


class TestCommonArtifactsTrack3:
    """Common artifacts for Track 3 runs."""

    def test_all_common_artifacts_present(self, minimal_track3_run: Path):
        for name in _REQUIRED_COMMON:
            path = minimal_track3_run / name
            assert path.exists(), f"Missing required common artifact: {name}"
            assert path.stat().st_size > 0, f"Common artifact is empty: {name}"
