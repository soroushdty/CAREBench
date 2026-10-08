"""Architecture-boundary tests: canonical package surface verification."""
from __future__ import annotations

import importlib
import importlib.util

import pytest


class TestCanonicalImports:
    """Verify canonical package surface is importable."""

    def test_shared_importable(self):
        mod = importlib.import_module("shared")
        assert mod is not None

    def test_tracks_importable(self):
        mod = importlib.import_module("tracks")
        assert mod is not None

    def test_tracks_representation_importable(self):
        mod = importlib.import_module("tracks.representation")
        assert mod is not None

    def test_tracks_reasoning_importable(self):
        mod = importlib.import_module("tracks.reasoning")
        assert mod is not None

    def test_adapters_importable(self):
        mod = importlib.import_module("adapters")
        assert mod is not None

    @pytest.mark.skipif(
        importlib.util.find_spec("pandas") is None,
        reason="pandas not installed — adapters.paired_context requires pandas at import time",
    )
    def test_adapters_paired_context_importable(self):
        mod = importlib.import_module("adapters.paired_context")
        assert mod is not None
