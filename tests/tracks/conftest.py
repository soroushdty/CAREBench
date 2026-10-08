"""Local conftest for tests/tracks/.

Overrides the root conftest's clear_model_registry autouse fixture so that
smoke tests in this directory can run without torch being installed.
"""
import pytest


@pytest.fixture(autouse=True)
def clear_model_registry():
    """No-op override: smoke tests don't use the ModelRegistry."""
    yield
