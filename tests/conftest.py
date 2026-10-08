import matplotlib
matplotlib.use("Agg")

import pytest

from tracks.representation.models.ModelRegistry import get_registry


@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear the ModelRegistry singleton before every test.

    Prevents cross-test contamination: bundles written to the same output
    path by different tests could produce a stale cache hit when file mtimes
    coincide (possible on NTFS whose resolution is 100 ns).
    """
    get_registry().clear()
    yield
    get_registry().clear()
