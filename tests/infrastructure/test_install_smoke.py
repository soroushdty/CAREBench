import pytest


def test_install_and_import():
    """Verify that canonical packages can be imported."""
    try:
        import shared
    except ImportError as e:
        pytest.fail(f"Import of shared failed: {e}")

    assert hasattr(shared, "__file__"), "shared module not found after install"

    try:
        import tracks
    except ImportError as e:
        pytest.fail(f"Import of tracks failed: {e}")

    assert hasattr(tracks, "__file__"), "tracks module not found after install"
