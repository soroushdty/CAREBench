"""Tests for tracks/representation/statistical/reporting/figures.py."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from tracks.representation.statistical.reporting.figures import figure1_delta_histogram


@pytest.mark.parametrize("n_classes", [1, 3, 10, 12])
def test_figure1_handles_any_number_of_classes(tmp_path, n_classes):
    rng = np.random.default_rng(0)
    delta_p = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(20, n_classes))
    out = tmp_path / "figure1.png"

    figure1_delta_histogram(delta_p, [f"Class {i}" for i in range(n_classes)], out)

    assert out.stat().st_size > 0
