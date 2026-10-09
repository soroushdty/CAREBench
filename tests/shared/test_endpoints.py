"""Named endpoints are defined once and used by every track (#15)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from shared.endpoints import ENDPOINTS, get_endpoint
from tracks.reasoning.endpoints import TRACK_ENDPOINTS as TRACK3
from tracks.representation.statistical.endpoints import TRACK_ENDPOINTS as TRACK1
from tracks.representation.statistical.orchestrator.run_analysis import _OUTPUTS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_NAME = re.compile(r"^[hH][1-4]_")


def test_registry_keys_match_names():
    assert all(name == endpoint.name for name, endpoint in ENDPOINTS.items())


def test_get_endpoint_unknown_name():
    with pytest.raises(KeyError, match="Known endpoints"):
        get_endpoint("h2")


@pytest.mark.parametrize("track", [TRACK1, TRACK3], ids=["track1", "track3"])
def test_track_declarations_use_registry_endpoints(track):
    names = [te.name for te in track]
    assert len(names) == len(set(names))
    for te in track:
        assert ENDPOINTS[te.name] is te.endpoint


def test_tracks_share_directional_alignment():
    """The question that was Track 1's H1 and Track 3's H2 now has one name."""
    t1 = {te.name: te.legacy_alias for te in TRACK1}
    t3 = {te.name: te.legacy_alias for te in TRACK3}
    assert t1["directional_alignment"] == "H1"
    assert t3["directional_alignment"] == "H2"


def test_heading_shows_legacy_alias():
    t3 = {te.name: te for te in TRACK3}
    assert t3["context_specificity"].heading == "Context specificity (formerly H4)"


def test_track1_outputs_are_named_by_endpoint():
    for te in TRACK1:
        assert any(path.startswith(f"{te.name}_") for path in _OUTPUTS), te.name
    assert not [path for path in _OUTPUTS if _LEGACY_NAME.match(path)]


def test_bundle_outputs_are_named_by_endpoint():
    script = (_REPO_ROOT / "scripts" / "analyze_llm_context_effects.py").read_text()
    csv_names = set(re.findall(r'"([A-Za-z0-9_]+\.csv)"', script))
    for te in TRACK3:
        assert f"{te.name}.csv" in csv_names, te.name
    assert not [name for name in csv_names if _LEGACY_NAME.match(name)]


def test_bundle_summary_columns_are_named_by_endpoint():
    import pandas as pd

    from tracks.reasoning.bundle.summary import make_hypothesis_summary

    def frame(**cols):
        return pd.DataFrame([{"model": "m", "category": "aggregate", **cols}])

    cells = pd.DataFrame([{
        "model": "m", "patient_id": "1", "item_text": "x", "category": "a",
        "delta_physician": 0.5,
    }])
    summary = make_hypothesis_summary(
        frame(mean_abs_delta_correct=0.2),
        frame(mean_alignment_correct=0.1),
        pd.DataFrame([{"model": "m", "pearson_correct": 0.3}]),
        frame(mean_alignment_difference=0.1, ci_low_difference=0.01),
        cells,
        pd.DataFrame(),
    )
    endpoint_cols = [c for c in summary.columns if c.split("_")[0] not in {"model", "n", "failure", "interpretation"}]
    t3_names = tuple(f"{te.name}_" for te in TRACK3)
    assert endpoint_cols and all(c.startswith(t3_names) for c in endpoint_cols)
    assert summary.iloc[0]["context_specificity_ci_low"] == pytest.approx(0.01)
