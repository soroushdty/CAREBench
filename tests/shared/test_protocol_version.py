"""The protocol version is recorded with every run and report."""

from __future__ import annotations

import json

from shared.endpoints import PROTOCOL_VERSION
from shared.evaluation.report_generator import ReportGenerator
from shared.prerequisites.manifests import write_run_manifest


def test_protocol_version_is_a_string():
    assert isinstance(PROTOCOL_VERSION, str) and PROTOCOL_VERSION


def test_run_manifest_records_protocol_version(tmp_path):
    path = write_run_manifest(tmp_path, run_id="r", track="reasoning", entry_command="x")
    assert json.loads(path.read_text())["protocol_version"] == PROTOCOL_VERSION


def test_track3_report_header_shows_protocol_version(tmp_path):
    out = tmp_path / "report.md"
    ReportGenerator({}, {}, {}, {"status": "skipped"}, "m", "t").generate(str(out))
    assert f"**Protocol version:** {PROTOCOL_VERSION}" in out.read_text()
