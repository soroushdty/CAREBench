"""Markdown tables in the Track 3 reports have a consistent number of cells per row."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from shared.evaluation.report_generator import ReportGenerator
from tests.test_report_generation import (
    CATS,
    _make_hypothesis_summary_df,
    _make_minimal_dfs,
)
from tracks.reasoning.bundle.report import generate_markdown_report
from tracks.reasoning.bundle.summary import make_model_comparison_summary

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _malformed_table_rows(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    bad = []
    for i, line in enumerate(lines[:-1]):
        separator = lines[i + 1]
        is_header = (
            line.startswith("|")
            and separator.startswith("|")
            and set(separator.replace("|", "").strip()) <= set("-: ")
        )
        if not is_header:
            continue
        n_cells = len(_UNESCAPED_PIPE.findall(line))
        j = i + 1
        while j < len(lines) and lines[j].startswith("|"):
            if len(_UNESCAPED_PIPE.findall(lines[j])) != n_cells:
                bad.append(lines[j])
            j += 1
    return bad


def test_run_report_tables(tmp_path):
    per_cat = {c: {"mean_abs_delta": 0.1, "ci_lower": 0.0, "ci_upper": 0.2} for c in CATS}
    report = ReportGenerator(
        h1_result={"mean_abs_delta": 0.1, "per_category": per_cat},
        h2_result={"sign_agreement_rate": 0.6},
        h3_result={"pearson_r": 0.4, "mean_delta_physician": np.zeros(3), "mean_delta_llm": np.zeros(3)},
        h4_result={"mean_diff": 0.05},
        model_id="m", run_timestamp="2026-01-01T00:00:00Z", category_names=CATS,
    )
    path = tmp_path / "report.md"
    report.generate(str(path))
    assert _malformed_table_rows(path.read_text()) == []


def test_bundle_report_tables(tmp_path):
    hs = _make_hypothesis_summary_df()
    h1, h2, h3e, h3c, h4 = _make_minimal_dfs()
    val_df = pd.DataFrame([{"check_id": 1, "description": "d", "status": "PASS", "count": 0, "notes": ""}])
    path = tmp_path / "report.md"
    generate_markdown_report(
        hs, h1, h2, h3e, h3c, h4, make_model_comparison_summary(hs), val_df,
        str(path), categories=CATS,
    )
    assert _malformed_table_rows(path.read_text()) == []
