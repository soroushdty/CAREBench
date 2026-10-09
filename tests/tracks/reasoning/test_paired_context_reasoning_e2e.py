"""
End-to-end test for paired-context reasoning dry-run.

Validates that the repaired reasoning pipeline:
1. Exits with code 0 on dry-run
2. Produces score CSVs with canonical-key columns
3. Computes all four endpoints (not skipped due to all-NaN)
4. Includes dry-run/mock metadata in the report

This test uses the synthetic example dataset (examples/synthetic/dataset.xlsx) and the dry_run backend
which returns deterministic mock responses without requiring GPU or API keys.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import pytest

from tracks.reasoning.run_assay import main
from tracks.reasoning.schema_validator import REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dry_run_output(tmp_path, monkeypatch):
    """Run the paired-context reasoning dry-run and return the output directory.

    Uses tmp_path as the output directory so tests are isolated.
    Monkeypatches the working directory to the repo root so that
    relative paths in configs/assay_config.yaml resolve correctly.
    """
    # Ensure we run from the repo root so data/ paths resolve
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(repo_root)

    config_path = repo_root / "configs" / "assay_config.yaml"
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    # Create a temporary config that points output to tmp_path
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["output_dir"] = str(tmp_path / "output")
    cfg["cache_dir"] = str(tmp_path / "cache")
    cfg["scores_dir"] = str(tmp_path / "scores")
    cfg["reports_dir"] = str(tmp_path / "reports")

    tmp_config = tmp_path / "test_config.yaml"
    with open(tmp_config, "w") as f:
        yaml.safe_dump(cfg, f)

    exit_code = main([
        "--config", str(tmp_config),
        "--dry_run",
        "--model", "dry_run_example",
    ])

    return {
        "exit_code": exit_code,
        "output_dir": tmp_path / "output",
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDryRunExecution:
    """Verify the dry-run command completes successfully."""

    def test_exits_zero(self, dry_run_output) -> None:
        """Example dry-run exits with code 0."""
        assert dry_run_output["exit_code"] == 0


class TestScoreCSVs:
    """Verify score CSV artifacts are produced correctly."""

    def _find_score_csvs(self, output_dir: Path) -> dict[str, Path]:
        """Find the three score CSVs in the run folder."""
        # The run folder is under output_dir/run_YYYYMMDD_.../scores/dry_run_example/
        score_files = list(output_dir.rglob("*_scores.csv"))
        result = {}
        for f in score_files:
            if "context_free" in f.name:
                result["context_free"] = f
            elif "correct_context" in f.name:
                result["correct_context"] = f
            elif "shuffled_context" in f.name:
                result["shuffled_context"] = f
        return result

    def test_score_csvs_exist(self, dry_run_output) -> None:
        """All three score CSVs are produced."""
        csvs = self._find_score_csvs(dry_run_output["output_dir"])
        assert "context_free" in csvs, "context_free_scores.csv not found"
        assert "correct_context" in csvs, "correct_context_scores.csv not found"
        assert "shuffled_context" in csvs, "shuffled_context_scores.csv not found"

    def test_score_csvs_use_canonical_columns(self, dry_run_output) -> None:
        """Score CSVs contain canonical-key columns (not display labels)."""
        csvs = self._find_score_csvs(dry_run_output["output_dir"])
        for condition, csv_path in csvs.items():
            df = pd.read_csv(csv_path)
            for key in REQUIRED_KEYS:
                assert key in df.columns, (
                    f"Canonical key '{key}' missing from {condition} CSV columns: "
                    f"{list(df.columns)}"
                )

    def test_score_csvs_have_data(self, dry_run_output) -> None:
        """Score CSVs are not empty (dry_run backend produces mock scores)."""
        csvs = self._find_score_csvs(dry_run_output["output_dir"])
        for condition, csv_path in csvs.items():
            df = pd.read_csv(csv_path)
            assert len(df) > 0, f"{condition} CSV is empty"


class TestHypothesisExecution:
    """Verify the endpoint analysis executes (not skipped)."""

    def _find_report(self, output_dir: Path) -> Path | None:
        """Find the analysis report markdown."""
        reports = list(output_dir.rglob("analysis_report.md"))
        return reports[0] if reports else None

    def test_report_exists(self, dry_run_output) -> None:
        """Analysis report is generated."""
        report = self._find_report(dry_run_output["output_dir"])
        assert report is not None, "analysis_report.md not found in output"

    def test_h1_h4_not_skipped(self, dry_run_output) -> None:
        """All four endpoints are computed (not skipped due to all-NaN score arrays)."""
        report = self._find_report(dry_run_output["output_dir"])
        if report is None:
            pytest.fail("Report not found")

        content = report.read_text()

        # H1 should have a numeric result (the section exists with data)
        assert "### Context sensitivity (formerly H1)" in content
        # The aggregate mean absolute delta line should exist (even if 0.0000)
        assert "Aggregate mean absolute delta:**" in content

        # H2 should have sign agreement rate section
        assert "### Directional alignment (formerly H2)" in content
        assert "Sign agreement rate:**" in content

        # H3 should have Pearson r section (may be NaN for constant mock data,
        # which is a valid statistical result — not a pipeline skip)
        assert "### Class-level correspondence (formerly H3)" in content
        assert "Pearson r:**" in content

        # H4 should either have results or explicit skip reason (not missing)
        assert "### Context specificity (formerly H4)" in content
        h4_section_start = content.index("### Context specificity")
        h4_section = content[h4_section_start:h4_section_start + 500]
        has_result = "Mean alignment difference" in h4_section
        has_skip = "Skipped" in h4_section
        assert has_result or has_skip, (
            "H4 section has neither results nor explicit skip status"
        )

        # Critical check: the pipeline did NOT skip due to all-NaN arrays.
        # This would manifest as the absence of hypothesis sections entirely
        # or the log message "Skipping endpoint analysis".
        assert "Skipping endpoint analysis" not in content


class TestDryRunMetadata:
    """Verify dry-run mock metadata is present."""

    def _find_report(self, output_dir: Path) -> Path | None:
        reports = list(output_dir.rglob("analysis_report.md"))
        return reports[0] if reports else None

    def test_report_has_dry_run_banner(self, dry_run_output) -> None:
        """Report contains DRY RUN warning metadata."""
        report = self._find_report(dry_run_output["output_dir"])
        if report is None:
            pytest.fail("Report not found")

        content = report.read_text()
        assert "DRY RUN" in content, (
            "Report does not contain 'DRY RUN' mock warning banner"
        )
        assert "not scientifically interpretable" in content.lower() or \
               "Not scientifically interpretable" in content, (
            "Report does not contain interpretability warning"
        )
