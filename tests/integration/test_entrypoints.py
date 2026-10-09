"""Smoke tests for CLI entry points.

These tests validate that all documented commands exit cleanly with --help.
They do NOT require data files, model weights, credentials, or heavy libraries.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Repo root (where main.py lives)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a CLI command and return the CompletedProcess."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# Help commands — should all exit 0
# ---------------------------------------------------------------------------


class TestHelpCommands:
    """All --help invocations must exit 0 without heavy imports."""

    def test_main_help(self):
        result = _run([sys.executable, "main.py", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--track" in result.stdout

    def test_main_track_reasoning_help(self):
        result = _run([sys.executable, "main.py", "--track", "reasoning", "--", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_main_track_representation_help(self):
        result = _run([sys.executable, "main.py", "--track", "representation", "--", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_module_reasoning_help(self):
        result = _run([sys.executable, "-m", "tracks.reasoning.run_assay", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_module_representation_help(self):
        result = _run([sys.executable, "-m", "tracks.representation.runner", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"


# ---------------------------------------------------------------------------
# Error cases — should exit nonzero
# ---------------------------------------------------------------------------


class TestErrorCases:
    """Invalid invocations must exit nonzero with clear messages."""

    def test_main_no_track_fails(self):
        result = _run([sys.executable, "main.py"])
        assert result.returncode != 0

    def test_main_invalid_track_fails(self):
        result = _run([sys.executable, "main.py", "--track", "invalid"])
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "invalid" in result.stdout.lower()
