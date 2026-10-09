"""Guard against references to the private methodology document.

Section-sign markers (U+00A7 followed by a number) pointed to a methodology
document that is not public.  Describe the method in place, or link to
docs/methodology.md, which uses named headings rather than numbered sections.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SECTION_SIGN = chr(0xA7)  # written as a code point so this file passes
_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".ipynb"}
# docs/ is the public documentation; the rest are generated or local-only.
_SKIP_DIRS = {
    "docs", "output", "data", ".git", ".venv", "venv", "build", "dist",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".ipynb_checkpoints",
}


def _source_files():
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        # Virtualenvs are skipped whatever their name (CI creates
        # ``smoke-venv`` in the checkout; installed packages can contain
        # section signs).
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not d.endswith(".egg-info")
            and not Path(dirpath, d, "pyvenv.cfg").exists()
        ]
        for name in filenames:
            path = Path(dirpath, name)
            if path.suffix in _SUFFIXES:
                yield path


def test_no_section_references_outside_docs():
    violations = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _SECTION_SIGN in line:
                rel = path.relative_to(_REPO_ROOT)
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    assert not violations, (
        "Section references to the private methodology document found "
        "(describe the method in place or link to docs/methodology.md):\n"
        + "\n".join(violations)
    )
