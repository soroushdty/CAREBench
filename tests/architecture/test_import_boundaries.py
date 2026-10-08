"""Import-boundary tests for the package architecture.

Enforces:
- shared/ does not import from tracks/ or adapters/
- adapters/ does not import from tracks/
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _collect_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return all top-level imported module roots."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module.split(".")[0])
    return modules


def _py_files(directory: Path):
    """Yield all .py files under a directory, skipping __pycache__."""
    if not directory.exists():
        return
    for py_file in directory.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        yield py_file


class TestSharedBoundaries:
    """shared/ must not depend on tracks/ or adapters/."""

    def test_shared_does_not_import_tracks(self):
        violations = []
        for py_file in _py_files(_REPO_ROOT / "shared"):
            imports = _collect_imports(py_file)
            for imp in imports:
                if imp == "tracks":
                    violations.append(
                        f"{py_file.relative_to(_REPO_ROOT)}: imports 'tracks'"
                    )
        assert not violations, "shared/ imports tracks/:\n" + "\n".join(violations)

    def test_shared_does_not_import_adapters(self):
        violations = []
        for py_file in _py_files(_REPO_ROOT / "shared"):
            imports = _collect_imports(py_file)
            for imp in imports:
                if imp == "adapters":
                    violations.append(
                        f"{py_file.relative_to(_REPO_ROOT)}: imports 'adapters'"
                    )
        assert not violations, "shared/ imports adapters/:\n" + "\n".join(violations)


class TestAdapterBoundaries:
    """adapters/ may import shared/ but must not depend on tracks/."""

    def test_adapters_do_not_import_tracks(self):
        violations = []
        for py_file in _py_files(_REPO_ROOT / "adapters"):
            for imp in _collect_imports(py_file):
                if imp == "tracks":
                    violations.append(
                        f"{py_file.relative_to(_REPO_ROOT)}: imports 'tracks'"
                    )
        assert not violations, "adapters/ imports tracks/:\n" + "\n".join(violations)
