#!/usr/bin/env python
"""Canonical import smoke test.

Verifies that the canonical package surface (shared, tracks, adapters) is
importable after an editable install.  Prints pass/fail per import and exits
with code 0 on full success, non-zero otherwise.
"""

import importlib
import sys

CANONICAL_IMPORTS = [
    "shared",
    "tracks",
    "tracks.representation",
    "tracks.reasoning",
    "adapters",
    "adapters.paired_context",
]


def main() -> int:
    failures: list[str] = []
    for module_name in CANONICAL_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"  PASS  {module_name}")
        except Exception as exc:
            print(f"  FAIL  {module_name} — {exc}")
            failures.append(module_name)

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} canonical import(s) could not be loaded.")
        for name in failures:
            print(f"  - {name}")
        return 1

    print("SMOKE TEST PASSED: all canonical imports succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
