#!/usr/bin/env python3
"""main.py — Thin CLI dispatcher for the clinical context-shift evaluation framework.

Selects a track (representation or reasoning) and delegates to the
corresponding runner. No pipeline logic lives here.

Usage:
    python main.py --track {representation,reasoning} [-- <runner_args>...]

Examples:
    python main.py --track representation -- --config configs/main_config.yaml
    python main.py --track reasoning -- --config configs/assay_config.yaml --dry_run
    python main.py --help
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback

# Maps track names → (module_path, function_name)
_TRACK_DISPATCH: dict[str, tuple[str, str]] = {
    "representation": ("tracks.representation.runner", "main"),
    "reasoning": ("tracks.reasoning.run_assay", "main"),
}


def _split_argv(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Split argv on '--' into dispatcher args and runner args."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, None


def main() -> int:
    """Parse --track, lazy-import the runner, forward remaining args, return exit code."""
    raw_argv = sys.argv[1:]
    dispatcher_argv, runner_argv = _split_argv(raw_argv)

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--track",
        required=True,
        choices=sorted(_TRACK_DISPATCH.keys()),
        help="Track to run: %(choices)s",
    )

    args = parser.parse_args(dispatcher_argv)
    track: str = args.track

    # --- Lazy import of the selected runner ---
    module_path, func_name = _TRACK_DISPATCH[track]
    try:
        module = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            f"Error: Failed to load track '{track}' from '{module_path}': {exc}",
            file=sys.stderr,
        )
        return 1

    runner_main = getattr(module, func_name, None)
    if runner_main is None:
        print(
            f"Error: Module '{module_path}' has no callable '{func_name}'.",
            file=sys.stderr,
        )
        return 1

    # --- Dispatch ---
    try:
        result = runner_main(runner_argv)
    except SystemExit as exc:
        # Let argparse --help (exit 0) and argparse errors propagate naturally.
        return exc.code if isinstance(exc.code, int) else 0
    except Exception:
        traceback.print_exc()
        return 1

    # Propagate integer exit code; treat None as success.
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
