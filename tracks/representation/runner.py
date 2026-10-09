#!/usr/bin/env python3
"""tracks.representation.runner — CLI entry point for the Track 1 representation pipeline.

Usage (via dispatcher):
    python main.py --track representation -- --config configs/main_config.yaml

Usage (direct):
    python -m tracks.representation.runner --config configs/main_config.yaml

This module contains NO pipeline logic at module level. All heavy imports
(numpy, pandas, torch, yaml, training code) are deferred to inside main().
"""
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Config discovery helpers (stdlib only — no yaml at module level)
# ---------------------------------------------------------------------------

_CONFIG_SEARCH_DIRS = [
    "configs",
    "configuration",
    "configurations",
    "cfg",
    "cfgs",
]

_CONFIG_SCHEMA_REQUIRED = [
    "DIR_DATASET",
    "TRAIN_SHEET",
    "TEST_SHEET",
    "INTERVIEW_SHEET",
    "DIR_OUTPUT",
    "training_config",
    "summary_config",
]

_CONFIG_SCHEMA_OPTIONAL = [
    "DIR_CONTEXT",
    "DIR_JSON_MAP",
    "DIR_SUMMARY",
    "patient_col",
    "physician_col",
    "item_col",
    "classes",
    "fuzzy_threshold",
    "normalization",
    "physician_count",
    "mismatch_error",
    "same_physicians_survey_interview",
    "llm",
    "llm_revision",
    "hf_local_files_only",
    "tokenizer_model_max_length",
    "pooling",
    "ENABLE_SUMMARY",
    "SUMMARY",
    "DEPENDENCIES",
    "REPRODUCIBILITY",
    "adapter",
    "strategy",
]

# Track 1 components selectable from config: name -> (module_path, class_name).
# Imports are deferred so the runner module stays stdlib-only at import time.
_ADAPTERS: dict[str, tuple[str, str]] = {
    "paired_context": (
        "adapters.paired_context.representation_adapter",
        "PairedContextRepresentationAdapter",
    ),
}

_STRATEGIES: dict[str, tuple[str, str]] = {
    "existing_ensemble": (
        "tracks.representation.strategies.existing_ensemble",
        "ExistingEnsembleTrainingStrategy",
    ),
}

_DEFAULT_ADAPTER = "paired_context"
_DEFAULT_STRATEGY = "existing_ensemble"


# ---------------------------------------------------------------------------
# Argument parsing (lightweight — no heavy deps)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse representation-track CLI arguments."""
    p = argparse.ArgumentParser(
        prog="tracks.representation.runner",
        description="Track 1: Representation learning pipeline (embedding, training, statistical analysis).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dir",
        default=".",
        metavar="PATH",
        help="Project root directory. Defaults to current directory.",
    )
    p.add_argument(
        "--hf_token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace API token. Overrides HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars.",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to main_config.yaml. If omitted, auto-discovery is used.",
    )
    p.add_argument(
        "--resume_from_dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the model output directory of a prior partial training run "
            "(e.g. output/run_20240101_120000_abcd1234/model). "
            "When provided, the pipeline skips completed folds and reuses this directory. "
            "The path must contain a 'checkpoints/' subdirectory."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Internal helpers (stdlib only)
# ---------------------------------------------------------------------------


def _inject_tokens(args: argparse.Namespace) -> None:
    """Propagate HF token to environment variables."""
    import os

    hf = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if hf:
        os.environ["HF_TOKEN"] = hf
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf


def _find_main_config(repo_root: "Path") -> "Path":
    """Discover main_config.yaml using standard search order."""
    from pathlib import Path as _Path

    for d in _CONFIG_SEARCH_DIRS:
        candidate = repo_root / d / "main_config.yaml"
        if candidate.exists():
            return candidate

    checked = {str(repo_root / d) for d in _CONFIG_SEARCH_DIRS}
    for path in repo_root.rglob("main_config.yaml"):
        if str(path.parent) not in checked:
            return path

    raise FileNotFoundError(
        "Error: No config file found. Please provide one via --config "
        "or place main_config.yaml in the repo."
    )


def _load_and_validate_config(config_path: "Path") -> dict:
    """Load and validate the main config YAML."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config YAML must be a mapping/object: {config_path}")

    missing = [k for k in _CONFIG_SCHEMA_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    import logging

    _logger = logging.getLogger(__name__)
    allowed = set(_CONFIG_SCHEMA_REQUIRED) | set(_CONFIG_SCHEMA_OPTIONAL)
    for k in cfg:
        if k not in allowed:
            _logger.warning("Unrecognized config key: %s", k)

    return cfg


def _resolve_config_path(args: argparse.Namespace, repo_root: "Path") -> "Path":
    """Resolve config path from CLI args or auto-discovery."""
    from pathlib import Path as _Path

    if args.config:
        config_path = _Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    return _find_main_config(repo_root)


def _resolve_repo_path(repo_root: "Path", raw_path, *, label: str) -> "Path":
    """Resolve a repo-relative path and verify existence."""
    from pathlib import Path as _Path

    path = _Path(raw_path).expanduser()
    resolved = path if path.is_absolute() else repo_root / path

    if not resolved.exists():
        raise FileNotFoundError(f"{label} file not found: {resolved}")

    return resolved.resolve()


def _load_component(registry: dict[str, tuple[str, str]], name: str, *, kind: str):
    """Import and return the class registered under ``name``."""
    import importlib

    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} '{name}'. Available: {available}")
    module_path, class_name = registry[name]
    return getattr(importlib.import_module(module_path), class_name)


def _build_adapter(cfg: dict):
    """Build the dataset adapter named by ``cfg['adapter']``."""
    name = cfg.get("adapter") or _DEFAULT_ADAPTER
    return _load_component(_ADAPTERS, name, kind="adapter")()


def _build_strategy(cfg: dict, *, resume_from_checkpoint: bool):
    """Build the training strategy named by ``cfg['strategy']``."""
    name = cfg.get("strategy") or _DEFAULT_STRATEGY
    strategy_cls = _load_component(_STRATEGIES, name, kind="strategy")
    return strategy_cls(resume_from_checkpoint=resume_from_checkpoint)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the Track 1 representation pipeline.

    Parameters
    ----------
    argv:
        Argument list (defaults to sys.argv[1:] if None).

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    args = _parse_args(argv)

    # --- Heavy imports (deferred) ---
    import logging
    from pathlib import Path

    import yaml

    from shared.prerequisites.load_prerequisites import load_prerequisites
    from shared.prerequisites.reproducibility_guards import (
        enforce_reproducibility,
        fetch_remote_json_with_integrity,
    )
    from tracks.representation.framework import RepresentationTrack

    logger = logging.getLogger(__name__)

    # --- Token injection ---
    _inject_tokens(args)

    # --- Path resolution ---
    repo_root = Path(__file__).resolve().parents[2]  # tracks/representation/runner.py → repo root
    project_dir = Path(args.dir).resolve()

    # --- Resume directory validation ---
    resume_dir: Path | None = None
    if args.resume_from_dir is not None:
        resume_dir = Path(args.resume_from_dir).resolve()
        if not (resume_dir / "checkpoints").exists():
            print(
                f"Error: --resume_from_dir '{resume_dir}' has no 'checkpoints/' subdirectory. "
                "Point to the model output directory "
                "(e.g. output/<run_id>/model), not the project root.",
                file=sys.stderr,
            )
            return 1

    # --- Config loading ---
    config_path = _resolve_config_path(args, repo_root)
    main_cfg = _load_and_validate_config(config_path)

    # Validate sub-config paths early
    training_config_path = _resolve_repo_path(
        repo_root, main_cfg["training_config"], label="training_config"
    )
    summary_config_path = _resolve_repo_path(
        repo_root, main_cfg["summary_config"], label="summary_config"
    )

    with open(summary_config_path, "r", encoding="utf-8") as fh:
        summary_cfg = yaml.safe_load(fh) or {}

    # --- Prerequisites ---
    cfg = load_prerequisites(
        project_dir=str(project_dir),
        config_path=config_path,
        resume_from_dir=resume_dir,
    )

    enforce_reproducibility(cfg.get("global_seed", 42))

    # Overlay dataset-level keys from main_config.yaml
    _RUNTIME_PATH_KEYS = {"DIR_MODEL", "DIR_SUMMARY", "DIR_REPRODUCIBILITY", "RUN_ID", "DIR_OUTPUT"}
    _main_overlay = {
        k: v
        for k, v in main_cfg.items()
        if k not in ("training_config", "summary_config") and k not in _RUNTIME_PATH_KEYS
    }
    cfg.update(_main_overlay)
    cfg.update(summary_cfg)

    if not cfg.get("DIR_JSON_MAP") and not cfg.get("fuzzy_threshold"):
        logger.info(
            "Neither DIR_JSON_MAP nor fuzzy_threshold is set; items will remain "
            "unresolved after preprocessing because no standardization is applied."
        )

    if cfg.get("DIR_JSON_MAP") and str(cfg["DIR_JSON_MAP"]).startswith("http"):
        fetch_remote_json_with_integrity(cfg["DIR_JSON_MAP"])

    # --- Build adapter and strategy from config ---
    adapter = _build_adapter(cfg)
    strategy = _build_strategy(cfg, resume_from_checkpoint=resume_dir is not None)

    track_cfg = {
        **cfg,
        "output_dir": str(Path(cfg["DIR_MODEL"]).parent),
        "_config_path": str(config_path),
    }

    logger.info("Starting Track 1 via RepresentationTrack...")
    result = RepresentationTrack(adapter=adapter, strategy=strategy).run(track_cfg)

    for split in ("train", "test"):
        metrics = result.get(split, {})
        for metric, value in metrics.items():
            logger.info("  %s/%s: %.4f", split, metric, value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
