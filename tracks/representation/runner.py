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
]


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
    import json
    import logging
    import os
    from pathlib import Path

    import numpy as np
    import yaml

    from shared.prerequisites.load_prerequisites import load_prerequisites
    from shared.prerequisites.reproducibility_guards import (
        enforce_reproducibility,
        fetch_remote_json_with_integrity,
    )
    from shared.embeddings.compute_embeddings import compute_embeddings
    from shared.preprocessing.preprocessing import preprocess
    from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline

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

    # --- Context JSON ---
    context_json = None
    if main_cfg.get("DIR_CONTEXT"):
        _ctx_path = Path(main_cfg["DIR_CONTEXT"])
        if _ctx_path.exists():
            context_json = json.loads(_ctx_path.read_text(encoding="utf-8"))
        else:
            logger.warning(
                "DIR_CONTEXT path not found: %s — Stage 2 fusion disabled.",
                _ctx_path,
            )

    if not cfg.get("DIR_JSON_MAP") and not cfg.get("fuzzy_threshold"):
        logger.info(
            "Neither DIR_JSON_MAP nor fuzzy_threshold is set; items will remain "
            "unresolved after preprocessing because no standardization is applied."
        )

    if cfg.get("DIR_JSON_MAP") and str(cfg["DIR_JSON_MAP"]).startswith("http"):
        fetch_remote_json_with_integrity(cfg["DIR_JSON_MAP"])

    # --- Preprocessing ---
    logger.info("Starting preprocessing...")
    df = preprocess(cfg)

    # --- Embeddings ---
    logger.info("Computing embeddings with model: %s", cfg["llm"])
    embedding_map = compute_embeddings(
        cfg["llm"],
        data=df[cfg["item_col"]],
        schema="pandas",
        batch_size=cfg.get("batch_size", 32),
        cfg=cfg,
    )

    # --- Context vectors (conditional) ---
    context_vectors = None
    if context_json is not None:
        from tracks.representation.training.stage2.stage2_context import build_context_vectors

        context_vectors = build_context_vectors(
            context_json,
            main_cfg["llm"],
            main_cfg,
        )

    # --- Build training arrays ---
    class_cols = [f"{cls}_survey" for cls in cfg["classes"]]
    item_col = cfg["item_col"]
    patient_col = cfg.get("patient_col", "Patient")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train = np.vstack(
        [embedding_map[item.casefold()] for item in train_df[item_col]]
    )
    Y_train = train_df[class_cols].to_numpy(dtype=np.float32)
    patient_ids_train = train_df[patient_col].to_numpy()

    X_test = np.vstack(
        [embedding_map[item.casefold()] for item in test_df[item_col]]
    )
    Y_test = test_df[class_cols].to_numpy(dtype=np.float32)

    _patient_col = cfg.get("patient_col", "Patient")
    _test_df = df[df["split"] == "test"].reset_index(drop=True)

    _interview_cols = [f"{c}_interview" for c in cfg["classes"]]
    _survey_cols = [f"{c}_survey" for c in cfg["classes"]]

    Y_test_interview = (
        _test_df[_interview_cols].to_numpy(dtype=np.float64).astype(np.float32)
    )
    Y_test_survey = (
        _test_df[_survey_cols].to_numpy(dtype=np.float64).astype(np.float32)
    )
    patient_ids_test = _test_df[_patient_col].to_numpy()

    item_strings_train = None
    item_strings_test = None
    unresolved_mask_train = None
    unresolved_mask_test = None
    embedding_cache = None

    if cfg.get("fuzzy_threshold"):
        item_strings_train = train_df[item_col].to_numpy()
        item_strings_test = test_df[item_col].to_numpy()
        unresolved_mask_train = (~train_df["item_json_resolved"]).to_numpy()
        unresolved_mask_test = (~test_df["item_json_resolved"]).to_numpy()
        embedding_cache = embedding_map

    # --- Train ensemble ---
    logger.info("Starting ensemble training pipeline...")
    result = train_ensemble_pipeline(
        X_train,
        Y_train,
        patient_ids_train,
        X_test,
        Y_test,
        cfg,
        item_strings_train=item_strings_train,
        item_strings_test=item_strings_test,
        unresolved_mask_train=unresolved_mask_train,
        unresolved_mask_test=unresolved_mask_test,
        embedding_cache=embedding_cache,
        resume_from_checkpoint=(resume_dir is not None),
        Y_test_interview=Y_test_interview,
        Y_test_survey=Y_test_survey,
        patient_ids_test=patient_ids_test,
        context_vectors=context_vectors,
    )

    logger.info("Pipeline complete.")

    for split in ("train", "test"):
        metrics = result.get(split, {})
        for metric, value in metrics.items():
            logger.info("  %s/%s: %.4f", split, metric, value)

    # --- Statistical analysis ---
    _stat_cfg = cfg.get("statistical_analysis", {}) or {}
    if _stat_cfg.get("enabled", False):
        logger.info("Starting statistical analysis...")
        _stat_hard_error = False
        try:
            from tracks.representation.statistical.orchestrator.run_analysis import run_statistical_analysis

            _stat_output_dir = (
                Path(cfg.get("DIR_MODEL", "output/model")).parent / "statistical_analysis"
            )
            _y_hat_cf = result.get("test_probs_cf_fold_pure")
            if _y_hat_cf is None:
                _y_hat_cf = result.get("test_probs_cf")
            _y_hat_ca = result.get("test_probs_ca_fold_pure")
            _avg_thresh = result.get("avg_thresh_f1opt")

            if _y_hat_ca is None:
                raise RuntimeError(
                    "Statistical analysis requires fold-pure Stage 2 predictions "
                    "(test_probs_ca_fold_pure).\n"
                    "These are required for confirmatory statistics.\n"
                    "Ensure the training pipeline completed successfully and "
                    "fold-pure predictions are available."
                )
            if _y_hat_cf is None:
                logger.warning(
                    "test_probs_cf_fold_pure and test_probs_cf are both None — "
                    "Stage 2 may not have been enabled. "
                    "Using fold-pure Stage 2 predictions as context-free fallback."
                )
                _y_hat_cf = _y_hat_ca
            if _avg_thresh is None:
                _avg_thresh = np.full(len(cfg["classes"]), float(cfg.get("tau", 0.5)))

            _arch_preds = result.get("arch_predictions")
            _stat_hard_error = True

            run_statistical_analysis(
                y_survey=Y_test_survey,
                y_interview=Y_test_interview,
                y_hat_cf=_y_hat_cf,
                y_hat_ca=_y_hat_ca,
                patient_ids=patient_ids_test,
                item_texts=_test_df[item_col].to_numpy(),
                class_list=cfg["classes"],
                avg_thresh_f1opt=_avg_thresh,
                tau_fixed=float(cfg.get("tau", 0.5)),
                dataset_path=Path(cfg["DIR_DATASET"]),
                sheet_names={
                    "train": cfg["TRAIN_SHEET"],
                    "test": cfg["TEST_SHEET"],
                    "interview": cfg["INTERVIEW_SHEET"],
                },
                context_json=context_json,
                llm=cfg["llm"],
                cfg=cfg,
                output_dir=_stat_output_dir,
                n_resamples=int(_stat_cfg.get("n_resamples", 1000)),
                n_permutations=int(_stat_cfg.get("n_permutations", 10000)),
                item_texts_train=item_strings_train,
                ensemble_bundle_path=(
                    Path(cfg.get("DIR_MODEL", "output/model")) / "ensemble_bundle.joblib"
                ),
                arch_predictions=_arch_preds,
                X_items_test=X_test,
            )
            logger.info("Statistical analysis complete. Outputs at: %s", _stat_output_dir)
        except Exception:
            if _stat_hard_error:
                logger.critical(
                    "Statistical analysis failed mid-run — required outputs may be incomplete.",
                    exc_info=True,
                )
                raise
            else:
                logger.exception(
                    "Statistical analysis could not start (setup error). "
                    "Check that the training pipeline produced valid outputs."
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
