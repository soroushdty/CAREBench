"""
Assay Runner CLI for the LLM Context-Shift Assay.

Entry point:

    python main.py --track reasoning -- [--config PATH] [--model MODEL_ID] [--dry_run]

Orchestration order:
1. Create timestamped run folder and configure logging
2. Load and validate config
3. Load dataset → PairedDataset
4. Build shuffled context map (deterministic, seeded)
5. For each model ID:
   a. Check cache status (resumability)
   b. Log resumability report
   c. For each (patient_id, item_text) pair × 3 conditions: run LLM (or load from cache)
   d. Parse scores → CSVs
   e. Compute the endpoints (context sensitivity, directional alignment,
      class-level correspondence, context specificity)
   f. Generate report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

import yaml

from tracks.reasoning.config_loader import (
    REQUIRED_DATA_KEYS,
    REQUIRED_TOP_LEVEL_KEYS,
    VALID_BACKENDS,
    ConfigError,
    _validate_model_ids,
    load_config,
    validate_label_space_config,
    validate_prompt_config,
)
from tracks.reasoning.context_builder import ContextBuilder
from tracks.reasoning.dataset_loader import DatasetLoader
from tracks.reasoning.endpoints import TRACK_ENDPOINTS
from shared.evaluation.hypothesis_analyzer import HypothesisAnalyzer
from tracks.reasoning.llm_client import LLMClient
from tracks.reasoning.prompt_template import PromptTemplate
from shared.evaluation.report_generator import ReportGenerator
from tracks.reasoning.response_cache import ResponseCache
from tracks.reasoning.score_parser import ScoreParser
from shared.prerequisites.logging_config import logging_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONDITIONS = ("context_free", "correct_context", "shuffled_context")

_DEFAULT_CONFIG_PATH = "configs/assay_config.yaml"

_DEFAULT_OUTPUT_DIR = "output/assay"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list (defaults to sys.argv[1:] if None).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - ``config``: path to assay_config.yaml
        - ``model``: optional model ID override
        - ``dry_run``: bool flag
    """
    parser = argparse.ArgumentParser(
        description="Run the LLM Context-Shift Assay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --track reasoning -- --dry_run\n"
            "  python main.py --track reasoning -- --config configs/assay_config.yaml --dry_run\n"
            "  python main.py --track reasoning -- --model meta-llama/Llama-3.1-8B-Instruct\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=(
            f"Path to assay_config.yaml (default: {_DEFAULT_CONFIG_PATH!r})"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Optional model ID to run. Overrides model_ids in config and "
            "runs only this model."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help=(
            "Override backend to 'dry_run'. No Hugging Face credentials needed. "
            "Returns deterministic mock responses."
        ),
    )
    parser.add_argument(
        "--clear-failed",
        action="store_true",
        default=False,
        dest="clear_failed",
        help=(
            "Delete cache entries with non-null error before running. "
            "Useful after schema validator or generation-config changes."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Run folder + logging bootstrap
# ---------------------------------------------------------------------------


def _create_run_folder(output_dir: str | Path) -> Path:
    """Create a unique timestamped run folder under *output_dir*.

    Folder name pattern: ``run_YYYYMMDD_HHMMSS_ffffff_XXXXXXXX``
    """
    now = datetime.now(timezone.utc)
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
    run_folder = Path(output_dir) / run_id
    run_folder.mkdir(parents=True, exist_ok=False)
    return run_folder


def _setup_logging(run_folder: Path, log_level: str = "INFO") -> None:
    """Configure root logger: file in *run_folder* + console handler."""
    log_path = run_folder / "log.txt"
    logging_config(log_path, log_level)

    # Also stream to stdout so progress is visible in the terminal.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.getLogger().level)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)s:  %(message)s",
            datefmt="%H:%M:%S  %m-%d-%y",
        )
    )
    logging.getLogger().addHandler(console)


# ---------------------------------------------------------------------------
# Shuffled context map builder
# ---------------------------------------------------------------------------


def _build_shuffled_context_map(
    context_builder: ContextBuilder,
    patient_ids: np.ndarray,
    item_texts: np.ndarray,
    seed: int,
) -> dict[tuple[str, str], str]:
    """Build a deterministic shuffled context map for all patient-item pairs.

    Parameters
    ----------
    context_builder:
        Initialised :class:`~assay.context_builder.ContextBuilder`.
    patient_ids:
        1-D array of patient IDs (one per row of the paired dataset).
    item_texts:
        1-D array of item texts parallel to ``patient_ids``.
    seed:
        Integer seed for the random generator (from ``cfg["shuffled_context_seed"]``).

    Returns
    -------
    dict
        ``{(patient_id_str, item_text): shuffled_context_text}`` for every
        unique ``(patient_id, item_text)`` pair.
    """
    rng = np.random.default_rng(seed)
    shuffled_map: dict[tuple[str, str], str] = {}

    seen: set[tuple[str, str]] = set()
    for pid, itxt in zip(patient_ids, item_texts):
        key = (str(pid), str(itxt))
        if key in seen:
            continue
        seen.add(key)
        shuffled_text = context_builder.build_shuffled_context(
            patient_id=pid,
            item_text=itxt,
            rng=rng,
        )
        shuffled_map[key] = shuffled_text

    return shuffled_map


# ---------------------------------------------------------------------------
# Score loading helpers
# ---------------------------------------------------------------------------


def _load_score_array(
    csv_path: Path,
    patient_ids: np.ndarray,
    item_texts: np.ndarray,
    category_names: list[str],
    label_space: Any | None = None,
) -> np.ndarray:
    """Load a score CSV and return an ``(N, D)`` numpy array aligned to the dataset.

    Rows in the CSV are matched to the dataset ordering by ``(patient_id,
    item_text)``.  Missing rows are filled with ``NaN``.

    Parameters
    ----------
    csv_path:
        Path to the score CSV file.
    patient_ids:
        1-D array of patient IDs from the paired dataset (shape ``(N,)``).
    item_texts:
        1-D array of item texts from the paired dataset (shape ``(N,)``).
    category_names:
        List of D canonical output-dimension keys (e.g.
        ``["behavioral_health", "diagnoses", ...]``).
    label_space:
        :class:`~shared.label_space.LabelSpace` used to find legacy
        display-name columns. Defaults to the ten SHARES categories.

    Returns
    -------
    np.ndarray
        Shape ``(N, D)`` where ``D = len(category_names)``.
        Rows with no matching CSV entry are ``NaN``.
    """
    from shared.label_space import DEFAULT_LABEL_SPACE

    space = label_space or DEFAULT_LABEL_SPACE
    n = len(patient_ids)
    d = len(category_names)
    result = np.full((n, d), np.nan, dtype=float)

    if not csv_path.exists():
        logging.warning(f"Score CSV not found: {csv_path}")
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logging.warning(f"Failed to read score CSV {csv_path}: {exc}")
        return result

    if df.empty:
        logging.warning(f"Score CSV is empty: {csv_path}")
        return result

    # Build column resolution map: for each expected canonical key, find the
    # actual CSV column name (handles legacy files with display-label columns).
    col_map: dict[str, str | None] = {}
    for cat in category_names:
        if cat in df.columns:
            col_map[cat] = cat
        else:
            # Attempt legacy display-label fallback
            dim = space.get(cat)
            display_match: str | None = (
                dim.display_name
                if dim is not None and dim.display_name in df.columns
                else None
            )
            if display_match is not None:
                logging.warning(
                    f"Score CSV {csv_path.name}: column '{display_match}' "
                    f"used as legacy fallback for canonical key '{cat}'. "
                    f"Future score files should use canonical keys."
                )
                col_map[cat] = display_match
            else:
                col_map[cat] = None

    # Build a lookup: (patient_id_str, item_text_str) → row index in CSV
    csv_lookup: dict[tuple[str, str], int] = {}
    for row_idx, row in df.iterrows():
        key = (str(row.get("patient_id", "")), str(row.get("item_text", "")))
        csv_lookup[key] = int(row_idx)  # type: ignore[arg-type]

    for i, (pid, itxt) in enumerate(zip(patient_ids, item_texts)):
        key = (str(pid), str(itxt))
        if key in csv_lookup:
            csv_row_idx = csv_lookup[key]
            for j, cat in enumerate(category_names):
                actual_col = col_map[cat]
                if actual_col is not None:
                    val = df.at[csv_row_idx, actual_col]
                    try:
                        result[i, j] = float(val)
                    except (TypeError, ValueError):
                        result[i, j] = np.nan

    # Post-load validation: warn about dimensions that are entirely NaN
    for j, cat in enumerate(category_names):
        if col_map[cat] is None:
            logging.warning(
                f"Score CSV {csv_path.name}: canonical key '{cat}' not found "
                f"in columns. All values for this dimension are NaN."
            )
        elif np.all(np.isnan(result[:, j])):
            logging.warning(
                f"Score CSV {csv_path.name}: all values for dimension "
                f"'{cat}' are NaN (column present but no parseable values)."
            )

    return result


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------


def _run_model(
    model_id: str,
    cfg: dict[str, Any],
    paired_dataset: Any,
    context_builder: ContextBuilder,
    shuffled_context_map: dict[tuple[str, str], str],
    run_timestamp: str,
) -> None:
    """Run the full assay pipeline for a single model ID.

    Parameters
    ----------
    model_id:
        The Hugging Face model ID (or any identifier for dry_run).
    cfg:
        Validated configuration dictionary.
    paired_dataset:
        :class:`~assay.dataset_loader.PairedDataset` instance.
    context_builder:
        Initialised :class:`~assay.context_builder.ContextBuilder`.
    shuffled_context_map:
        Pre-built ``{(patient_id, item_text): shuffled_context_text}`` dict.
    run_timestamp:
        ISO 8601 timestamp string for the run.
    """
    logging.info("=" * 60)
    logging.info(f"Model: {model_id}")
    logging.info("=" * 60)

    cache_dir = cfg["cache_dir"]
    scores_dir = cfg["scores_dir"]
    reports_dir = cfg["reports_dir"]

    patient_ids = paired_dataset.patient_ids
    item_texts = paired_dataset.item_texts
    label_space = paired_dataset.label_space
    canonical_keys = paired_dataset.canonical_category_names

    # Build the list of all expected calls for this model
    expected_calls: list[tuple[str, str, str, str]] = []
    for pid, itxt in zip(patient_ids, item_texts):
        for condition in _CONDITIONS:
            expected_calls.append((model_id, condition, str(pid), str(itxt)))

    # --- Step a: Check cache status (resumability) ---
    cache = ResponseCache(cache_dir, expected_calls)
    status = cache.get_status()
    n_completed = len(status["completed"])
    n_failed = len(status["failed"])
    n_pending = len(status["pending"])

    # --- Step b: Log resumability report ---
    logging.info(
        f"Model {model_id}: {n_completed} completed, "
        f"{n_failed} failed, {n_pending} pending"
    )

    # --- Step c: Run LLM calls (or load from cache) ---
    client = LLMClient(cfg, label_space=label_space)
    template = PromptTemplate.from_config(cfg.get("prompt"), label_space)

    total_pairs = len(patient_ids)
    batch_size = int(cfg.get("local_batch_size", 8)) if cfg["backend"] == "local_transformers" else 1
    logging.info(
        f"Processing {total_pairs} patient-item pairs × 3 conditions "
        f"(batch_size={batch_size})..."
    )

    # Pre-scan cache and build the list of items that still need inference.
    # Collecting all pending work upfront allows batching across pairs and
    # conditions instead of issuing one GPU call per item. A cached response
    # made with a different prompt (for example after changing the label
    # space or the prompt settings) is stale and is run again.
    pending: list[dict[str, str]] = []
    n_stale = 0
    for pid, itxt in zip(patient_ids, item_texts):
        pid_str = str(pid)
        itxt_str = str(itxt)
        for condition in _CONDITIONS:
            if condition == "context_free":
                prompt = template.build_context_free_prompt(itxt_str)
            elif condition == "correct_context":
                correct_ctx = context_builder.build_correct_context(pid_str, itxt_str)
                prompt = template.build_correct_context_prompt(itxt_str, correct_ctx)
            else:
                shuffled_ctx = shuffled_context_map.get((pid_str, itxt_str), "")
                prompt = template.build_shuffled_context_prompt(itxt_str, shuffled_ctx)
            cached = cache.get(model_id, condition, pid_str, itxt_str)
            if cached is not None and cached.error is None:
                if cached.prompt_hash == PromptTemplate.get_prompt_hash(prompt):
                    continue
                n_stale += 1
            pending.append({
                "prompt": prompt,
                "model_id": model_id,
                "condition": condition,
                "patient_id": pid_str,
                "item_text": itxt_str,
            })

    if n_stale:
        logging.warning(
            f"  {n_stale} cached response(s) were made with a different prompt "
            "and will be run again."
        )
    n_pending = len(pending)
    logging.info(f"  {n_pending} items to infer ({total_pairs * len(_CONDITIONS) - n_pending} cached)")

    n_batches = (n_pending + batch_size - 1) // batch_size if n_pending else 0
    for batch_idx, batch_start in enumerate(range(0, n_pending, batch_size), start=1):
        batch = pending[batch_start:batch_start + batch_size]
        responses = client.batch_call(batch)
        for response in responses:
            cache.put(response)
        if batch_idx % 10 == 0 or batch_idx == n_batches:
            processed = min(batch_start + batch_size, n_pending)
            logging.info(f"  Inferred {processed}/{n_pending} items")

    # Report any failed calls
    failed_calls = client.get_failed_calls()
    if failed_calls:
        logging.warning(f"{len(failed_calls)} call(s) failed after all retries.")

    # --- Step d: Parse scores → CSVs ---
    logging.info("Parsing scores...")
    parser = ScoreParser(cache_dir, scores_dir, paired_dataset)
    parse_result = parser.parse_all(model_id)
    logging.info(
        "Scores written: "
        + ", ".join(
            f"{cond}={count}"
            for cond, count in parse_result["scores_written"].items()
        )
    )
    if parse_result["errors_written"] > 0:
        logging.warning(f"Errors written: {parse_result['errors_written']}")

    # --- Step e: Load score arrays from CSVs ---
    slug = LLMClient.model_id_slug(model_id)
    scores_model_dir = Path(scores_dir) / slug

    context_free_scores = _load_score_array(
        scores_model_dir / "context_free_scores.csv",
        patient_ids,
        item_texts,
        canonical_keys,
        label_space,
    )
    correct_context_scores = _load_score_array(
        scores_model_dir / "correct_context_scores.csv",
        patient_ids,
        item_texts,
        canonical_keys,
        label_space,
    )
    shuffled_context_scores = _load_score_array(
        scores_model_dir / "shuffled_context_scores.csv",
        patient_ids,
        item_texts,
        canonical_keys,
        label_space,
    )

    # Handle case where score arrays might be all NaN (no valid responses)
    has_cf_cc_valid = (
        not np.all(np.isnan(context_free_scores))
        and not np.all(np.isnan(correct_context_scores))
    )
    has_shuffled = not np.all(np.isnan(shuffled_context_scores))

    if not has_cf_cc_valid:
        logging.warning(
            "No valid scores found for context_free or correct_context. "
            "Skipping endpoint analysis."
        )
        return

    # Build valid row mask: require context_free + correct_context valid.
    # Shuffled is only needed for context specificity.
    valid_mask_cf_cc = (
        ~np.any(np.isnan(context_free_scores), axis=1)
        & ~np.any(np.isnan(correct_context_scores), axis=1)
    )

    if not np.any(valid_mask_cf_cc):
        logging.warning(
            "No rows with valid scores across context_free and correct_context. "
            "Skipping endpoint analysis."
        )
        return

    n_valid = int(np.sum(valid_mask_cf_cc))
    n_total = len(valid_mask_cf_cc)
    if n_valid < n_total:
        logging.info(
            f"{n_valid}/{n_total} rows have valid context_free + correct_context "
            "scores. Endpoint analysis restricted to valid rows."
        )

    # Subset to valid rows
    cf_scores = context_free_scores[valid_mask_cf_cc]
    cc_scores = correct_context_scores[valid_mask_cf_cc]
    delta_phys = paired_dataset.delta_physician[valid_mask_cf_cc]
    valid_patient_ids = patient_ids[valid_mask_cf_cc]

    # --- Step f: Compute the endpoints ---
    logging.info("Running hypothesis analysis...")

    # Build a minimal paired_dataset-like object with valid_patient_ids
    # for the HypothesisAnalyzer (it only needs patient_ids and category_names)
    class _SubsetDataset:
        def __init__(self, pids: np.ndarray, cats: list[str]) -> None:
            self.patient_ids = pids
            self.category_names = cats

    subset_dataset = _SubsetDataset(valid_patient_ids, list(canonical_keys))
    analyzer = HypothesisAnalyzer(subset_dataset, cfg)

    h1_result = analyzer.compute_h1(cf_scores, cc_scores)
    logging.info(f"Context sensitivity (mean |delta|): {h1_result['mean_abs_delta']:.4f}")

    h2_result = analyzer.compute_h2(cf_scores, cc_scores, delta_phys)
    logging.info(
        f"Directional alignment (sign agreement rate): {h2_result['sign_agreement_rate']:.4f}"
    )

    h3_result = analyzer.compute_h3(cf_scores, cc_scores, delta_phys)
    logging.info(f"Class-level correspondence (Pearson r): {h3_result['pearson_r']:.4f}")

    # Context specificity requires shuffled-context scores
    if has_shuffled:
        # Build mask requiring all three conditions valid
        valid_mask_all = valid_mask_cf_cc & ~np.any(np.isnan(shuffled_context_scores), axis=1)
        if np.any(valid_mask_all):
            sc_scores = shuffled_context_scores[valid_mask_all]
            cf_h4 = context_free_scores[valid_mask_all]
            cc_h4 = correct_context_scores[valid_mask_all]
            dp_h4 = paired_dataset.delta_physician[valid_mask_all]
            pids_h4 = patient_ids[valid_mask_all]

            subset_h4 = _SubsetDataset(pids_h4, list(canonical_keys))
            analyzer_h4 = HypothesisAnalyzer(subset_h4, cfg)
            h4_result = analyzer_h4.compute_h4(cf_h4, cc_h4, sc_scores, dp_h4)
            logging.info(f"Context specificity (mean diff): {h4_result['mean_diff']:.4f}")
        else:
            h4_result = {
                "status": "skipped",
                "reason": "No rows with valid scores across all three conditions.",
            }
            logging.info(
                "Context specificity skipped: no rows with valid scores across all "
                "three conditions."
            )
    else:
        h4_result = {
            "status": "skipped",
            "reason": "shuffled_context_scores absent or all-NaN",
        }
        logging.info("Context specificity skipped: shuffled-context scores not available.")

    # --- Step g: Generate report ---
    logging.info("Generating report...")
    report_dir = Path(reports_dir) / slug
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = str(report_dir / "analysis_report.md")

    generator = ReportGenerator(
        h1_result=h1_result,
        h2_result=h2_result,
        h3_result=h3_result,
        h4_result=h4_result,
        model_id=model_id,
        run_timestamp=run_timestamp,
        label_space=label_space,
        category_type=template.category_type,
        legacy_aliases={te.name: te.legacy_alias for te in TRACK_ENDPOINTS},
        is_dry_run=(cfg.get("backend") == "dry_run"),
        n_context_entities=len(set(patient_ids.tolist())),
        n_task_pairs=len(patient_ids),
    )
    generator.generate(report_path)
    logging.info(f"Report written to {report_path}")


# ---------------------------------------------------------------------------
# Config validation helper (validates an already-loaded dict)
# ---------------------------------------------------------------------------


def _validate_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a configuration dict (already loaded from YAML).

    Applies the same validation rules as :func:`assay.config_loader.load_config`
    but operates on an in-memory dict rather than a file path.  This allows
    CLI overrides (e.g. ``--dry_run``) to be applied before validation.

    Parameters
    ----------
    cfg:
        Configuration dictionary to validate.

    Returns
    -------
    dict
        The validated configuration dictionary (same object, returned for
        convenience).

    Raises
    ------
    ConfigError
        On any validation failure.
    """
    import os

    # Required top-level keys
    missing_top = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing_top:
        raise ConfigError(
            f"Missing required configuration parameter(s): {', '.join(missing_top)}"
        )

    # Backend
    backend: str = cfg["backend"]
    if backend not in VALID_BACKENDS:
        raise ConfigError(
            f"Invalid backend '{backend}'. Must be one of: "
            f"{', '.join(sorted(VALID_BACKENDS))}"
        )

    # model_ids
    _validate_model_ids(cfg["model_ids"])

    # HF credentials (only required for huggingface backend)
    if backend == "huggingface":
        if not cfg.get("hf_token"):
            hf_token_env: str | None = cfg.get("hf_token_env")
            if not (hf_token_env and os.environ.get(hf_token_env)):
                raise ConfigError(
                    "backend=huggingface requires a Hugging Face token. "
                    "Set 'hf_token' in the config or provide 'hf_token_env' "
                    "pointing to a non-empty environment variable (e.g., HF_TOKEN)."
                )

    # data section
    data_section = cfg.get("data")
    if not isinstance(data_section, dict):
        raise ConfigError("The 'data' configuration key must be a mapping.")

    missing_data = [k for k in REQUIRED_DATA_KEYS if k not in data_section]
    if missing_data:
        raise ConfigError(
            f"Missing required 'data' configuration parameter(s): "
            f"{', '.join(missing_data)}"
        )

    validate_label_space_config(data_section)
    validate_prompt_config(cfg.get("prompt"))
    return cfg


# ---------------------------------------------------------------------------
# Failed cache clearing helper
# ---------------------------------------------------------------------------


def _clear_failed_cache_entries(cache_dir: str | Path) -> None:
    """Delete all cache JSON files whose ``error`` field is not null.

    Parameters
    ----------
    cache_dir:
        Root directory to search recursively for ``*.json`` cache files.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logging.info("Cache directory does not exist; nothing to clear.")
        return

    cleared = 0
    for json_file in cache_path.rglob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            logging.debug(f"Skipping unparseable cache file: {json_file}")
            continue

        if isinstance(data, dict) and data.get("error") is not None:
            try:
                os.unlink(json_file)
                cleared += 1
            except OSError as exc:
                logging.warning(f"Could not delete {json_file}: {exc}")

    logging.info(f"Cleared {cleared} failed cache entries")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the assay runner.

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

    # --- Bootstrap: peek at output_dir from YAML before full config load ---
    # We need a run folder (and log file) as early as possible, even before
    # config validation, so errors during loading are captured in the log.
    output_dir = _DEFAULT_OUTPUT_DIR
    config_path = Path(args.config)
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            _raw = yaml.safe_load(fh)
        if isinstance(_raw, dict) and _raw.get("output_dir"):
            output_dir = str(_raw["output_dir"])
    except Exception:
        pass  # Fall back to default; error will surface during proper load below.

    run_folder = _create_run_folder(output_dir)
    _setup_logging(run_folder, log_level="INFO")

    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Loading config from: {args.config}")

    # --- Step 1: Load and validate config ---
    # Load the raw YAML so we can apply CLI overrides before validation.
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        return 1
    except Exception as exc:
        logging.error(f"Failed to read configuration: {exc}")
        return 1

    if not isinstance(cfg, dict):
        logging.error("Configuration file must contain a YAML mapping.")
        return 1

    # Apply CLI overrides before validation
    if args.dry_run:
        cfg["backend"] = "dry_run"
        logging.info("Backend overridden to: dry_run")

    if args.model:
        cfg["model_ids"] = [args.model]
        logging.info(f"Model overridden to: {args.model}")

    try:
        cfg = _validate_config_dict(cfg)
    except ConfigError as exc:
        logging.error(f"Configuration validation failed: {exc}")
        return 1

    # Override scores_dir and reports_dir to live inside the timestamped run folder.
    # Cache stays at its configured global location for cross-run resumability.
    cfg["scores_dir"] = str(run_folder / "scores")
    cfg["reports_dir"] = str(run_folder / "reports")

    model_ids: list[str] = cfg["model_ids"]
    logging.info(f"Models to run: {model_ids}")
    logging.info(f"Backend: {cfg['backend']}")
    logging.info(f"Cache dir: {cfg['cache_dir']}")
    logging.info(f"Scores dir: {cfg['scores_dir']}")
    logging.info(f"Reports dir: {cfg['reports_dir']}")

    # --- Step 2: Load dataset → PairedDataset ---
    logging.info("Loading dataset...")
    try:
        loader = DatasetLoader(cfg)
        paired_dataset = loader.load()
    except Exception as exc:
        logging.error(f"Failed to load dataset: {exc}")
        return 1

    undefined = [d.key for d in paired_dataset.label_space if not d.definition]
    if undefined:
        logging.warning(
            "No definition for output dimension(s) %s; the prompt lists them by key "
            "only. Add them to data.class_definitions to describe them to the model.",
            undefined,
        )

    n_pairs = len(paired_dataset.patient_ids)
    logging.info(f"Dataset loaded: {n_pairs} patient-item pairs")

    # --- Step 3: Build shuffled context map (deterministic, seeded) ---
    logging.info("Building shuffled context map...")
    try:
        context_builder = ContextBuilder(
            cfg["data"]["patient_summaries_path"],
            patient_ids=list(paired_dataset.patient_ids),
            skip_label_leakage_check=cfg.get("skip_label_leakage_check", False),
        )
        shuffled_context_map = _build_shuffled_context_map(
            context_builder=context_builder,
            patient_ids=paired_dataset.patient_ids,
            item_texts=paired_dataset.item_texts,
            seed=cfg["shuffled_context_seed"],
        )
    except Exception as exc:
        logging.error(f"Failed to build shuffled context map: {exc}")
        return 1

    logging.info(f"Shuffled context map built: {len(shuffled_context_map)} unique pairs")

    # Capture run timestamp once (shared across all models)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"

    # --- Step 4 (optional): Clear failed cache entries before running ---
    if args.clear_failed:
        _clear_failed_cache_entries(cfg["cache_dir"])

    # --- Step 5: For each model ID, run the full pipeline ---
    for model_id in model_ids:
        try:
            _run_model(
                model_id=model_id,
                cfg=cfg,
                paired_dataset=paired_dataset,
                context_builder=context_builder,
                shuffled_context_map=shuffled_context_map,
                run_timestamp=run_timestamp,
            )
        except Exception as exc:
            logging.error(f"Pipeline failed for model '{model_id}': {exc}")
            # Continue with remaining models rather than aborting entirely
            continue

    logging.info("Assay run complete.")

    # --- Emit manifests ---
    _emit_assay_manifests(run_folder, cfg, model_ids, loader)

    return 0


def _emit_assay_manifests(
    run_folder: Path,
    cfg: dict[str, Any],
    model_ids: list[str],
    loader: DatasetLoader,
) -> None:
    """Emit run_manifest.json, adapter_manifest.json, artifact_index.json for Track 3."""
    from shared.prerequisites.manifests import (
        write_adapter_manifest,
        write_artifact_index,
        write_resolved_config,
        write_run_manifest,
    )
    from tracks.reasoning.llm_client import LLMClient

    run_id = run_folder.name
    is_dry_run = cfg.get("backend") == "dry_run"

    write_run_manifest(
        run_folder,
        run_id=run_id,
        track="reasoning",
        entry_command="python main.py --track reasoning",
        config_path=cfg.get("_config_path"),
        dry_run=is_dry_run,
        extra={"model_ids": model_ids},
    )

    # Adapter manifest
    data_section = cfg.get("data", {})
    write_adapter_manifest(
        run_folder,
        adapter_name=loader.adapter_name,
        adapter_class=loader.adapter_class,
        label_space=loader.manifest()["label_mapping"],
        dataset_path=data_section.get("dataset_path"),
        patient_summaries_path=data_section.get("patient_summaries_path"),
        mapping_file_path=data_section.get("mapping_file_path"),
    )

    # Artifact index
    role_to_path: dict[str, str] = {}
    for model_id in model_ids:
        slug = LLMClient.model_id_slug(model_id)
        scores_dir = Path(cfg["scores_dir"]) / slug
        reports_dir = Path(cfg["reports_dir"]) / slug

        if (scores_dir / "context_free_scores.csv").exists():
            role_to_path["track3.scores.context_free"] = str(
                (scores_dir / "context_free_scores.csv").relative_to(run_folder)
            )
        if (scores_dir / "correct_context_scores.csv").exists():
            role_to_path["track3.scores.correct_context"] = str(
                (scores_dir / "correct_context_scores.csv").relative_to(run_folder)
            )
        if (scores_dir / "shuffled_context_scores.csv").exists():
            role_to_path["track3.scores.shuffled_context"] = str(
                (scores_dir / "shuffled_context_scores.csv").relative_to(run_folder)
            )
        if (reports_dir / "analysis_report.md").exists():
            role_to_path["track3.hypothesis_summary"] = str(
                (reports_dir / "analysis_report.md").relative_to(run_folder)
            )

    write_artifact_index(run_folder, role_to_path)
    write_resolved_config(run_folder, cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
