#!/usr/bin/env python3
"""
Analyze LLM context effects and generate a full analysis bundle.

Consumes pre-generated LLM score CSVs and physician consensus data, then
produces paired cell deltas, the four Track 3 endpoint results
(shared/endpoints.py), a Markdown report,
and a ZIP bundle.

Usage (assay scores directory — auto-discovers all model subdirectories):
    python scripts/analyze_llm_context_effects.py \\
        --dataset dataset.xlsx \\
        --scores-dir output/assay/scores \\
        --output-dir outputs/analysis

    The scores directory must follow the layout written by run_assay.py:
        {scores_dir}/{model_slug}/context_free_scores.csv
        {scores_dir}/{model_slug}/correct_context_scores.csv
        {scores_dir}/{model_slug}/shuffled_context_scores.csv
    The model name is recovered from the slug (-- → /).

Usage (three merged CSV files, one model or pre-merged across models):
    python scripts/analyze_llm_context_effects.py \\
        --dataset dataset.xlsx \\
        --context-free outputs/llm_judgments/context_free_scores.csv \\
        --correct-context outputs/llm_judgments/correct_context_scores.csv \\
        --shuffled-context outputs/llm_judgments/shuffled_context_scores.csv \\
        --output-dir outputs/analysis \\
        --seed 2026 --epsilon 0.01 --n-bootstrap 1000 --n-permutations 10000

Usage (combined long-format file with 'condition' column):
    python scripts/analyze_llm_context_effects.py \\
        --dataset dataset.xlsx \\
        --llm-scores outputs/llm_judgments/parsed_scores.csv \\
        --output-dir outputs/analysis

The categories default to the ten SHARES categories. For another taxonomy,
pass the assay config with --config; its data.classes and
data.class_definitions are used.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml

from shared.label_space import DEFAULT_LABEL_SPACE, LabelSpace
from tracks.reasoning.bundle.loader import load_llm_scores, load_physician_consensus
from tracks.reasoning.prompt_template import DEFAULT_CATEGORY_TYPE
from tracks.reasoning.bundle.validator import validate_inputs
from tracks.reasoning.bundle.delta_builder import build_paired_cell_deltas
from tracks.reasoning.bundle.hypotheses import compute_h1, compute_h2, compute_h3, compute_h4
from tracks.reasoning.bundle.summary import make_hypothesis_summary, make_model_comparison_summary
from tracks.reasoning.bundle.report import generate_markdown_report
from tracks.reasoning.bundle.bundle import write_analysis_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("analyze_llm_context_effects")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full LLM context-shift analysis bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input files
    parser.add_argument("--dataset", required=True, help="Path to dataset.xlsx")
    parser.add_argument(
        "--scores-dir", dest="scores_dir", default=None,
        help=(
            "Root directory written by run_assay.py "
            "(e.g. output/assay/scores). Each subdirectory is treated as one "
            "model slug and must contain context_free_scores.csv, "
            "correct_context_scores.csv, and shuffled_context_scores.csv. "
            "Takes priority over --context-free / --correct-context / "
            "--shuffled-context when supplied."
        ),
    )
    parser.add_argument(
        "--context-free", dest="context_free", default=None,
        help="Path to context_free_scores.csv (single model or pre-merged multi-model)",
    )
    parser.add_argument(
        "--correct-context", dest="correct_context", default=None,
        help="Path to correct_context_scores.csv",
    )
    parser.add_argument(
        "--shuffled-context", dest="shuffled_context", default=None,
        help="Path to shuffled_context_scores.csv",
    )
    parser.add_argument(
        "--llm-scores", dest="llm_scores", default=None,
        help="Combined long-format CSV with 'model' and 'condition' columns",
    )

    # Output
    parser.add_argument(
        "--output-dir", dest="output_dir", default="outputs/analysis",
        help="Directory for all output files (default: outputs/analysis)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files",
    )

    # Statistical parameters
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed (default: 2026)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Context-change threshold (default: 0.01)")
    parser.add_argument("--n-bootstrap", dest="n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples (default: 1000)")
    parser.add_argument("--n-permutations", dest="n_permutations", type=int, default=10000,
                        help="Permutation iterations (default: 10000)")

    # Dataset column overrides
    parser.add_argument("--patient-col", dest="patient_col", default="Patient")
    parser.add_argument("--physician-col", dest="physician_col", default="Physician")
    parser.add_argument("--item-col", dest="item_col", default="Item")
    parser.add_argument("--train-sheet", dest="train_sheet", default="train")
    parser.add_argument("--survey-sheet", dest="survey_sheet", default="test")
    parser.add_argument("--interview-sheet", dest="interview_sheet", default="interview")
    parser.add_argument(
        "--config", default=None,
        help=(
            "Assay config whose data.classes and data.class_definitions define the "
            "categories, and whose prompt.category_type sets the report wording "
            "(default: the ten SHARES privacy categories)"
        ),
    )

    return parser.parse_args(argv)


def _load_label_space(config_path: str | None) -> tuple[LabelSpace, dict | None, str]:
    """Label space, class_definitions and category type from an assay config.

    Without a config: the ten SHARES categories and ``"privacy"``.
    """
    if config_path is None:
        return DEFAULT_LABEL_SPACE, None, DEFAULT_CATEGORY_TYPE
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    data_cfg = cfg.get("data") or {}
    if "classes" not in data_cfg:
        raise SystemExit(f"{config_path}: no data.classes entry")
    definitions = data_cfg.get("class_definitions")
    category_type = (cfg.get("prompt") or {}).get("category_type", DEFAULT_CATEGORY_TYPE)
    return (
        LabelSpace.from_config(data_cfg["classes"], definitions),
        definitions,
        category_type,
    )


_CONDITION_FILES = {
    "context_free": "context_free_scores.csv",
    "correct_context": "correct_context_scores.csv",
    "shuffled_context": "shuffled_context_scores.csv",
}


def merge_scores_dir(scores_dir: str) -> tuple[str, str, str]:
    """Merge per-model score subdirectories written by run_assay.py.

    Expects the layout::

        {scores_dir}/
          {model_slug_a}/
            context_free_scores.csv
            correct_context_scores.csv
            shuffled_context_scores.csv
          {model_slug_b}/
            ...

    The model name is recovered by replacing ``--`` with ``/`` in the slug
    (matching the slug convention in ``assay.llm_client.LLMClient.model_id_slug``).

    Returns three paths to merged temporary CSV files written alongside the
    scores_dir:  ``{scores_dir}/_merged_{condition}.csv``.  Each file has a
    ``model`` column containing the recovered model name.

    Parameters
    ----------
    scores_dir:
        Root directory containing per-model subdirectories.

    Returns
    -------
    (context_free_path, correct_context_path, shuffled_context_path)
    """
    root = Path(scores_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--scores-dir '{scores_dir}' does not exist or is not a directory.")

    model_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    if not model_dirs:
        raise FileNotFoundError(f"No model subdirectories found in '{scores_dir}'.")

    merged: dict[str, list[pd.DataFrame]] = {c: [] for c in _CONDITION_FILES}
    found_models: list[str] = []

    for model_dir in model_dirs:
        # Recover model name from slug: "google--medgemma-1.5-4b-it" → "google/medgemma-1.5-4b-it"
        model_name = model_dir.name.replace("--", "/", 1)
        missing = []
        for cond, fname in _CONDITION_FILES.items():
            csv_path = model_dir / fname
            if not csv_path.exists():
                missing.append(fname)
                continue
            df = pd.read_csv(csv_path)
            df["model"] = model_name
            merged[cond].append(df)

        if missing:
            logger.warning(
                "Model '%s': missing score file(s) %s — skipping this model.",
                model_name, missing,
            )
        else:
            found_models.append(model_name)

    if not found_models:
        raise ValueError(
            f"No complete model score sets found in '{scores_dir}'. "
            "Each model subdirectory must contain all three condition CSVs."
        )

    logger.info(
        "Merged %d model(s) from '%s': %s",
        len(found_models), scores_dir, found_models,
    )

    out_paths: dict[str, str] = {}
    for cond, dfs in merged.items():
        if not dfs:
            raise ValueError(f"No data for condition '{cond}' after merging.")
        combined = pd.concat(dfs, ignore_index=True)
        out_path = str(root / f"_merged_{cond}.csv")
        combined.to_csv(out_path, index=False)
        out_paths[cond] = out_path
        logger.info("  Written: %s (%d rows)", out_path, len(combined))

    return out_paths["context_free"], out_paths["correct_context"], out_paths["shuffled_context"]


def _check_output_collision(output_dir: str, overwrite: bool) -> None:
    """Error if any expected output files already exist and --overwrite not set."""
    expected = [
        "paired_cell_deltas.csv",
        "hypothesis_summary.csv",
        "context_sensitivity.csv",
        "directional_alignment.csv",
        "class_correspondence_effects.csv",
        "class_correspondence.csv",
        "context_specificity.csv",
        "model_comparison_summary.csv",
        "validation_summary.csv",
        "LLM_CONTEXT_SHIFT_REPORT.md",
    ]
    if not overwrite:
        existing = [
            f for f in expected if os.path.exists(os.path.join(output_dir, f))
        ]
        if existing:
            logger.error(
                "Output files already exist in '%s': %s. "
                "Use --overwrite to replace them.",
                output_dir,
                existing,
            )
            sys.exit(1)


def _save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    logger.info("Saved: %s (%d rows)", path, len(df))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    _check_output_collision(args.output_dir, args.overwrite)

    out = lambda name: os.path.join(args.output_dir, name)

    label_space, class_definitions, category_type = _load_label_space(args.config)
    categories = label_space.keys()
    run_params = {
        "seed": args.seed,
        "epsilon": args.epsilon,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "dataset": args.dataset,
        "categories": categories,
    }

    # ------------------------------------------------------------------
    # Step 0: Resolve score inputs
    # ------------------------------------------------------------------
    if args.scores_dir is not None:
        if args.llm_scores or args.context_free or args.correct_context or args.shuffled_context:
            logger.warning(
                "--scores-dir supplied alongside other score inputs. "
                "Using --scores-dir and ignoring the other inputs."
            )
        logger.info("Merging per-model score directories from: %s", args.scores_dir)
        args.context_free, args.correct_context, args.shuffled_context = merge_scores_dir(
            args.scores_dir
        )
        args.llm_scores = None  # ensure three-file path is taken
        run_params["scores_dir"] = args.scores_dir

    # ------------------------------------------------------------------
    # Step 1: Load physician consensus
    # ------------------------------------------------------------------
    logger.info("Loading physician consensus from %s ...", args.dataset)
    physician_df = load_physician_consensus(
        dataset_path=args.dataset,
        patient_col=args.patient_col,
        physician_col=args.physician_col,
        item_col=args.item_col,
        train_sheet=args.train_sheet,
        survey_sheet=args.survey_sheet,
        interview_sheet=args.interview_sheet,
        excel_classes=label_space.display_names(),
        class_definitions=class_definitions,
    )

    # ------------------------------------------------------------------
    # Step 2: Load LLM scores
    # ------------------------------------------------------------------
    logger.info("Loading LLM scores ...")
    scores_long_df = load_llm_scores(
        context_free=args.context_free,
        correct_context=args.correct_context,
        shuffled_context=args.shuffled_context,
        llm_scores=args.llm_scores,
        categories=categories,
    )

    # ------------------------------------------------------------------
    # Step 3: Validate inputs
    # ------------------------------------------------------------------
    logger.info("Validating inputs ...")
    validation_summary, failed_df = validate_inputs(
        physician_df, scores_long_df, categories, epsilon=args.epsilon
    )
    _save(validation_summary, out("validation_summary.csv"))
    _save(failed_df, out("failed_or_missing_rows.csv"))

    n_fails = (validation_summary["status"] == "FAIL").sum()
    if n_fails > 0:
        logger.warning(
            "%d validation checks failed. Analysis will continue but results "
            "may be unreliable. See validation_summary.csv for details.",
            n_fails,
        )

    # ------------------------------------------------------------------
    # Step 4: Build paired cell deltas
    # ------------------------------------------------------------------
    logger.info("Building paired cell deltas ...")
    cells_df = build_paired_cell_deltas(
        physician_df, scores_long_df, categories, epsilon=args.epsilon
    )
    _save(cells_df, out("paired_cell_deltas.csv"))

    # ------------------------------------------------------------------
    # Step 5: Per-model hypothesis analyses
    # ------------------------------------------------------------------
    models = sorted(cells_df["model"].unique())
    logger.info("Running hypothesis analyses for %d model(s): %s", len(models), models)

    all_h1, all_h2, all_h3_effects, all_h3_corr, all_h4 = [], [], [], [], []

    for model in models:
        logger.info("  Model: %s", model)
        model_cells = cells_df[cells_df["model"] == model].copy()

        h1 = compute_h1(model_cells, model, categories, args.epsilon, args.n_bootstrap, args.seed)
        h2 = compute_h2(model_cells, model, categories, args.epsilon, args.n_bootstrap, args.seed)
        h3_eff, h3_corr = compute_h3(model_cells, model, categories, args.n_permutations, args.seed)
        h4 = compute_h4(model_cells, model, categories, args.n_bootstrap, args.n_permutations, args.seed)

        all_h1.append(h1)
        all_h2.append(h2)
        all_h3_effects.append(h3_eff)
        all_h3_corr.append(h3_corr)
        all_h4.append(h4)

    h1_df = pd.concat(all_h1, ignore_index=True)
    h2_df = pd.concat(all_h2, ignore_index=True)
    h3_effects_df = pd.concat(all_h3_effects, ignore_index=True)
    h3_corr_df = pd.concat(all_h3_corr, ignore_index=True)
    h4_df = pd.concat(all_h4, ignore_index=True)

    _save(h1_df, out("context_sensitivity.csv"))
    _save(h2_df, out("directional_alignment.csv"))
    _save(h3_effects_df, out("class_correspondence_effects.csv"))
    _save(h3_corr_df, out("class_correspondence.csv"))
    _save(h4_df, out("context_specificity.csv"))

    # ------------------------------------------------------------------
    # Step 6: Summaries
    # ------------------------------------------------------------------
    logger.info("Computing hypothesis and model comparison summaries ...")
    hypothesis_summary = make_hypothesis_summary(
        h1_df, h2_df, h3_corr_df, h4_df, cells_df, failed_df, epsilon=args.epsilon
    )
    model_comparison = make_model_comparison_summary(hypothesis_summary)

    _save(hypothesis_summary, out("hypothesis_summary.csv"))
    _save(model_comparison, out("model_comparison_summary.csv"))

    # ------------------------------------------------------------------
    # Step 7: Markdown report
    # ------------------------------------------------------------------
    report_path = out("LLM_CONTEXT_SHIFT_REPORT.md")
    logger.info("Generating Markdown report: %s ...", report_path)
    generate_markdown_report(
        hypothesis_summary=hypothesis_summary,
        h1_df=h1_df,
        h2_df=h2_df,
        h3_effects_df=h3_effects_df,
        h3_corr_df=h3_corr_df,
        h4_df=h4_df,
        model_comparison=model_comparison,
        validation_summary=validation_summary,
        output_path=report_path,
        run_params=run_params,
        categories=categories,
        category_type=category_type,
    )

    # ------------------------------------------------------------------
    # Step 8: ZIP bundle
    # ------------------------------------------------------------------
    logger.info("Creating analysis bundle ZIP ...")
    zip_path = write_analysis_bundle(args.output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS BUNDLE COMPLETE")
    print("=" * 60)
    print(f"Output directory : {os.path.abspath(args.output_dir)}")
    print(f"Models analyzed  : {', '.join(models)}")
    print(f"Validation       : {(validation_summary['status'] == 'PASS').sum()} passed, "
          f"{n_fails} failed")
    print(f"Patients         : {cells_df['patient'].nunique() if 'patient' in cells_df.columns else '?'}")
    print(f"Total cells      : {len(cells_df)}")
    print("\nFiles created:")
    for fname in [
        "paired_cell_deltas.csv",
        "validation_summary.csv",
        "failed_or_missing_rows.csv",
        "context_sensitivity.csv",
        "directional_alignment.csv",
        "class_correspondence_effects.csv",
        "class_correspondence.csv",
        "context_specificity.csv",
        "hypothesis_summary.csv",
        "model_comparison_summary.csv",
        "LLM_CONTEXT_SHIFT_REPORT.md",
        "analysis_bundle.zip",
    ]:
        full = os.path.join(args.output_dir, fname)
        status = "✓" if os.path.exists(full) else "✗"
        print(f"  {status} {fname}")

    print("\nExample re-run command:")
    scores_input = (
        f" --scores-dir {args.scores_dir}" if args.scores_dir else
        f" --llm-scores {args.llm_scores}" if args.llm_scores else
        f" --context-free {args.context_free}"
        f" --correct-context {args.correct_context}"
        f" --shuffled-context {args.shuffled_context}"
    )
    print(
        f"  python scripts/analyze_llm_context_effects.py"
        f" --dataset {args.dataset}"
        + scores_input
        + f" --output-dir {args.output_dir}"
        f" --seed {args.seed}"
        f" --epsilon {args.epsilon}"
        f" --n-bootstrap {args.n_bootstrap}"
        f" --n-permutations {args.n_permutations}"
        + (f" --config {args.config}" if args.config else "")
        + " --overwrite"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
