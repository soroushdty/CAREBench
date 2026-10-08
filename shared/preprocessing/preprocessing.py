"""shared.preprocessing.preprocessing — Main preprocessing pipeline.

Provides the canonical `preprocessing()` (pure, returns a PreprocessingResult)
and `preprocess()` (also writes index maps and summary artifacts) entry points.

Canonical path: shared/preprocessing/preprocessing.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
import logging
import pandas as pd

from shared.utils.json_utils import load_json_from_source
from shared.utils.mapping_utils import is_valid_grouped_mapping_obj
from .input_loading import (
    _config_root,
    _resolve_cfg_path,
    _resolve_input_path,
    _resolve_split_sheet_names,
    _read_dataset,
    _assert_raw_test_eval_items_equal,
    _assert_item_column_exists,
)
from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
from shared.reference.diagnostics.duplicate_response_audit import _merge_test_interview
from .item_standardization import standardize_items
from shared.utils.text_utils import normalize_for_matching
from shared.reporting.input_summary import save_index_mapping_outputs, summary as run_summary

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Return value of preprocessing(). Separates computed data from write targets.

    The caller (run()) owns all I/O: index-map saving and summary generation.
    """
    df: pd.DataFrame
    split_index_maps: dict          # passed to save_index_mapping_outputs
    summary_kwargs: dict | None     # kwargs for run_summary (exc. cfg); None when ENABLE_SUMMARY=False
    # Per-split boolean Series: True = row item was covered by JSON mapping.
    # Keys are the raw sheet names (train_sheet_name, test_sheet_name, eval_sheet_name).
    item_resolved_by_json: dict = field(default_factory=dict)


def preprocessing(
    cfg: dict,
    sheet_names: Iterable[str] | None = None,
    *,
    enforce_raw_test_eval_equality: bool | None = None,
) -> PreprocessingResult:
    """Run preprocessing pipeline and return a PreprocessingResult.

    Pipeline stages:
      raw → json_standardized (if JSON available, else unchanged) →
      post_physician_merge → final

    The result's ``df`` DataFrame contains:
      * ``split = 'train'``: physician-aggregated training rows.
      * ``split = 'test'``: physician-aggregated test rows LEFT-JOINED with
        physician-aggregated interview rows.  Interview label columns carry an
        ``_interview`` suffix.
      * ``item_json_resolved`` (bool): True when the row's item was mapped by
        JSON; False when unresolved (candidates for later fuzzy fallback).

    Fuzzy fallback is NOT performed here.  It is a separate train-only runtime
    stage that must run after CV fold construction to avoid leaking held-out
    patient vocabulary into the fuzzy candidate space.

    No I/O is performed here. The caller (run()) is responsible for writing
    the index-map JSON and generating summary artifacts.
    """

    dataset_path = _resolve_input_path(cfg)
    (
        train_sheet_name,
        test_sheet_name,
        eval_sheet_name,
        selected_sheets,
    ) = _resolve_split_sheet_names(cfg, sheet_names)

    dfs = _read_dataset(dataset_path, selected_sheets, cfg=cfg)
    summary_enabled = bool(cfg.get("ENABLE_SUMMARY", False))

    item_col = cfg["item_col"]
    patient_col = cfg.get("patient_col")
    physician_col = cfg.get("physician_col")
    class_cols = list(cfg["classes"])
    _assert_item_column_exists(dfs=dfs, item_col=item_col)

    raw_splits = (
        {
            "train": dfs[train_sheet_name].copy(deep=True),
            "test": dfs[test_sheet_name].copy(deep=True),
            "interview": dfs[eval_sheet_name].copy(deep=True),
        }
        if summary_enabled
        else {}
    )
    _assert_raw_test_eval_items_equal(
        test_df=dfs[test_sheet_name],
        eval_df=dfs[eval_sheet_name],
        item_col=item_col,
        patient_col=patient_col,
        physician_col=physician_col,
    )
    logger.info(
        "Raw load equality check passed: test and eval '%s' values are exactly equal.",
        item_col,
    )

    # ── Resolve JSON mapping source ──────────────────────────────────────────
    mapping_source: Mapping[str, Iterable[str]] | str | Path | None = None

    mapping_path_value = cfg.get("DIR_JSON_MAP")
    mapping_is_url = False
    mapping_path: Path | None = None
    if mapping_path_value:
        parsed = urlparse(str(mapping_path_value))
        mapping_is_url = parsed.scheme in {"http", "https"}
        if not mapping_is_url:
            mapping_path = Path(_resolve_cfg_path(cfg, "DIR_JSON_MAP"))

    mapping_file_is_valid = False
    if mapping_path is not None and mapping_path.exists() and mapping_path.is_file():
        try:
            mapping_file_is_valid = is_valid_grouped_mapping_obj(
                load_json_from_source(
                    mapping_path,
                    allow_json_payload=False,
                    base_dir=_config_root(cfg),
                )
            )
        except Exception:
            mapping_file_is_valid = False

    # Hardcoded behavior: do not strip spaces during standardization/mapping lookup.
    # Space trimming is reserved for test/eval equality auditing only.
    mapping_remove_spaces = False
    if mapping_is_url and mapping_path_value:
        loaded = load_json_from_source(
            str(mapping_path_value),
            allow_json_payload=False,
            base_dir=_config_root(cfg),
        )
        if not is_valid_grouped_mapping_obj(loaded):
            raise ValueError(
                "Remote mapping JSON is not a valid grouped mapping. "
                "Expected {canonical_item: [alias1, alias2, ...]} format."
            )
        mapping_source = loaded
        logger.info(
            "Item standardization method: JSON mapping. "
            "Using remote mapping from cfg['DIR_JSON_MAP']='%s'.",
            mapping_path_value,
        )
    elif mapping_path is not None and mapping_file_is_valid:
        mapping_source = mapping_path
        logger.info(
            "Item standardization method: JSON mapping. "
            "Using local mapping file from cfg['DIR_JSON_MAP'] at '%s'.",
            mapping_path,
        )
    else:
        # No valid JSON mapping: items will be left unresolved for the later
        # train-only fuzzy fallback stage.  This is correct — do NOT build a
        # fuzzy mapping here from the full training sheet, which would leak
        # held-out patient vocabulary before CV fold construction.
        if mapping_path is None and not mapping_is_url:
            logger.info(
                "Item standardization method: none (no DIR_JSON_MAP configured). "
                "Items will remain unresolved for later train-only fuzzy fallback."
            )
        elif mapping_path is not None and not mapping_path.exists():
            logger.info(
                "Item standardization method: none (DIR_JSON_MAP not found at '%s'). "
                "Items will remain unresolved for later train-only fuzzy fallback.",
                mapping_path,
            )
        elif mapping_path is not None and not mapping_path.is_file():
            logger.info(
                "Item standardization method: none (DIR_JSON_MAP is not a file: '%s'). "
                "Items will remain unresolved for later train-only fuzzy fallback.",
                mapping_path,
            )
        else:
            logger.info(
                "Item standardization method: none (DIR_JSON_MAP exists but is not a "
                "valid grouped mapping: '%s'). "
                "Items will remain unresolved for later train-only fuzzy fallback.",
                mapping_path,
            )

    # ── Standardize items via JSON (partial mapping is fine) ─────────────────
    standardized_train, train_index_map, train_resolved = standardize_items(
        df=dfs[train_sheet_name],
        item_col=item_col,
        mapping_json=mapping_source,
        cfg=cfg,
        remove_spaces=mapping_remove_spaces,
        return_index_map=True,
        return_resolved_mask=True,
    )
    standardized_test, test_index_map, test_resolved = standardize_items(
        df=dfs[test_sheet_name],
        item_col=item_col,
        mapping_json=mapping_source,
        cfg=cfg,
        remove_spaces=mapping_remove_spaces,
        return_index_map=True,
        return_resolved_mask=True,
    )
    standardized_eval, eval_index_map, eval_resolved = standardize_items(
        df=dfs[eval_sheet_name],
        item_col=item_col,
        mapping_json=mapping_source,
        cfg=cfg,
        remove_spaces=mapping_remove_spaces,
        return_index_map=True,
        return_resolved_mask=True,
    )

    split_index_maps = {
        train_sheet_name: train_index_map,
        test_sheet_name: test_index_map,
        eval_sheet_name: eval_index_map,
    }
    item_resolved_by_json = {
        train_sheet_name: train_resolved,
        test_sheet_name: test_resolved,
        eval_sheet_name: eval_resolved,
    }
    standardized: dict[str, pd.DataFrame] = {
        train_sheet_name: standardized_train,
        test_sheet_name: standardized_test,
        eval_sheet_name: standardized_eval,
    }

    standardized_splits = (
        {
            "train": standardized[train_sheet_name].copy(deep=True),
            "test": standardized[test_sheet_name].copy(deep=True),
            "interview": standardized[eval_sheet_name].copy(deep=True),
        }
        if summary_enabled
        else {}
    )

    # ── Stage: post_physician_merge ─────────────────────────────────────────
    # Aggregate physician rows per (Patient, Item) within each sheet separately.
    agg_kwargs: dict[str, Any] = dict(
        patient_col=patient_col or item_col,
        physician_col=physician_col,
        item_col=item_col,
        class_cols=class_cols,
        physician_count=int(cfg.get("physician_count", 2)),
        mismatch_error=bool(cfg.get("mismatch_error", False)),
    )
    if patient_col:
        collapsed_train = _aggregate_physicians(standardized[train_sheet_name], **agg_kwargs)
        collapsed_test = _aggregate_physicians(standardized[test_sheet_name], **agg_kwargs)
        collapsed_eval = _aggregate_physicians(standardized[eval_sheet_name], **agg_kwargs)
    else:
        collapsed_train = standardized[train_sheet_name].copy()
        collapsed_test = standardized[test_sheet_name].copy()
        collapsed_eval = standardized[eval_sheet_name].copy()

    post_physician_splits = (
        {
            "train": collapsed_train.copy(deep=True),
            "test": collapsed_test.copy(deep=True),
            "interview": collapsed_eval.copy(deep=True),
        }
        if summary_enabled
        else {}
    )

    # ── No leakage removal ───────────────────────────────────────────────────
    # Test rows are kept regardless of item overlap with train.
    # Overlap is captured in the `stratum` column added later.
    cleaned_train = collapsed_train
    cleaned_test = collapsed_test
    cleaned_eval = collapsed_eval

    # ── Test / interview merge ───────────────────────────────────────────────
    # LEFT OUTER JOIN cleaned test (left) with cleaned interview on (Patient, Item).
    if patient_col:
        merged_test_interview = _merge_test_interview(
            cleaned_test,
            cleaned_eval,
            patient_col=patient_col,
            item_col=item_col,
            class_cols=class_cols,
        )
    else:
        merged_test_interview = cleaned_test.copy()

    # ── Build final single DataFrame ─────────────────────────────────────────
    # Rename survey (non-interview) class columns to {col}_survey in both splits.
    survey_rename = {col: f"{col}_survey" for col in class_cols}

    train_final = cleaned_train.rename(
        columns={k: v for k, v in survey_rename.items() if k in cleaned_train.columns}
    ).copy()
    train_final["split"] = "train"

    test_final = merged_test_interview.rename(
        columns={k: v for k, v in survey_rename.items() if k in merged_test_interview.columns}
    ).copy()
    test_final["split"] = "test"

    final_df = pd.concat([train_final, test_final], axis=0, ignore_index=True)

    # Drop physician_ids_interview when same_physicians_survey_interview is True (default).
    if bool(cfg.get("same_physicians_survey_interview", True)) and "physician_ids_interview" in final_df.columns:
        final_df = final_df.drop(columns=["physician_ids_interview"])

    # Add stratum column: null for train; 'repeated' / 'novel' for test.
    train_items_normalized = {normalize_for_matching(v) for v in cleaned_train[item_col].dropna()}
    norm_items = final_df[item_col].map(normalize_for_matching)
    test_mask = final_df["split"] == "test"
    in_train_set = norm_items.isin(train_items_normalized)
    stratum: pd.Series = pd.Series(pd.NA, index=final_df.index, dtype=object)
    stratum[test_mask & in_train_set] = "repeated"
    stratum[test_mask & ~in_train_set] = "novel"
    final_df["stratum"] = stratum

    # ── Add item_json_resolved column ────────────────────────────────────────
    # Since JSON mapping is deterministic per item string, all physician rows
    # for a given item share the same resolution status.  Build item-string →
    # resolved lookup per split and map it into final_df.
    train_item_res: dict[str, bool] = dict(
        zip(standardized_train[item_col].values, train_resolved.values)
    )
    test_item_res: dict[str, bool] = dict(
        zip(standardized_test[item_col].values, test_resolved.values)
    )

    def _lookup_resolved(row: pd.Series) -> bool:
        if row["split"] == "train":
            return bool(train_item_res.get(row[item_col], False))
        return bool(test_item_res.get(row[item_col], False))

    final_df["item_json_resolved"] = final_df.apply(_lookup_resolved, axis=1).astype(bool)

    # Keep the summary stage snapshot aligned with final_df by including stratum on test rows.
    test_item_norm = merged_test_interview[item_col].map(normalize_for_matching)
    test_in_train = test_item_norm.isin(train_items_normalized)
    test_stratum: pd.Series = pd.Series(pd.NA, index=merged_test_interview.index, dtype=object)
    test_stratum[test_in_train] = "repeated"
    test_stratum[~test_in_train] = "novel"
    merged_test_interview = merged_test_interview.copy(deep=True)
    merged_test_interview["stratum"] = test_stratum

    summary_kwargs: dict | None = None
    if summary_enabled:
        final_splits: dict[str, pd.DataFrame] = {
            "combined": final_df.copy(deep=True),
        }
        stage_frames = {
            "raw": raw_splits,
            "standardized": standardized_splits,
            "post_physician_merge": post_physician_splits,
            "final": final_splits,
        }
        final_class_columns = list(class_cols)
        metadata: dict[str, Any] = {
            "split_names": {
                "train": train_sheet_name,
                "test": test_sheet_name,
                "interview": eval_sheet_name,
            },
            "item_col": item_col,
            "patient_col": patient_col,
            "physician_col": physician_col,
            "raw_class_cols": list(class_cols),
            "final_class_cols": final_class_columns,
            "stage_class_cols": {
                stage: {
                    split: [
                        c for c in (list(class_cols) if stage != "final" else final_class_columns)
                        if c in df.columns
                    ]
                    for split, df in stage_splits.items()
                }
                for stage, stage_splits in stage_frames.items()
            },
            "row_counts": {
                stage: {k: int(len(v)) for k, v in stage_splits.items()}
                for stage, stage_splits in stage_frames.items()
            },
            "unique_patient_counts": {
                stage: {
                    split: (
                        int(df[patient_col].nunique(dropna=True))
                        if patient_col and patient_col in df.columns
                        else None
                    )
                    for split, df in stage_splits.items()
                }
                for stage, stage_splits in stage_frames.items()
            },
            "unique_item_counts": {
                stage: {
                    split: int(df[item_col].nunique(dropna=True)) if item_col in df.columns else None
                    for split, df in stage_splits.items()
                }
                for stage, stage_splits in stage_frames.items()
            },
            "by_physician_collapse_counts": {
                "train": int(len(standardized[train_sheet_name]) - len(collapsed_train)),
                "test": int(len(standardized[test_sheet_name]) - len(collapsed_test)),
                "interview": int(len(standardized[eval_sheet_name]) - len(collapsed_eval)),
            },
            "duplicate_removal_exists": False,
            "warnings": [],
            "summary_config_used": cfg.get("SUMMARY", {}),
        }
        summary_kwargs = {
            "raw_splits":            raw_splits,
            "standardized_splits":   standardized_splits,
            "post_physician_splits": post_physician_splits,
            "final_splits":          final_splits,
            "metadata":              metadata,
        }

    return PreprocessingResult(
        df                  = final_df,
        split_index_maps    = split_index_maps,
        summary_kwargs      = summary_kwargs,
        item_resolved_by_json = item_resolved_by_json,
    )


def preprocess(
    cfg: dict,
    sheet_names: Iterable[str] | None = None,
    *,
    enforce_raw_test_eval_equality: bool | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the combined DataFrame.

    The returned DataFrame contains a ``split`` column (``'train'`` or
    ``'test'``) and includes physician-aggregated label columns.  Test rows
    carry additional ``{col}_interview`` columns from the LEFT-OUTER-JOIN with
    the interview sheet.

    The old 4-tuple payload ``((X_train, y_train), …, context)`` is gone.
    """
    result = preprocessing(
        cfg, sheet_names=sheet_names,
        enforce_raw_test_eval_equality=enforce_raw_test_eval_equality,
    )

    save_index_mapping_outputs(cfg=cfg, split_index_maps=result.split_index_maps)

    if result.summary_kwargs is not None:
        summary_result = run_summary(cfg=cfg, **result.summary_kwargs)
        summary_logger = logging.getLogger("shared.reporting.input_summary")
        _run_root = Path(cfg.get("DIR_SUMMARY", ".")).parent

        def _rel(path_str: str) -> str:
            try:
                return str(Path(path_str).relative_to(_run_root))
            except ValueError:
                return path_str

        logger.info(
            "Summary generation finished. dir=%s files=%d tables=%d plots=%d removed=%d warnings=%d skipped=%d",
            summary_result.get("summary_dir"),
            len(summary_result.get("generated_files", [])),
            len(summary_result.get("generated_tables", [])),
            len(summary_result.get("generated_plots", [])),
            len(summary_result.get("removed_files", [])),
            len(summary_result.get("warnings", [])),
            len(summary_result.get("skipped", [])),
        )

        stage_stats = summary_result.get("stage_stats", [])
        if isinstance(stage_stats, list):
            for row in stage_stats:
                if isinstance(row, dict):
                    summary_logger.info(
                        "Summary stage stats: stage=%s split=%s rows=%s patients=%s items=%s class_cols=%s",
                        row.get("stage"),
                        row.get("split"),
                        row.get("row_count"),
                        row.get("unique_patients"),
                        row.get("unique_items"),
                        row.get("class_column_count"),
                    )

        summary_logger.info("--- Post-run file manifest ---")
        for path in summary_result.get("generated_files", []):
            summary_logger.info("Summary file generated: %s", _rel(path))
        for path in summary_result.get("removed_files", []):
            summary_logger.info("Summary obsolete file removed: %s", _rel(path))
        for message in summary_result.get("warnings", []):
            summary_logger.warning("Summary warning: %s", message)
        for message in summary_result.get("skipped", []):
            summary_logger.info("Summary output skipped: %s", message)

        warning_count = len(summary_result.get("warnings", []))
        if warning_count:
            logger.warning("Summary emitted %d warning(s); see summary_log for details.", warning_count)

    return result.df
