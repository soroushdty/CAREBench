"""
Data loaders for the analysis bundle pipeline.

Loads physician consensus from dataset.xlsx (via DatasetLoader) and LLM score
CSVs from three conditions or a combined long-format file.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical category names and Excel → snake_case mapping
# ---------------------------------------------------------------------------

# Single source of truth: import from schema_validator
from tracks.reasoning.schema_validator import (
    DISPLAY_TO_CANONICAL,
    REQUIRED_KEYS as CANONICAL_CATEGORIES,
)

# Standard Excel column names → canonical snake_case (re-exported alias)
_EXCEL_TO_CANONICAL: dict[str, str] = DISPLAY_TO_CANONICAL

# Standard Excel class column names (for DatasetLoader cfg)
_DEFAULT_EXCEL_CLASSES: list[str] = list(_EXCEL_TO_CANONICAL.keys())


# ---------------------------------------------------------------------------
# load_physician_consensus
# ---------------------------------------------------------------------------


def load_physician_consensus(
    dataset_path: str,
    patient_col: str = "Patient",
    physician_col: str = "Physician",
    item_col: str = "Item",
    train_sheet: str = "train",
    survey_sheet: str = "test",
    interview_sheet: str = "interview",
    excel_classes: Optional[List[str]] = None,
    class_definitions: Optional[dict] = None,
) -> pd.DataFrame:
    """Load physician survey and interview consensus from dataset.xlsx.

    Uses :class:`assay.dataset_loader.DatasetLoader` for physician-pair
    aggregation and triplet-correspondence verification.

    Parameters
    ----------
    dataset_path:
        Path to dataset.xlsx.
    patient_col:
        Column name for patient IDs in the Excel file.
    physician_col:
        Column name for physician IDs.
    item_col:
        Column name for item text.
    train_sheet:
        Sheet name for the training/reference sheet (used by DatasetLoader
        but not included in output).
    survey_sheet:
        Sheet name for the context-free survey phase (test sheet).
    interview_sheet:
        Sheet name for the context-aware interview phase.
    excel_classes:
        List of category column names as they appear in the Excel file.
        Defaults to the standard 10-class names (e.g. "Behavioral health").
    class_definitions:
        Optional keys per class (the assay config's ``data.class_definitions``),
        as accepted by :meth:`shared.label_space.LabelSpace.from_config`.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per patient × item × category.
        Columns: patient_id, item_text, item_id, category,
        physician_survey_consensus, physician_interview_consensus,
        delta_physician.
    """
    from tracks.reasoning.dataset_loader import DatasetLoader

    classes = excel_classes if excel_classes is not None else _DEFAULT_EXCEL_CLASSES

    cfg: dict = {
        "data": {
            "dataset_path": str(dataset_path),
            "patient_col": patient_col,
            "physician_col": physician_col,
            "item_col": item_col,
            "classes": classes,
            "class_definitions": class_definitions,
            "train_sheet": train_sheet,
            "test_sheet": survey_sheet,
            "interview_sheet": interview_sheet,
        }
    }

    loader = DatasetLoader(cfg, dataset_path=dataset_path)
    paired = loader.load()

    # Category keys from the dataset's label space
    canonical_cats = paired.canonical_category_names

    rows: list[dict] = []
    for i, (pid, itext) in enumerate(zip(paired.patient_ids, paired.item_texts)):
        for j, cat in enumerate(canonical_cats):
            rows.append(
                {
                    "patient_id": str(pid),
                    "item_text": str(itext),
                    "item_id": i,
                    "category": cat,
                    "physician_survey_consensus": float(
                        paired.survey_consensus[i, j]
                    ),
                    "physician_interview_consensus": float(
                        paired.interview_consensus[i, j]
                    ),
                    "delta_physician": float(paired.delta_physician[i, j]),
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "Loaded physician consensus: %d patients, %d items, %d categories, "
        "%d total rows",
        df["patient_id"].nunique(),
        df.groupby("patient_id")["item_text"].nunique().mean(),
        df["category"].nunique(),
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# load_llm_scores
# ---------------------------------------------------------------------------

_CONDITION_LONG = "context_free"
_CONDITION_CORRECT = "correct_context"
_CONDITION_SHUFFLED = "shuffled_context"
_ALL_CONDITIONS = [_CONDITION_LONG, _CONDITION_CORRECT, _CONDITION_SHUFFLED]


def load_llm_scores(
    context_free: Optional[str] = None,
    correct_context: Optional[str] = None,
    shuffled_context: Optional[str] = None,
    llm_scores: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load LLM scores from individual condition CSVs or a combined long-format file.

    Parameters
    ----------
    context_free:
        Path to context_free_scores.csv.
    correct_context:
        Path to correct_context_scores.csv.
    shuffled_context:
        Path to shuffled_context_scores.csv.
    llm_scores:
        Path to a combined long-format CSV with a ``condition`` column.
        If provided, individual files are ignored (with a warning if also given).
    categories:
        List of canonical category column names.  Defaults to
        :data:`CANONICAL_CATEGORIES`.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        model, patient_id, item_text, condition, {category columns}.
        ``condition`` values are one of: context_free, correct_context,
        shuffled_context.
    """
    cats = categories if categories is not None else CANONICAL_CATEGORIES

    if llm_scores is not None:
        if any(p is not None for p in [context_free, correct_context, shuffled_context]):
            warnings.warn(
                "--llm-scores supplied alongside individual condition files. "
                "Using the long-format file and ignoring individual files.",
                UserWarning,
                stacklevel=2,
            )
        df = _read_long_format(llm_scores, cats)
    else:
        if any(p is None for p in [context_free, correct_context, shuffled_context]):
            missing = [
                name
                for name, p in [
                    ("context_free", context_free),
                    ("correct_context", correct_context),
                    ("shuffled_context", shuffled_context),
                ]
                if p is None
            ]
            raise ValueError(
                f"Missing score files for conditions: {missing}. "
                "Provide all three files or use --llm-scores for long format."
            )
        df = _read_three_csvs(
            context_free, correct_context, shuffled_context, cats
        )

    logger.info(
        "Loaded LLM scores: %d models, %d conditions, %d rows",
        df["model"].nunique(),
        df["condition"].nunique(),
        len(df),
    )
    return df


def _normalise_model_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'model' column exists, normalising from 'model_id' if needed."""
    if "model" not in df.columns:
        if "model_id" in df.columns:
            df = df.rename(columns={"model_id": "model"})
        else:
            df = df.copy()
            df["model"] = "unknown"
    return df


def _normalise_patient_item_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise patient/item column names."""
    rename: dict[str, str] = {}
    if "patient_id" not in df.columns:
        for alt in ("Patient", "patient", "PatientID"):
            if alt in df.columns:
                rename[alt] = "patient_id"
                break
    if "item_text" not in df.columns:
        for alt in ("Item", "item", "item_id"):
            if alt in df.columns:
                rename[alt] = "item_text"
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def _coerce_category_cols(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    """Ensure category columns are present and numeric."""
    for cat in cats:
        if cat not in df.columns:
            # Try case-insensitive match
            lower_map = {c.lower(): c for c in df.columns}
            canon_lower = cat.lower()
            if canon_lower in lower_map:
                df = df.rename(columns={lower_map[canon_lower]: cat})
        if cat in df.columns:
            df[cat] = pd.to_numeric(df[cat], errors="coerce")
    return df


def _read_long_format(path: str, cats: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalise_model_col(df)
    df = _normalise_patient_item_cols(df)
    df = _coerce_category_cols(df, cats)
    df["patient_id"] = df["patient_id"].astype(str)
    if "condition" not in df.columns:
        raise ValueError(
            f"Long-format file '{path}' must have a 'condition' column."
        )
    return df[["model", "patient_id", "item_text", "condition"] + cats].copy()


def _read_three_csvs(
    cf_path: str,
    cc_path: str,
    sc_path: str,
    cats: list[str],
) -> pd.DataFrame:
    parts = []
    for path, cond in [
        (cf_path, _CONDITION_LONG),
        (cc_path, _CONDITION_CORRECT),
        (sc_path, _CONDITION_SHUFFLED),
    ]:
        df = pd.read_csv(path)
        df = _normalise_model_col(df)
        df = _normalise_patient_item_cols(df)
        df = _coerce_category_cols(df, cats)
        df["patient_id"] = df["patient_id"].astype(str)
        df["condition"] = cond
        # Drop extra columns not needed
        keep = ["model", "patient_id", "item_text", "condition"] + cats
        keep = [c for c in keep if c in df.columns]
        parts.append(df[keep])

    return pd.concat(parts, ignore_index=True)
