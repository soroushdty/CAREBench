"""Input loading: path resolution, dataset reading, and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse
import pandas as pd

from shared.utils.text_utils import trim_item_value


def _config_root(cfg: dict) -> Path:
    return Path(cfg.get("PROJECT_ROOT", ".")).resolve()


def _resolve_cfg_path(cfg: dict, key: str) -> str | Path:
    if key not in cfg or not cfg[key]:
        raise KeyError(f"Config must include `{key}`.")

    value = cfg[key]
    parsed = urlparse(str(value))
    if parsed.scheme in {"http", "https"}:
        return str(value)

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _config_root(cfg) / candidate
    return candidate.resolve()


def _resolve_input_path(cfg: dict) -> str | Path:
    return _resolve_cfg_path(cfg, "DIR_DATASET")


def _resolve_split_sheet_names(
    cfg: dict,
    sheet_names: Iterable[str] | None,
) -> tuple[str, str, str, list[str]]:
    required_keys = ("TRAIN_SHEET", "TEST_SHEET", "INTERVIEW_SHEET")
    missing = [key for key in required_keys if key not in cfg or not cfg[key]]
    if missing:
        raise KeyError(
            "Config must include non-empty keys: TRAIN_SHEET, TEST_SHEET, INTERVIEW_SHEET. "
            f"Missing: {', '.join(missing)}"
        )

    train_sheet_name = cfg["TRAIN_SHEET"]
    test_sheet_name = cfg["TEST_SHEET"]
    eval_sheet_name = cfg["INTERVIEW_SHEET"]

    if sheet_names is not None:
        selected_sheets = list(sheet_names)
    else:
        selected_sheets = [train_sheet_name, test_sheet_name, eval_sheet_name]

    if not selected_sheets:
        raise ValueError("At least one sheet name must be provided.")

    required_selected = [train_sheet_name, test_sheet_name, eval_sheet_name]
    missing_selected = [name for name in required_selected if name not in selected_sheets]
    if missing_selected:
        raise ValueError(
            "Selected sheets must include configured train/test/interview sheet names. "
            f"Missing: {', '.join(missing_selected)}"
        )

    return train_sheet_name, test_sheet_name, eval_sheet_name, selected_sheets


def _read_dataset(
    dataset_path: str | Path,
    sheet_names: list[str],
    *,
    cfg: dict | None = None,
) -> dict[str, pd.DataFrame]:
    class_list: list[str] = [str(c) for c in (cfg or {}).get("classes") or []]
    dtype_map: dict[str, str] = {cls: "Float64" for cls in class_list}
    for col in ("Item", "Patient", "Physician"):
        if col not in dtype_map:
            dtype_map[col] = "string"
    if cfg:
        item_col = str(cfg.get("item_col") or "Item")
        patient_col = str(cfg.get("patient_col") or "Patient")
        physician_col = str(cfg.get("physician_col") or "Physician")
        dtype_map[item_col] = "string"
        dtype_map[patient_col] = "string"
        dtype_map[physician_col] = "string"
    return {
        # Use pandas nullable dtypes so missing cells remain valid (<NA>)
        # while still enforcing explicit column types for known fields.
        sheet: pd.read_excel(
            dataset_path,
            sheet_name=sheet,
            dtype=dtype_map,
        )
        for sheet in sheet_names
    }


def _assert_raw_test_eval_items_equal(
    *,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    item_col: str,
    patient_col: str | None = None,
    physician_col: str | None = None,
) -> None:
    if item_col not in test_df.columns:
        raise ValueError(f"Column '{item_col}' missing from test sheet.")
    if item_col not in eval_df.columns:
        raise ValueError(f"Column '{item_col}' missing from eval sheet.")

    # Compare the raw survey/interview rows by key triplets rather than row order.
    # This tolerates simple reordering while still rejecting content mismatches.
    if patient_col is not None and physician_col is not None:
        if patient_col not in test_df.columns:
            raise ValueError(f"Column '{patient_col}' missing from test sheet.")
        if patient_col not in eval_df.columns:
            raise ValueError(f"Column '{patient_col}' missing from eval sheet.")
        if physician_col not in test_df.columns:
            raise ValueError(f"Column '{physician_col}' missing from test sheet.")
        if physician_col not in eval_df.columns:
            raise ValueError(f"Column '{physician_col}' missing from eval sheet.")

        key_cols = [patient_col, physician_col, item_col]

        def _normalized_triplets(df: pd.DataFrame) -> pd.DataFrame:
            normalized = df[key_cols].copy(deep=True)
            normalized[item_col] = normalized[item_col].map(trim_item_value)
            return normalized.sort_values(by=key_cols, kind="mergesort").reset_index(drop=True)

        test_triplets = _normalized_triplets(test_df)
        eval_triplets = _normalized_triplets(eval_df)
        if not test_triplets.equals(eval_triplets):
            raise ValueError(
                "Raw test/eval triplet mismatch at load time. "
                "The pipeline requires exact (patient, physician, item) correspondence "
                "between test and eval sheets after accounting for row order."
            )
    else:
        # Fallback for incomplete configs: keep the prior column-wise checks.
        test_items = test_df[item_col].map(trim_item_value).reset_index(drop=True)
        eval_items = eval_df[item_col].map(trim_item_value).reset_index(drop=True)
        if not test_items.equals(eval_items):
            raise ValueError(
                "Raw test/eval item mismatch at load time. "
                "The pipeline requires exact item-wise equality immediately after loading."
            )

        if patient_col is not None:
            if patient_col not in test_df.columns:
                raise ValueError(f"Column '{patient_col}' missing from test sheet.")
            if patient_col not in eval_df.columns:
                raise ValueError(f"Column '{patient_col}' missing from eval sheet.")
            test_patients = test_df[patient_col].reset_index(drop=True)
            eval_patients = eval_df[patient_col].reset_index(drop=True)
            if not test_patients.equals(eval_patients):
                raise ValueError(
                    "Raw test/eval patient mismatch at load time. "
                    "The pipeline requires exact (patient, physician, item) triple correspondence "
                    "between test and eval sheets immediately after loading."
                )

        if physician_col is not None:
            if physician_col not in test_df.columns:
                raise ValueError(f"Column '{physician_col}' missing from test sheet.")
            if physician_col not in eval_df.columns:
                raise ValueError(f"Column '{physician_col}' missing from eval sheet.")
            test_physicians = test_df[physician_col].reset_index(drop=True)
            eval_physicians = eval_df[physician_col].reset_index(drop=True)
            if not test_physicians.equals(eval_physicians):
                raise ValueError(
                    "Raw test/eval physician mismatch at load time. "
                    "The pipeline requires exact (patient, physician, item) triple correspondence "
                    "between test and eval sheets immediately after loading."
                )


def _trim_item_column_all_splits(
    *,
    dfs: Mapping[str, pd.DataFrame],
    item_col: str,
) -> dict[str, pd.DataFrame]:
    trimmed: dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        out = df.copy(deep=True)
        out[item_col] = out[item_col].map(trim_item_value)
        trimmed[name] = out
    return trimmed


def _assert_item_column_exists(
    *,
    dfs: Mapping[str, pd.DataFrame],
    item_col: str,
) -> None:
    missing = [name for name, df in dfs.items() if item_col not in df.columns]
    if missing:
        raise ValueError(
            f"Column '{item_col}' missing from sheets: {', '.join(sorted(missing))}."
        )
