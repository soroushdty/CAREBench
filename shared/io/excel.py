"""Generic Excel workbook loading.

This module provides dataset-neutral workbook reading. It does NOT reference
any dataset-specific column names, sheet names, label names, or observer
counts. Dataset-specific validation belongs in the relevant adapter.

Canonical path: shared/io/excel.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class SheetNotFoundError(Exception):
    """Raised when one or more requested sheets are missing from the workbook."""


def load_workbook_sheets(
    path: str | Path,
    sheet_names: list[str],
    *,
    dtype_map: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read specified sheets from an Excel workbook and return raw DataFrames.

    Parameters
    ----------
    path : str | Path
        Path to the Excel file (.xlsx / .xls).
    sheet_names : list[str]
        Sheet names to read. At least one must be provided.
    dtype_map : dict[str, str] | None
        Optional column → dtype mapping passed to ``pd.read_excel``.
        If ``None``, pandas infers types.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of sheet name → DataFrame with raw data.

    Raises
    ------
    SheetNotFoundError
        If any requested sheet is not present in the workbook.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If ``sheet_names`` is empty.
    """
    if not sheet_names:
        raise ValueError("At least one sheet name must be provided.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    # Discover available sheets without reading full data
    xl = pd.ExcelFile(path)
    available = set(xl.sheet_names)
    missing = [s for s in sheet_names if s not in available]
    if missing:
        raise SheetNotFoundError(
            f"Requested sheet(s) not found in workbook '{path.name}': "
            f"{', '.join(sorted(missing))}. "
            f"Available sheets: {', '.join(sorted(available))}"
        )

    read_kwargs: dict[str, Any] = {}
    if dtype_map is not None:
        read_kwargs["dtype"] = dtype_map

    results: dict[str, pd.DataFrame] = {}
    for sheet in sheet_names:
        results[sheet] = pd.read_excel(xl, sheet_name=sheet, **read_kwargs)

    xl.close()
    return results
