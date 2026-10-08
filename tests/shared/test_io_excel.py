"""Tests for shared/io/excel.py — Generic Excel loading.

Verifies generic missing-sheet error and successful multi-sheet load behavior
without any dataset-specific assertions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.io.excel import SheetNotFoundError, load_workbook_sheets


@pytest.fixture
def sample_workbook(tmp_path: Path) -> Path:
    """Create a sample Excel workbook with two sheets."""
    path = tmp_path / "sample.xlsx"
    df1 = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"col_c": [4.0, 5.0], "col_d": [True, False]})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="sheet_one", index=False)
        df2.to_excel(writer, sheet_name="sheet_two", index=False)

    return path


class TestSuccessfulLoad:
    """Verify successful workbook loading."""

    def test_load_single_sheet(self, sample_workbook: Path):
        result = load_workbook_sheets(sample_workbook, ["sheet_one"])
        assert "sheet_one" in result
        assert len(result) == 1
        assert isinstance(result["sheet_one"], pd.DataFrame)

    def test_load_multiple_sheets(self, sample_workbook: Path):
        result = load_workbook_sheets(sample_workbook, ["sheet_one", "sheet_two"])
        assert len(result) == 2
        assert "sheet_one" in result
        assert "sheet_two" in result

    def test_dataframe_content(self, sample_workbook: Path):
        result = load_workbook_sheets(sample_workbook, ["sheet_one"])
        df = result["sheet_one"]
        assert list(df.columns) == ["col_a", "col_b"]
        assert len(df) == 3

    def test_dtype_map_applied(self, sample_workbook: Path):
        result = load_workbook_sheets(
            sample_workbook,
            ["sheet_one"],
            dtype_map={"col_a": "string"},
        )
        df = result["sheet_one"]
        assert df["col_a"].dtype == "string"


class TestMissingSheet:
    """Verify error handling for missing sheets."""

    def test_single_missing_sheet(self, sample_workbook: Path):
        with pytest.raises(SheetNotFoundError, match="nonexistent"):
            load_workbook_sheets(sample_workbook, ["nonexistent"])

    def test_one_valid_one_missing(self, sample_workbook: Path):
        with pytest.raises(SheetNotFoundError, match="missing"):
            load_workbook_sheets(sample_workbook, ["sheet_one", "missing"])

    def test_error_message_includes_available(self, sample_workbook: Path):
        with pytest.raises(SheetNotFoundError, match="sheet_one"):
            load_workbook_sheets(sample_workbook, ["not_here"])


class TestEdgeCases:
    """Verify edge case handling."""

    def test_empty_sheet_names_raises(self, sample_workbook: Path):
        with pytest.raises(ValueError, match="[Aa]t least one"):
            load_workbook_sheets(sample_workbook, [])

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_workbook_sheets(tmp_path / "nope.xlsx", ["sheet"])
