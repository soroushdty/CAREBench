"""Shared fixtures for paired-context adapter tests.

Provides synthetic DataFrames, temp Excel files, and temp JSON summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapters.paired_context.column_map import PairedContextColumnMap
from adapters.paired_context import labels as paired_context_labels


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = paired_context_labels.display_names()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def column_map() -> PairedContextColumnMap:
    """Default paired-context column map."""
    return PairedContextColumnMap()


@pytest.fixture
def synthetic_patients() -> dict[str, dict]:
    """Synthetic patient summaries for context adapter tests."""
    return {
        "1": {
            "summary": "45-year-old patient with hypertension.",
            "medical_history": ["Hypertension", "Mild anxiety"],
            "allergies": ["Penicillin"],
            "medication_history": ["Lisinopril 10 mg"],
            "social_history": ["Employed", "Non-smoker"],
            "labs": ["CBC (normal)"],
            "radiology": ["Chest X-ray (normal)"],
            "procedures": ["EKG (normal)"],
        },
        "2": {
            "summary": "32-year-old patient with type 2 diabetes.",
            "medical_history": ["Type 2 diabetes mellitus"],
            "allergies": ["No known allergies"],
            "medication_history": ["Metformin 500 mg"],
            "social_history": ["Unemployed"],
            "labs": ["HbA1c 8.2%"],
            "radiology": [],
            "procedures": ["Foot exam"],
        },
        "3": {
            "summary": "60-year-old patient with chronic back pain.",
            "medical_history": ["Chronic low back pain"],
            "allergies": ["Sulfa drugs"],
            "medication_history": ["Ibuprofen 400 mg"],
            "social_history": ["Retired"],
            "labs": ["TSH (normal)"],
            "radiology": ["Lumbar spine MRI"],
            "procedures": ["Physical therapy"],
        },
    }


@pytest.fixture
def summaries_file(tmp_path: Path, synthetic_patients: dict) -> Path:
    """Write synthetic patient_summaries.json to a temp file."""
    path = tmp_path / "patient_summaries.json"
    path.write_text(json.dumps(synthetic_patients), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_excel(tmp_path: Path) -> Path:
    """Create a synthetic Excel workbook with train/test/interview sheets."""
    path = tmp_path / "dataset.xlsx"

    patients = [1, 1, 2, 2]
    items = ["item_a", "item_b", "item_a", "item_b"]
    physicians = [1, 2]

    def make_raw_df(label_value: float = 0.0) -> pd.DataFrame:
        rows = []
        for pat, item in zip(patients, items):
            for phys in physicians:
                row = {"Patient": pat, "Physician": phys, "Item": item}
                for cls in CLASSES:
                    row[cls] = label_value
                rows.append(row)
        return pd.DataFrame(rows)

    train_df = make_raw_df(0.0)
    test_df = make_raw_df(0.0)
    interview_df = make_raw_df(1.0)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="train", index=False)
        test_df.to_excel(writer, sheet_name="test", index=False)
        interview_df.to_excel(writer, sheet_name="interview", index=False)

    return path
