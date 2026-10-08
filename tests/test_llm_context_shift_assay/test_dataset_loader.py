"""
Tests for tracks/reasoning/dataset_loader.py — Pairing Verification.

All tests use synthetic DataFrames; no real dataset.xlsx is required.
The tests patch PairedContextDatasetAdapter.load_sheets so the Excel file is never read.

Run with:
    pytest tests/test_llm_context_shift_assay/test_dataset_loader.py --noconftest
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter
from tracks.reasoning.dataset_loader import DatasetLoader, PairedDataset, ValidationError


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

CLASSES = [
    "behavioral_health",
    "diagnoses",
    "disabilities",
    "infectious_diseases",
    "genetics",
    "medications",
    "sexual_reproductive_health",
    "social_determinants_of_health",
    "violence",
    "other",
]

MINIMAL_CFG = {
    "data": {
        "dataset_path": "examples/synthetic/dataset.xlsx",
        "patient_summaries_path": "examples/synthetic/patient_summaries.json",
        "train_sheet": "train",
        "test_sheet": "test",
        "interview_sheet": "interview",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "item_col": "Item",
        "classes": CLASSES,
    }
}


def _make_raw_df(
    patients: list[int],
    physicians: list[int],
    items: list[str],
    label_value: float = 0.0,
) -> pd.DataFrame:
    """Build a raw (pre-aggregation) DataFrame with two physicians per patient-item pair."""
    rows = []
    for pat, item in zip(patients, items):
        for phys in physicians:
            row = {"Patient": pat, "Physician": phys, "Item": item}
            for cls in CLASSES:
                row[cls] = label_value
            rows.append(row)
    return pd.DataFrame(rows)


def _make_loader() -> DatasetLoader:
    return DatasetLoader(MINIMAL_CFG, dataset_path="examples/synthetic/dataset.xlsx")


# ---------------------------------------------------------------------------
# 13.2  test_triplet_correspondence
# ---------------------------------------------------------------------------


class TestTripletCorrespondence:
    """triplets must match between test and interview sheets."""

    def test_matching_triplets_do_not_raise(self):
        """When test and interview have identical triplets, no error is raised."""
        patients = [1, 1, 2, 2]
        items = ["item_a", "item_b", "item_a", "item_b"]
        physicians = [1, 2]

        test_df = _make_raw_df(patients, physicians, items, label_value=0.0)
        interview_df = _make_raw_df(patients, physicians, items, label_value=1.0)

        loader = _make_loader()
        # Should not raise
        loader._verify_triplet_correspondence(test_df, interview_df)

    def test_mismatched_triplets_raise_validation_error(self):
        """When interview has an extra triplet not in test, ValidationError is raised."""
        patients_test = [1, 1]
        items_test = ["item_a", "item_b"]
        physicians = [1, 2]

        # Interview has an extra patient-item pair
        patients_interview = [1, 1, 2]
        items_interview = ["item_a", "item_b", "item_a"]

        test_df = _make_raw_df(patients_test, physicians, items_test)
        interview_df = _make_raw_df(patients_interview, physicians, items_interview)

        loader = _make_loader()
        with pytest.raises(ValidationError):
            loader._verify_triplet_correspondence(test_df, interview_df)

    def test_missing_triplet_in_interview_raises_validation_error(self):
        """When test has a triplet absent from interview, ValidationError is raised."""
        patients = [1, 1, 2]
        items = ["item_a", "item_b", "item_a"]
        physicians = [1, 2]

        test_df = _make_raw_df(patients, physicians, items)
        # Interview is missing patient 2 / item_a
        interview_df = _make_raw_df([1, 1], physicians, ["item_a", "item_b"])

        loader = _make_loader()
        with pytest.raises(ValidationError):
            loader._verify_triplet_correspondence(test_df, interview_df)

    def test_full_load_with_matching_triplets(self):
        """End-to-end: DatasetLoader.load() succeeds when triplets match."""
        patients = [1, 1, 2, 2]
        items = ["item_a", "item_b", "item_a", "item_b"]
        physicians = [1, 2]

        test_df = _make_raw_df(patients, physicians, items, label_value=0.0)
        interview_df = _make_raw_df(patients, physicians, items, label_value=1.0)

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            result = loader.load()

        assert isinstance(result, PairedDataset)

    def test_full_load_with_mismatched_triplets_raises(self):
        """End-to-end: DatasetLoader.load() raises ValidationError on mismatch."""
        patients_test = [1, 1]
        items_test = ["item_a", "item_b"]
        physicians = [1, 2]

        test_df = _make_raw_df(patients_test, physicians, items_test)
        # Interview has an extra pair
        interview_df = _make_raw_df([1, 1, 2], physicians, ["item_a", "item_b", "item_a"])

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            with pytest.raises(ValidationError):
                loader.load()


# ---------------------------------------------------------------------------
# 13.3  test_consensus_shapes
# ---------------------------------------------------------------------------


class TestConsensusShapes:
    """survey_consensus and interview_consensus must have the same shape."""

    def _build_paired_dataset(
        self,
        n_patients: int = 3,
        n_items_per_patient: int = 2,
        survey_value: float = 0.0,
        interview_value: float = 1.0,
    ) -> PairedDataset:
        patients = list(range(1, n_patients + 1))
        items = [f"item_{i}" for i in range(n_items_per_patient)]
        physicians = [1, 2]

        all_patients = [p for p in patients for _ in items]
        all_items = items * n_patients

        test_df = _make_raw_df(all_patients, physicians, all_items, label_value=survey_value)
        interview_df = _make_raw_df(all_patients, physicians, all_items, label_value=interview_value)

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            return loader.load()

    def test_survey_and_interview_have_same_shape(self):
        """survey_consensus.shape == interview_consensus.shape."""
        ds = self._build_paired_dataset(n_patients=3, n_items_per_patient=2)
        assert ds.survey_consensus.shape == ds.interview_consensus.shape

    def test_consensus_has_10_columns(self):
        """Both consensus arrays have exactly 10 category columns."""
        ds = self._build_paired_dataset(n_patients=3, n_items_per_patient=2)
        assert ds.survey_consensus.shape[1] == 10
        assert ds.interview_consensus.shape[1] == 10

    def test_consensus_shape_matches_n_pairs(self):
        """Number of rows equals n_patients × n_items_per_patient."""
        n_patients = 4
        n_items = 3
        ds = self._build_paired_dataset(n_patients=n_patients, n_items_per_patient=n_items)
        expected_rows = n_patients * n_items
        assert ds.survey_consensus.shape == (expected_rows, 10)
        assert ds.interview_consensus.shape == (expected_rows, 10)

    def test_delta_physician_shape_matches_consensus(self):
        """delta_physician.shape == survey_consensus.shape."""
        ds = self._build_paired_dataset(n_patients=2, n_items_per_patient=3)
        assert ds.delta_physician.shape == ds.survey_consensus.shape


# ---------------------------------------------------------------------------
# 13.4  test_delta_physician_computation
# ---------------------------------------------------------------------------


class TestDeltaPhysicianComputation:
    """Delta_Physician = interview_consensus - survey_consensus."""

    def _load_with_known_values(
        self,
        survey_value: float,
        interview_value: float,
    ) -> PairedDataset:
        patients = [1, 2]
        items = ["item_a", "item_b"]
        physicians = [1, 2]

        all_patients = [p for p in patients for _ in items]
        all_items = items * len(patients)

        test_df = _make_raw_df(all_patients, physicians, all_items, label_value=survey_value)
        interview_df = _make_raw_df(all_patients, physicians, all_items, label_value=interview_value)

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            return loader.load()

    def test_delta_is_interview_minus_survey(self):
        """delta_physician == interview_consensus - survey_consensus exactly."""
        survey_val = 0.25
        interview_val = 0.75
        ds = self._load_with_known_values(survey_val, interview_val)

        expected_delta = ds.interview_consensus - ds.survey_consensus
        np.testing.assert_array_almost_equal(ds.delta_physician, expected_delta)

    def test_delta_zero_when_values_equal(self):
        """When survey and interview labels are identical, delta is all zeros."""
        ds = self._load_with_known_values(0.5, 0.5)
        np.testing.assert_array_equal(ds.delta_physician, np.zeros_like(ds.delta_physician))

    def test_delta_positive_when_interview_greater(self):
        """Delta is positive when interview > survey."""
        ds = self._load_with_known_values(0.0, 1.0)
        assert np.all(ds.delta_physician > 0)

    def test_delta_negative_when_survey_greater(self):
        """Delta is negative when survey > interview."""
        ds = self._load_with_known_values(1.0, 0.0)
        assert np.all(ds.delta_physician < 0)

    def test_delta_exact_known_values(self):
        """Verify exact numeric delta with known survey=0.0, interview=1.0."""
        ds = self._load_with_known_values(0.0, 1.0)
        # Each physician labels 0 or 1; mean of [0, 0] = 0.0, mean of [1, 1] = 1.0
        # delta = 1.0 - 0.0 = 1.0 for every cell
        np.testing.assert_array_almost_equal(
            ds.delta_physician,
            np.ones_like(ds.delta_physician),
        )

    def test_delta_with_mixed_physician_labels(self):
        """Verify delta when physicians disagree: mean([0,1])=0.5 for both sheets."""
        patients = [1]
        items = ["item_a"]
        physicians = [1, 2]

        # Physician 1 labels 0, Physician 2 labels 1 — for both test and interview
        rows_test = []
        rows_interview = []
        for phys, label in zip(physicians, [0, 1]):
            row = {"Patient": 1, "Physician": phys, "Item": "item_a"}
            for cls in CLASSES:
                row[cls] = label
            rows_test.append(dict(row))
            rows_interview.append(dict(row))

        test_df = pd.DataFrame(rows_test)
        interview_df = pd.DataFrame(rows_interview)

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            ds = loader.load()

        # Both consensus values are 0.5, so delta = 0.0
        np.testing.assert_array_almost_equal(
            ds.delta_physician,
            np.zeros_like(ds.delta_physician),
        )


# ---------------------------------------------------------------------------
# 13.5  test_aggregation_range
# ---------------------------------------------------------------------------


class TestAggregationRange:
    """physician pair aggregation produces values in [0, 1]."""

    def _load_with_binary_labels(
        self,
        survey_labels: list[int],
        interview_labels: list[int],
    ) -> PairedDataset:
        """Build a dataset where two physicians have the given binary labels."""
        assert len(survey_labels) == 2
        assert len(interview_labels) == 2

        rows_test = []
        rows_interview = []
        for phys_idx, (s_label, i_label) in enumerate(
            zip(survey_labels, interview_labels), start=1
        ):
            row_test = {"Patient": 1, "Physician": phys_idx, "Item": "item_a"}
            row_interview = {"Patient": 1, "Physician": phys_idx, "Item": "item_a"}
            for cls in CLASSES:
                row_test[cls] = s_label
                row_interview[cls] = i_label
            rows_test.append(row_test)
            rows_interview.append(row_interview)

        test_df = pd.DataFrame(rows_test)
        interview_df = pd.DataFrame(rows_interview)

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            return loader.load()

    @pytest.mark.parametrize(
        "survey_labels,interview_labels",
        [
            ([0, 0], [0, 0]),  # both 0 → consensus 0.0
            ([1, 1], [1, 1]),  # both 1 → consensus 1.0
            ([0, 1], [0, 1]),  # disagree → consensus 0.5
            ([1, 0], [1, 0]),  # disagree (reversed) → consensus 0.5
            ([0, 0], [1, 1]),  # survey=0.0, interview=1.0
            ([1, 1], [0, 0]),  # survey=1.0, interview=0.0
        ],
    )
    def test_consensus_values_in_unit_interval(self, survey_labels, interview_labels):
        """All survey_consensus and interview_consensus values are in [0, 1]."""
        ds = self._load_with_binary_labels(survey_labels, interview_labels)
        assert np.all(ds.survey_consensus >= 0.0)
        assert np.all(ds.survey_consensus <= 1.0)
        assert np.all(ds.interview_consensus >= 0.0)
        assert np.all(ds.interview_consensus <= 1.0)

    def test_consensus_values_in_unit_interval_multi_patient(self):
        """Aggregation stays in [0, 1] across multiple patients and items."""
        patients = [1, 1, 2, 2, 3, 3]
        items = ["item_a", "item_b"] * 3
        physicians = [1, 2]

        # Alternate labels: physician 1 always 0, physician 2 always 1
        rows_test = []
        rows_interview = []
        for pat, item in zip(patients, items):
            for phys, label in zip(physicians, [0, 1]):
                row = {"Patient": pat, "Physician": phys, "Item": item}
                for cls in CLASSES:
                    row[cls] = label
                rows_test.append(dict(row))
                rows_interview.append(dict(row))

        test_df = pd.DataFrame(rows_test)
        interview_df = pd.DataFrame(rows_interview)

        sheets = {
            "train": test_df.copy(),
            "test": test_df,
            "interview": interview_df,
        }

        loader = _make_loader()
        with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
            ds = loader.load()

        assert np.all(ds.survey_consensus >= 0.0)
        assert np.all(ds.survey_consensus <= 1.0)
        assert np.all(ds.interview_consensus >= 0.0)
        assert np.all(ds.interview_consensus <= 1.0)

    def test_survey_consensus_both_zero_gives_zero(self):
        """When both physicians label 0, consensus is exactly 0.0."""
        ds = self._load_with_binary_labels([0, 0], [1, 1])
        np.testing.assert_array_almost_equal(
            ds.survey_consensus, np.zeros_like(ds.survey_consensus)
        )

    def test_interview_consensus_both_one_gives_one(self):
        """When both physicians label 1, consensus is exactly 1.0."""
        ds = self._load_with_binary_labels([0, 0], [1, 1])
        np.testing.assert_array_almost_equal(
            ds.interview_consensus, np.ones_like(ds.interview_consensus)
        )

    def test_consensus_disagree_gives_half(self):
        """When physicians disagree (0 and 1), consensus is exactly 0.5."""
        ds = self._load_with_binary_labels([0, 1], [0, 1])
        np.testing.assert_array_almost_equal(
            ds.survey_consensus, np.full_like(ds.survey_consensus, 0.5)
        )
        np.testing.assert_array_almost_equal(
            ds.interview_consensus, np.full_like(ds.interview_consensus, 0.5)
        )


# ---------------------------------------------------------------------------
# Patient summaries are optional for reference-label loading
# ---------------------------------------------------------------------------


def test_load_without_patient_summaries_path():
    """Reference labels load without a patient summaries file (e.g. post-hoc analysis)."""
    cfg = {"data": {k: v for k, v in MINIMAL_CFG["data"].items() if k != "patient_summaries_path"}}
    patients = [1, 1, 2, 2]
    items = ["item_a", "item_b", "item_a", "item_b"]
    test_df = _make_raw_df(patients, [1, 2], items, label_value=0.0)
    interview_df = _make_raw_df(patients, [1, 2], items, label_value=1.0)
    sheets = {"train": test_df.copy(), "test": test_df, "interview": interview_df}

    loader = DatasetLoader(cfg)
    with patch.object(PairedContextDatasetAdapter, "load_sheets", return_value=sheets):
        result = loader.load()

    assert result.survey_consensus.shape == (4, len(CLASSES))
    assert loader.manifest()["context"] is None
