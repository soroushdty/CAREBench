from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from shared.preprocessing.preprocessing import preprocessing, preprocess


def _write_dataset(path, train: pd.DataFrame, test: pd.DataFrame, interview: pd.DataFrame):
    with pd.ExcelWriter(path) as writer:
        train.to_excel(writer, sheet_name="train", index=False)
        test.to_excel(writer, sheet_name="test", index=False)
        interview.to_excel(writer, sheet_name="interview", index=False)


def test_preprocessing_with_mapping_json(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "Patient": [1.0, 1.0, 2.0],
            "Physician": [10.0, 11.0, 10.0],
            "Item": ["CBC", "Complete Blood Count", "BMP"],
            "Diagnoses": [1, 1, 0],
            "Medications": [0, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["cbc", "Lipid Panel"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["cbc", "Lipid Panel"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    mapping = {
        "Complete Blood Count": ["CBC", "cbc", "Complete Blood Count", "Complete blood count"],
        "Basic Metabolic Panel": ["BMP"],
        "Lipid Panel": ["Lipid Panel"],
        "ECG": ["ECG"],
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "remove_spaces": True,
    }

    df = preprocess(cfg)

    assert isinstance(df, pd.DataFrame)
    assert "split" in df.columns
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert "complete blood count" in set(train_df["Item"])
    # cbc maps to complete blood count → now retained in test with stratum='repeated'
    assert "complete blood count" in set(test_df["Item"])
    assert "cbc" not in set(train_df["Item"])
    assert "Complete Blood Count" not in set(train_df["Item"])
    assert "basic metabolic panel" in set(train_df["Item"])
    # index_map.json is now written to the reproducibility_artifacts directory.
    combined_index_map_path = summary_dir.parent / "reproducibility_artifacts" / "index_map.json"
    assert combined_index_map_path.exists()
    combined_payload = json.loads(combined_index_map_path.read_text(encoding="utf-8"))
    assert set(combined_payload.keys()) == {"train", "test", "interview"}


def test_preprocessing_with_fuzzy_threshold_leaves_items_unresolved(tmp_path):
    """preprocessing() must NOT apply fuzzy matching.

    When only fuzzy_threshold is configured (no DIR_JSON_MAP), preprocessing()
    must complete without error and leave all items unchanged (unresolved).
    item_json_resolved must be False for every row.  Fuzzy standardization is a
    separate train-only runtime stage that happens after CV fold construction.
    """
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0, 2.0],
            "Physician": [10.0, 10.0],
            "Item": ["ECG", "Electrocardiogram"],
            "Diagnoses": [1, 1],
            "Medications": [0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["ecg", "Lipid Panel"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["ecg", "Lipid Panel"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "fuzzy_threshold": 85,
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    df = preprocess(cfg)

    assert isinstance(df, pd.DataFrame)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert len(train_df) == 2
    assert len(test_df) == 2

    # Items must NOT be fuzzy-merged by preprocessing — they remain as-is.
    train_items = {str(v).strip().casefold() for v in train_df["Item"]}
    assert "ecg" in train_items
    assert "electrocardiogram" in train_items
    # "ECG" and "Electrocardiogram" are NOT merged into one canonical.
    assert len(train_items) == 2

    # item_json_resolved must be False for every row (no JSON was provided).
    assert "item_json_resolved" in df.columns
    assert not df["item_json_resolved"].any(), (
        "No items should be marked JSON-resolved when no DIR_JSON_MAP is configured."
    )

    # Standard structural checks still pass.
    assert "Diagnoses_interview" in test_df.columns
    assert "Medications_interview" in test_df.columns
    assert "Diagnoses_survey" in test_df.columns
    assert "Medications_survey" in test_df.columns
    assert (summary_dir.parent / "reproducibility_artifacts" / "index_map.json").exists()


def test_preprocessing_raises_on_raw_test_eval_mismatch(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1],
            "Medications": [0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["A", "B"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["A", "C"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "fuzzy_threshold": 85,
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    with pytest.raises(ValueError, match="Raw test/eval triplet mismatch"):
        preprocessing(cfg)


def test_preprocessing_allows_raw_test_eval_reordering(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1],
            "Medications": [0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 11.0],
            "Item": ["A", "B"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [3.0, 2.0],
            "Physician": [11.0, 10.0],
            "Item": ["B", "A"],
            "Diagnoses": [0, 1],
            "Medications": [1, 0],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "fuzzy_threshold": 85,
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    df = preprocess(cfg)

    assert isinstance(df, pd.DataFrame)
    assert len(df[df["split"] == "test"]) == 2


def test_preprocessing_still_raises_when_raw_check_flag_is_false(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1],
            "Medications": [0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["A", "B"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["A", "C"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "fuzzy_threshold": 85,
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    with pytest.raises(ValueError, match="Raw test/eval triplet mismatch"):
        preprocessing(cfg, enforce_raw_test_eval_equality=False)


def test_preprocessing_enforced_raw_check_keeps_eval_equal_to_test(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0, 2.0],
            "Physician": [10.0, 10.0],
            "Item": ["ECG", "Electrocardiogram"],
            "Diagnoses": [1, 1],
            "Medications": [0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["ecg", "Lipid Panel"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["ecg", "Lipid Panel"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "fuzzy_threshold": 85,
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "enforce_raw_test_eval_equality": True,
        "physician_count": 1,
    }

    df = preprocess(cfg)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    # Interview data for matching items is merged into *_interview columns.
    assert "Diagnoses_interview" in test_df.columns
    assert "Medications_interview" in test_df.columns
    # Survey class columns carry _survey suffix.
    assert "Diagnoses_survey" in test_df.columns
    assert "Medications_survey" in test_df.columns
    # Both test items are retained (no leakage removal); ecg overlaps with train.
    test_items = {str(v).strip().casefold() for v in test_df["Item"]}
    assert test_items == {"ecg", "lipid panel"}



def test_preprocessing_physician_aggregation_produces_mean_labels(tmp_path):
    """Physician aggregation computes per-(Patient, Item) mean of label columns."""
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    # Two physicians label the same (Patient=1, Item="CBC") differently.
    # Patient 2/BMP also has two physician rows so physician_count=2 is satisfied.
    train = pd.DataFrame(
        {
            "Patient": [1.0, 1.0, 2.0, 2.0],
            "Physician": [10.0, 11.0, 10.0, 11.0],
            "Item": ["CBC", "CBC", "BMP", "BMP"],
            "Diagnoses": [1.0, 0.0, 1.0, 1.0],
            "Medications": [0.0, 0.0, 0.0, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [3.0, 3.0],
            "Physician": [10.0, 11.0],
            "Item": ["BMP", "BMP"],
            "Diagnoses": [1.0, 1.0],
            "Medications": [0.0, 0.0],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 2,
    }

    df = preprocess(cfg)
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    # (Patient=1, "CBC") is collapsed; mean Diagnoses = (1+0)/2 = 0.5
    row = train_df[train_df["Item"] == "cbc"]
    assert len(row) == 1
    assert float(row["Diagnoses_survey"].iloc[0]) == pytest.approx(0.5)
    assert float(row["Medications_survey"].iloc[0]) == pytest.approx(0.0)
    # physician_ids should be a list of int physician IDs
    assert "physician_ids" in train_df.columns
    assert sorted(row["physician_ids"].iloc[0]) == [10, 11]


def test_preprocessing_physician_aggregation_condition_b_raises_error(tmp_path):
    """Condition B: unresolvable mismatch raises ValueError when mismatch_error=True."""
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    # physician_count=2 but Patient 1 has only 1 row → Condition B.
    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1.0],
            "Medications": [0.0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 2.0],
            "Physician": [10.0, 11.0],
            "Item": ["BMP", "BMP"],
            "Diagnoses": [1.0, 1.0],
            "Medications": [0.0, 0.0],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 2,
        "mismatch_error": True,
    }

    with pytest.raises(ValueError, match=r"\[ERROR\] Reference observer count mismatch"):
        preprocessing(cfg)


def test_preprocessing_physician_aggregation_condition_a_deduplicates(tmp_path, caplog):
    """Condition A: resolvable duplicate mismatch deduplicates rows and warns."""
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    # (Patient=1, Item="CBC") has 4 rows: 2 unique rows each repeated twice.
    # physician_count=2, unique count=2 → Condition A.
    train = pd.DataFrame(
        {
            "Patient": [1.0, 1.0, 1.0, 1.0],
            "Physician": [10.0, 11.0, 10.0, 11.0],
            "Item": ["CBC", "CBC", "CBC", "CBC"],
            "Diagnoses": [1.0, 0.0, 1.0, 0.0],
            "Medications": [0.0, 0.0, 0.0, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 2.0],
            "Physician": [10.0, 11.0],
            "Item": ["BMP", "BMP"],
            "Diagnoses": [1.0, 0.0],
            "Medications": [0.0, 0.0],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 2,
    }

    with caplog.at_level(logging.INFO):
        df = preprocess(cfg)

    # Condition A info message must be present with "ROWS DROPPED".
    assert "ROWS DROPPED" in caplog.text
    assert "Reference observer count mismatch" in caplog.text

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    # After dedup the group collapses to 1 aggregated row.
    row = train_df[train_df["Item"] == "cbc"]
    assert len(row) == 1
    # Mean Diagnoses = (1+0)/2 = 0.5 from the 2 unique physician rows.
    assert float(row["Diagnoses_survey"].iloc[0]) == pytest.approx(0.5)


def test_preprocessing_test_interview_merge_requires_raw_triplet_match(tmp_path):
    """A mismatched interview sheet now fails during the raw equality check."""
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1.0],
            "Medications": [0.0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0],
            "Physician": [10.0],
            "Item": ["BMP"],
            "Diagnoses": [1.0],
            "Medications": [0.0],
        }
    )
    # Interview has an extra row that has no match in test.
    interview = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["BMP", "Lipid Panel"],
            "Diagnoses": [1.0, 0.0],
            "Medications": [0.0, 1.0],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"], "Lipid Panel": ["Lipid Panel"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "enforce_raw_test_eval_equality": False,
        "physician_count": 1,
    }

    with pytest.raises(ValueError, match="Raw test/eval triplet mismatch"):
        preprocessing(cfg)



def test_preprocessing_mapping_without_remove_spaces_config_still_runs(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame({"Item": [" CBC "], "Diagnoses": [1], "Medications": [0], "Patient": [1.0], "Physician": [10.0]})
    test = pd.DataFrame({"Item": ["cbc"], "Diagnoses": [1], "Medications": [0], "Patient": [2.0], "Physician": [10.0]})
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"Complete Blood Count": ["cbc", " CBC "]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_SUMMARY": str(summary_dir),
        "DIR_JSON_MAP": str(mapping_path),
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "patient_col": "Patient",
        "physician_col": "Physician",
        "physician_count": 1,
    }

    df = preprocess(cfg)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert set(train_df["Item"]) == {"complete blood count"}
    assert len(test_df) == 1
    assert set(test_df["Item"]) == {"complete blood count"}
    assert (test_df["stratum"] == "repeated").all()


def test_preprocessing_casefold_normalization_applies_to_mapping_and_overlap(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame({"Item": ["straße"], "Diagnoses": [1], "Medications": [0], "Patient": [1.0], "Physician": [10.0]})
    test = pd.DataFrame({"Item": ["Strasse"], "Diagnoses": [1], "Medications": [0], "Patient": [2.0], "Physician": [10.0]})
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"Straße": ["straße", "Strasse"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "patient_col": "Patient",
        "physician_col": "Physician",
        "physician_count": 1,
    }

    df = preprocess(cfg)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    assert set(train_df["Item"]) == {"strasse"}
    assert len(test_df) == 1
    assert set(test_df["Item"]) == {"strasse"}
    assert (test_df["stratum"] == "repeated").all()


def test_preprocessing_run_returns_single_dataframe(tmp_path):
    """run() returns a single DataFrame (not the old 4-tuple)."""
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "Patient": [1.0, 2.0],
            "Physician": [10.0, 11.0],
            "Item": ["CBC", "BMP"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [3.0],
            "Physician": [10.0],
            "Item": ["Lipid Panel"],
            "Diagnoses": [1],
            "Medications": [0],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping_path = data_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps({"cbc": ["CBC"], "bmp": ["BMP"], "lipid panel": ["Lipid Panel"]}),
        encoding="utf-8",
    )

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": "data/mapping.json",
        "DIR_SUMMARY": str(summary_dir),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "PROJECT_ROOT": str(tmp_path),
        "physician_count": 1,
    }

    result = preprocess(cfg)
    assert isinstance(result, pd.DataFrame)
    assert "split" in result.columns
    assert set(result["split"].unique()).issubset({"train", "test"})


def test_stratum_column_values(tmp_path):
    """stratum is null for train rows, 'repeated' for overlapping test rows, 'unique' otherwise."""
    dataset_path = tmp_path / "dataset.xlsx"
    train = pd.DataFrame(
        {
            "Patient": [1.0, 2.0],
            "Physician": [10.0, 10.0],
            "Item": ["CBC", "BMP"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [3.0, 4.0],
            "Physician": [10.0, 10.0],
            "Item": ["Lipid Panel", "CBC"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"], "Lipid Panel": ["Lipid Panel"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    df = preprocess(cfg)
    assert "stratum" in df.columns

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # All train rows have null stratum.
    assert train_df["stratum"].isna().all()
    # Lipid Panel is novel (not in train).
    lipid_rows = test_df[test_df["Item"].str.casefold() == "lipid panel"]
    assert (lipid_rows["stratum"] == "novel").all()
    # CBC overlaps with train → 'repeated'.
    cbc_rows = test_df[test_df["Item"].str.casefold() == "cbc"]
    assert (cbc_rows["stratum"] == "repeated").all()


def test_physician_ids_interview_excluded_by_default(tmp_path):
    """physician_ids_interview is absent from output when same_physicians_surv-intw=True (default)."""
    dataset_path = tmp_path / "dataset.xlsx"
    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1],
            "Medications": [0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0],
            "Physician": [10.0],
            "Item": ["BMP"],
            "Diagnoses": [0],
            "Medications": [1],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
        # same_physicians_surv-intw defaults to True — physician_ids_interview should be absent.
    }

    df = preprocess(cfg)
    assert "physician_ids_interview" not in df.columns
    assert "physician_ids" in df.columns


def test_physician_ids_interview_included_when_config_false(tmp_path):
    """physician_ids_interview is present when same_physicians_surv-intw=False."""
    dataset_path = tmp_path / "dataset.xlsx"
    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [1],
            "Medications": [0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0],
            "Physician": [10.0],
            "Item": ["BMP"],
            "Diagnoses": [0],
            "Medications": [1],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
        "same_physicians_survey_interview": False,
    }

    df = preprocess(cfg)
    test_df = df[df["split"] == "test"]
    assert "physician_ids_interview" in test_df.columns
    # Items in physician_ids_interview must be int.
    ids = test_df["physician_ids_interview"].dropna().iloc[0]
    assert all(isinstance(v, int) for v in ids)


def test_physician_ids_items_are_int(tmp_path):
    """physician_ids list elements are int (not float)."""
    dataset_path = tmp_path / "dataset.xlsx"
    train = pd.DataFrame(
        {
            "Patient": [1.0, 1.0],
            "Physician": [10.0, 11.0],
            "Item": ["CBC", "CBC"],
            "Diagnoses": [1.0, 0.0],
            "Medications": [0.0, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 2.0],
            "Physician": [10.0, 11.0],
            "Item": ["BMP", "BMP"],
            "Diagnoses": [1.0, 1.0],
            "Medications": [0.0, 0.0],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    mapping = {"CBC": ["CBC"], "BMP": ["BMP"]}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(mapping_path),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 2,
    }

    df = preprocess(cfg)
    train_df = df[df["split"] == "train"]
    ids = train_df["physician_ids"].iloc[0]
    assert ids == [10, 11]
    assert all(isinstance(v, int) for v in ids)

