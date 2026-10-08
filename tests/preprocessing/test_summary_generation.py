from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from shared.preprocessing.preprocessing import preprocess as preprocessing
from shared.reporting.input_summary.utils.logging_utils import log_high_level_event, setup_summary_logger


def _write_dataset(path, train: pd.DataFrame, test: pd.DataFrame, interview: pd.DataFrame):
    with pd.ExcelWriter(path) as writer:
        train.to_excel(writer, sheet_name="train", index=False)
        test.to_excel(writer, sheet_name="test", index=False)
        interview.to_excel(writer, sheet_name="interview", index=False)


def _close_summary_logger():
    """Helper to close all handlers on the summary logger."""
    logger = logging.getLogger("shared.reporting.input_summary")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_preprocessing_generates_summary_artifacts_when_enabled(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0, 1.0, 2.0],
            "Physician": [10.0, 11.0, 10.0],
            "Item": ["CBC", "cbc", "BMP"],
            "Diagnoses": [1, 0, 0],
            "Medications": [0, 1, 1],
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
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_SUMMARY": str(summary_dir),
        "ENABLE_SUMMARY": True,
        "SUMMARY": {
            "outputs": {
                "item_frequency_raw_csv": True,
                "item_frequency_standardized_csv": True,
                "stage_audit_csv": True,
                "final_combined_csv": True,
            }
        },
        "fuzzy_threshold": 85,
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    df = preprocessing(cfg)

    assert isinstance(df, pd.DataFrame)
    assert "split" in df.columns
    assert (summary_dir / "audits" / "stage_audit.csv").exists()
    reproducibility_dir = summary_dir.parent / "reproducibility_artifacts"
    assert (reproducibility_dir / "summary_index.json").exists()
    assert (reproducibility_dir / "summary_config.json").exists()
    assert (summary_dir / "preprocessing_stages" / "raw" / "train" / "item_frequency_raw.csv").exists()
    assert (summary_dir / "preprocessing_stages" / "standardized" / "train" / "item_frequency_standardized.csv").exists()
    assert (summary_dir / "final" / "final_combined.csv").exists()
    # Consolidated missingness workbook (replaced 13 individual xlsx files).
    assert (summary_dir / "audits" / "missingness_summary.xlsx").exists()
    stage_audit = pd.read_csv(summary_dir / "audits" / "stage_audit.csv")
    assert {"stage", "split", "row_count"}.issubset(stage_audit.columns)
    assert (stage_audit["stage"] == "raw").any()
    runtime_cfg = json.loads((reproducibility_dir / "summary_config.json").read_text(encoding="utf-8"))
    assert "runtime_summary_config" in runtime_cfg
    assert "provenance" in runtime_cfg
    assert runtime_cfg["runtime_summary_config"]["outputs"]["missingness_summary_xlsx"] is True
    raw_item_freq = pd.read_csv(summary_dir / "preprocessing_stages" / "raw" / "train" / "item_frequency_raw.csv")
    assert {"item", "count", "percentage"}.issubset(raw_item_freq.columns)
    assert raw_item_freq["count"].sum() == len(train)


def test_preprocessing_generates_delta_summary_outputs(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0],
            "Physician": [10.0],
            "Item": ["CBC"],
            "Diagnoses": [0.0],
            "Medications": [1.0],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["CBC", "Lipid Panel"],
            "Diagnoses": [0.0, 1.0],
            "Medications": [1.0, 0.0],
        }
    )
    interview = pd.DataFrame(
        {
            "Patient": [2.0, 3.0],
            "Physician": [10.0, 10.0],
            "Item": ["CBC", "Lipid Panel"],
            "Diagnoses": [1.0, 0.0],
            "Medications": [0.0, 0.0],
        }
    )
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_SUMMARY": str(summary_dir),
        "ENABLE_SUMMARY": True,
        "SUMMARY": {
            "outputs": {
                "class_distribution_csv": False,
                "class_distribution_plot": False,
                "class_cooccurrence_csv": False,
                "class_cooccurrence_plot": False,
            }
        },
        "fuzzy_threshold": 85,
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 1,
    }

    _ = preprocessing(cfg)

    delta_dir = summary_dir / "final" / "delta"
    pct_per_class_path = delta_dir / "percentage_per_class.csv"
    count_per_class_path = delta_dir / "count_per_class.csv"
    pct_per_stratum_path = delta_dir / "percentage_per_stratum.csv"
    count_per_stratum_path = delta_dir / "count_per_stratum.csv"

    assert pct_per_class_path.exists()
    assert count_per_class_path.exists()
    assert pct_per_stratum_path.exists()
    assert count_per_stratum_path.exists()

    expected_class_cols = [
        "class",
        "positive_delta",
        "negative_delta",
        "zero_delta",
        "delta_-1",
        "delta_+1",
        "delta_-0.5",
        "delta_+0.5",
    ]
    expected_stratum_cols = [
        "class",
        "stratum",
        "positive_delta",
        "negative_delta",
        "zero_delta",
        "delta_-1",
        "delta_+1",
        "delta_-0.5",
        "delta_+0.5",
    ]

    pct_per_class = pd.read_csv(pct_per_class_path)
    count_per_class = pd.read_csv(count_per_class_path)
    pct_per_stratum = pd.read_csv(pct_per_stratum_path)
    count_per_stratum = pd.read_csv(count_per_stratum_path)

    assert list(pct_per_class.columns) == expected_class_cols
    assert list(count_per_class.columns) == expected_class_cols
    assert list(pct_per_stratum.columns) == expected_stratum_cols
    assert list(count_per_stratum.columns) == expected_stratum_cols

    assert pct_per_class["class"].tolist()[-1] == "Total"
    assert count_per_class["class"].tolist()[-1] == "Total"
    assert set(pct_per_stratum[pct_per_stratum["class"] == "Total"]["stratum"]) == {
        "repeated",
        "novel",
    }
    assert set(count_per_stratum[count_per_stratum["class"] == "Total"]["stratum"]) == {
        "repeated",
        "novel",
    }

    diag_counts = count_per_class[count_per_class["class"] == "Diagnoses"].iloc[0]
    med_counts = count_per_class[count_per_class["class"] == "Medications"].iloc[0]
    total_counts = count_per_class[count_per_class["class"] == "Total"].iloc[0]

    assert int(diag_counts["delta_+1"]) == 1
    assert int(diag_counts["delta_-1"]) == 1
    assert int(med_counts["delta_-1"]) == 1
    assert int(med_counts["zero_delta"]) == 1

    assert int(total_counts["positive_delta"]) == 1
    assert int(total_counts["negative_delta"]) == 2
    assert int(total_counts["zero_delta"]) == 1

    repeated_diag = count_per_stratum[
        (count_per_stratum["class"] == "Diagnoses") & (count_per_stratum["stratum"] == "repeated")
    ].iloc[0]
    unique_diag = count_per_stratum[
        (count_per_stratum["class"] == "Diagnoses") & (count_per_stratum["stratum"] == "novel")
    ].iloc[0]
    assert int(repeated_diag["delta_+1"]) == 1
    assert int(unique_diag["delta_-1"]) == 1


def test_preprocessing_runtime_summary_config_artifact_toggle(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame(
        {
            "Patient": [1.0, 1.0],
            "Physician": [10.0, 11.0],
            "Item": ["CBC", "CBC"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [2.0, 2.0],
            "Physician": [10.0, 11.0],
            "Item": ["BMP", "BMP"],
            "Diagnoses": [1, 0],
            "Medications": [0, 1],
        }
    )
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_SUMMARY": str(summary_dir),
        "ENABLE_SUMMARY": True,
        "SUMMARY": {
            "outputs": {
                "runtime_summary_config_json": True,
                "class_distribution_csv": False,
                "class_distribution_plot": False,
                "class_cooccurrence_csv": False,
                "class_cooccurrence_plot": False,
                "missingness_matrix_xlsx": False,
                "missingness_matrix_raw_train_xlsx": False,
                "missingness_matrix_raw_test_xlsx": False,
                "missingness_matrix_raw_interview_xlsx": False,
                "missingness_matrix_raw_combined_xlsx": False,
                "missingness_matrix_standardized_train_xlsx": False,
                "missingness_matrix_standardized_test_xlsx": False,
                "missingness_matrix_standardized_interview_xlsx": False,
                "missingness_matrix_standardized_combined_xlsx": False,
                "missingness_matrix_post_physician_merge_train_xlsx": False,
                "missingness_matrix_post_physician_merge_test_xlsx": False,
                "missingness_matrix_post_physician_merge_interview_xlsx": False,
                "missingness_matrix_post_physician_merge_combined_xlsx": False,
                "missingness_matrix_final_xlsx": False,
            }
        },
        "REPRODUCIBILITY": {
            "folder_enabled": True,
            "folder_path": "reproducibility_artifacts",
            "artifacts": {
                "summary_index_json": False,
                "summary_config_json": False,
            },
        },
        "fuzzy_threshold": 85,
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "physician_count": 2,
    }

    _ = preprocessing(cfg)

    reproducibility_dir = summary_dir.parent / "reproducibility_artifacts"
    assert not (reproducibility_dir / "summary_config.json").exists()


def test_preprocessing_summary_writes_combined_index_map(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"

    train = pd.DataFrame({"Item": ["CBC"], "Diagnoses": [1], "Medications": [0]})
    train["Patient"] = [1.0]
    train["Physician"] = [10.0]
    test = pd.DataFrame({"Item": ["CBC"], "Diagnoses": [1], "Medications": [0], "Patient": [2.0], "Physician": [10.0]})
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_SUMMARY": str(summary_dir),
        "fuzzy_threshold": 85,
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "patient_col": "Patient",
        "physician_col": "Physician",
        "physician_count": 1,
    }

    preprocessing(cfg)

    # index_map.json now always goes to the reproducibility dir (default: parent / reproducibility_artifacts)
    default_repro_dir = summary_dir.parent / "reproducibility_artifacts"
    assert (default_repro_dir / "index_map.json").exists()
    assert not (summary_dir / "train_map.json").exists()
    assert not (summary_dir / "test_map.json").exists()
    assert not (summary_dir / "eval_map.json").exists()


def test_preprocessing_writes_combined_index_map_to_reproducibility_dir_when_configured(tmp_path):
    dataset_path = tmp_path / "dataset.xlsx"
    summary_dir = tmp_path / "summary"
    reproducibility_dir = tmp_path / "reproducibility"

    train = pd.DataFrame({"Item": ["CBC"], "Diagnoses": [1], "Medications": [0], "Patient": [1.0], "Physician": [10.0]})
    test = pd.DataFrame({"Item": ["CBC"], "Diagnoses": [1], "Medications": [0], "Patient": [2.0], "Physician": [10.0]})
    interview = test.copy(deep=True)
    _write_dataset(dataset_path, train, test, interview)

    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_SUMMARY": str(summary_dir),
        "DIR_REPRODUCIBILITY": str(reproducibility_dir),
        "fuzzy_threshold": 85,
        "item_col": "Item",
        "classes": ["Diagnoses", "Medications"],
        "patient_col": "Patient",
        "physician_col": "Physician",
        "physician_count": 1,
    }

    preprocessing(cfg)

    assert (reproducibility_dir / "index_map.json").exists()
    assert not (summary_dir / "index_map.json").exists()
    assert not (summary_dir.parent / "reproducibility_artifacts" / "index_map.json").exists()


def test_setup_summary_logger_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            logger = setup_summary_logger(summary_dir, "summary_log.txt")

            log_file = summary_dir / "summary_log.txt"
            assert log_file.exists()

            logger.info("Test message")
            assert log_file.stat().st_size > 0
        finally:
            _close_summary_logger()


def test_setup_summary_logger_with_relative_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            logger = setup_summary_logger(summary_dir, "logs/pipeline.log")

            log_file = summary_dir / "logs" / "pipeline.log"
            assert log_file.exists()

            logger.info("Test message")
            assert log_file.stat().st_size > 0
        finally:
            _close_summary_logger()


def test_setup_summary_logger_with_absolute_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as abs_tmpdir:
            try:
                summary_dir = Path(tmpdir)
                log_path = Path(abs_tmpdir) / "summary.log"
                logger = setup_summary_logger(summary_dir, str(log_path))

                assert log_path.exists()

                logger.info("Test message")
                assert log_path.stat().st_size > 0
            finally:
                _close_summary_logger()


def test_log_high_level_event_logs_to_both():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            summary_logger = setup_summary_logger(summary_dir, "summary_log.txt")
            root_logger = logging.getLogger()

            with patch.object(root_logger, "log") as mock_root_log:
                log_high_level_event(
                    summary_logger, root_logger, logging.INFO,
                    "Test high-level event"
                )

                mock_root_log.assert_called_once()
                call_args = mock_root_log.call_args
                assert call_args[0][0] == logging.INFO
                assert "[SUMMARY] Test high-level event" in call_args[0][1]

            log_file = summary_dir / "summary_log.txt"
            content = log_file.read_text()
            assert "Test high-level event" in content
        finally:
            _close_summary_logger()


def test_log_high_level_event_with_none_root_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            summary_logger = setup_summary_logger(summary_dir, "summary_log.txt")

            log_high_level_event(
                summary_logger, None, logging.INFO,
                "Test event without root logger"
            )

            log_file = summary_dir / "summary_log.txt"
            content = log_file.read_text()
            assert "Test event without root logger" in content
        finally:
            _close_summary_logger()


def test_setup_summary_logger_default_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            _ = setup_summary_logger(summary_dir)

            log_file = summary_dir / "summary_log.txt"
            assert log_file.exists()
        finally:
            _close_summary_logger()


def test_summary_logger_isolation():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            logger = setup_summary_logger(summary_dir, "summary_log.txt")

            assert logger.propagate is False
        finally:
            _close_summary_logger()


def test_summary_logger_formats_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            summary_dir = Path(tmpdir)
            logger = setup_summary_logger(summary_dir, "summary_log.txt")
            logger.info("Test formatting")

            log_file = summary_dir / "summary_log.txt"
            content = log_file.read_text()

            lines = content.strip().split("\n")
            assert len(lines) > 0
            assert "INFO" in lines[0]
            assert "Test formatting" in lines[0]
        finally:
            _close_summary_logger()


def test_load_default_summary_returns_valid_config():
    from shared.reporting.input_summary.utils.settings_loader import load_default_summary
    load_default_summary.cache_clear()
    cfg = load_default_summary()
    assert isinstance(cfg, dict)
    assert "outputs" in cfg
    assert "plots" in cfg
    assert isinstance(cfg["strict"], bool)
    assert "missing_csv" in cfg["outputs"]
