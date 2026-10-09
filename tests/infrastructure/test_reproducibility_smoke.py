from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracks.representation.models.EnsemblePredictor import load_ensemble_predictor
from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline
import shared.embeddings.compute_embeddings as _ce_mod
from shared.preprocessing.preprocessing import preprocess


def _write_dataset(path: Path, train: pd.DataFrame, test: pd.DataFrame, interview: pd.DataFrame) -> None:
    with pd.ExcelWriter(path) as writer:
        train.to_excel(writer, sheet_name="train", index=False)
        test.to_excel(writer, sheet_name="test", index=False)
        interview.to_excel(writer, sheet_name="interview", index=False)


def _base_cfg(tmp_path: Path, model_dir: Path) -> dict:
    return {
        "PROJECT_ROOT": str(tmp_path),
        "DIR_DATASET": str(tmp_path / "dataset.xlsx"),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "DIR_JSON_MAP": str(tmp_path / "mapping.json"),
        "DIR_CONTEXT": str(tmp_path / "context.json"),
        "DIR_SUMMARY": str(tmp_path / "summary"),
        "item_col": "Item",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "classes": ["Diagnoses", "Medications"],
        "llm": "offline-smoke-model",
        "llm_revision": "offline",
        "hf_local_files_only": True,
        "batch_size": 8,
        "global_seed": 13,
        "pca_n_components": 50,
        "pca_whiten": False,
        "outer_folds": 2,
        "inner_folds": 2,
        "eval_pos_threshold": 0.5,
        "lr": 0.001,
        "weight_decay": 0.0,
        "weight_cap": 10.0,
        "early_stopping_patience": 1,
        "default_classes": ["Diagnoses", "Medications"],
        "DIR_MODEL": str(model_dir),
        "calibration_method": "none",
        "threshold_objective": "F0.5",
        "threshold_beta": 0.5,
        "threshold_search_grid": [0.25, 0.5, 0.75],
        "threshold_min_precision_floor": None,
        "threshold_min_recall_floor": None,
        "threshold_min_value": 0.01,
        "threshold_max_value": 0.99,
        "threshold_max_predicted_positive_rate": None,
        "hidden_dims": [],
        "dropout": 0.0,
        "activation": "gelu",
    }


def _extract_split_arrays(
    df: pd.DataFrame,
    split: str,
    embedding_map: dict,
    class_cols: list[str],
) -> tuple:
    """Extract feature matrix, label array and patient IDs for a given split."""
    split_df = df[df["split"] == split].reset_index(drop=True)
    X = np.vstack([embedding_map[item.casefold()] for item in split_df["Item"]])
    Y = split_df[class_cols].to_numpy(dtype=float)
    patient_ids = split_df["Patient"].to_numpy()
    return X, Y, patient_ids


def test_offline_reproducibility_smoke_pipeline(monkeypatch, tmp_path: Path):
    train = pd.DataFrame(
        {
            "Patient": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "Physician": [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11],
            "Item": [
                "CBC", "CBC", "BMP", "BMP",
                "Lipid", "Lipid", "ECG", "ECG",
                "Chem", "Chem", "UA", "UA",
                "TSH", "TSH", "A1C", "A1C",
            ],
            "Diagnoses": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            "Medications": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "Patient": [5, 5, 5, 5, 6, 6, 6, 6],
            "Physician": [10, 11, 10, 11, 10, 11, 10, 11],
            "Item": ["Echo", "Echo", "MRI", "MRI", "CT", "CT", "US", "US"],
            "Diagnoses": [1, 1, 0, 0, 1, 1, 0, 0],
            "Medications": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    interview = test.copy(deep=True)

    _write_dataset(tmp_path / "dataset.xlsx", train, test, interview)
    (tmp_path / "mapping.json").write_text(
        json.dumps(
            {
                "cbc": ["CBC"],
                "bmp": ["BMP"],
                "lipid": ["Lipid"],
                "ecg": ["ECG"],
                "chem": ["Chem"],
                "ua": ["UA"],
                "tsh": ["TSH"],
                "a1c": ["A1C"],
                "echo": ["Echo"],
                "mri": ["MRI"],
                "ct": ["CT"],
                "us": ["US"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "context.json").write_text(json.dumps({"source": "offline"}), encoding="utf-8")

    deterministic_embeddings = {
        "cbc": np.array([1.0, 0.0, 0.0]),
        "bmp": np.array([0.0, 1.0, 0.0]),
        "lipid": np.array([0.0, 0.0, 1.0]),
        "ecg": np.array([1.0, 1.0, 0.0]),
        "chem": np.array([1.0, 0.5, 0.0]),
        "ua": np.array([0.5, 1.0, 0.0]),
        "tsh": np.array([0.0, 1.0, 1.0]),
        "a1c": np.array([1.0, 0.0, 1.0]),
        "echo": np.array([0.2, 0.2, 0.8]),
        "mri": np.array([0.8, 0.2, 0.2]),
        "ct": np.array([0.3, 0.7, 0.2]),
        "us": np.array([0.6, 0.1, 0.3]),
    }

    monkeypatch.setattr(
        "shared.embeddings.compute_embeddings.compute_embeddings",
        lambda *args, **kwargs: deterministic_embeddings,
    )

    cfg_1 = _base_cfg(tmp_path, tmp_path / "model_a")
    df_1 = preprocess(cfg_1)
    context = json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))
    assert context == {"source": "offline"}

    embedding_map_1 = _ce_mod.compute_embeddings(cfg_1["llm"], cfg=cfg_1, base_dir=tmp_path)
    class_cols = ["Diagnoses_survey", "Medications_survey"]
    X_train_1, y_train_1, patient_ids_train_1 = _extract_split_arrays(df_1, "train", embedding_map_1, class_cols)
    X_test_1, y_test_1, _ = _extract_split_arrays(df_1, "test", embedding_map_1, class_cols)

    result_1 = train_ensemble_pipeline(X_train_1, y_train_1, patient_ids_train_1, X_test_1, y_test_1, cfg_1, patient_ids_test=np.arange(X_test_1.shape[0]))
    bundle_1 = Path(cfg_1["DIR_MODEL"]) / "ensemble_bundle.joblib"
    assert bundle_1.exists()

    predictor = load_ensemble_predictor(bundle_1)
    preds = predictor.predict_proba(X_test_1)
    assert preds.shape == y_test_1.shape

    cfg_2 = _base_cfg(tmp_path, tmp_path / "model_b")
    df_2 = preprocess(cfg_2)
    embedding_map_2 = _ce_mod.compute_embeddings(cfg_2["llm"], cfg=cfg_2, base_dir=tmp_path)
    X_train_2, y_train_2, patient_ids_train_2 = _extract_split_arrays(df_2, "train", embedding_map_2, class_cols)
    X_test_2, y_test_2, _ = _extract_split_arrays(df_2, "test", embedding_map_2, class_cols)
    result_2 = train_ensemble_pipeline(X_train_2, y_train_2, patient_ids_train_2, X_test_2, y_test_2, cfg_2, patient_ids_test=np.arange(X_test_2.shape[0]))

    assert result_1.keys() == result_2.keys()

    # Compare scalar macro metrics.
    for split in ("train", "test"):
        for metric, value in result_1[split].items():
            assert result_2[split][metric] == pytest.approx(value)

    # Compare deterministic probability outputs.
    np.testing.assert_allclose(result_1["oof_probs_cf"], result_2["oof_probs_cf"])
    if result_1["test_probs_cf"] is not None and result_2["test_probs_cf"] is not None:
        np.testing.assert_allclose(result_1["test_probs_cf"], result_2["test_probs_cf"])

    if "oof_strata" in result_1:
        assert "oof_strata" in result_2
        np.testing.assert_array_equal(result_1["oof_strata"], result_2["oof_strata"])
