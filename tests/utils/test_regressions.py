import pandas as pd
from shared.preprocessing.preprocessing import preprocess

def test_notebook_split_logic(tmp_path):
    # Simulate a dataset with notebook split logic
    df = pd.DataFrame({
        "Patient": [1, 1, 2, 2, 3, 3],
        "split": ["train", "test", "train", "test", "train", "test"],
        "Item": ["A", "B", "C", "D", "E", "F"],
        "Diagnoses": [1, 0, 1, 0, 1, 0],
        "Medications": [0, 1, 0, 1, 0, 1],
    })
    dataset_path = tmp_path / "dataset.xlsx"
    with pd.ExcelWriter(dataset_path) as writer:
        df[df["split"] == "train"].to_excel(writer, sheet_name="train", index=False)
        df[df["split"] == "test"].to_excel(writer, sheet_name="test", index=False)
        df[df["split"] == "test"].to_excel(writer, sheet_name="interview", index=False)
    cfg = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "item_col": "Item",
        "patient_col": "Patient",
        "classes": ["Diagnoses", "Medications"],
        "DIR_SUMMARY": str(tmp_path / "summary"),
        "physician_count": 1,
        "llm": "offline-smoke-model",
        "hf_local_files_only": True,
    }
    result = preprocess(cfg)
    assert set(result["split"]) == {"train", "test"}

