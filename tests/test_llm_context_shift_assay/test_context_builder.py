"""
Tests for tracks/reasoning/context_builder.py.

Run with:
    pytest tests/test_llm_context_shift_assay/test_context_builder.py --noconftest
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tracks.reasoning.context_builder import (
    CONTEXT_FIELDS,
    ContextBuilder,
    LabelLeakageError,
)

# ---------------------------------------------------------------------------
# Privacy_Category label names
# ---------------------------------------------------------------------------

PRIVACY_CATEGORY_LABELS = [
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

# ---------------------------------------------------------------------------
# Synthetic patient data fixture
# ---------------------------------------------------------------------------

SYNTHETIC_PATIENTS = {
    "P1": {
        "summary": "45-year-old patient with hypertension and mild anxiety.",
        "medical_history": ["Hypertension", "Mild anxiety disorder"],
        "allergies": ["Penicillin"],
        "medication_history": ["Lisinopril 10 mg", "Lorazepam 0.5 mg"],
        "social_history": ["Employed", "Non-smoker"],
        "labs": ["CBC (normal)", "BMP (normal)"],
        "radiology": ["Chest X-ray (normal)"],
        "procedures": ["EKG (normal)"],
    },
    "P2": {
        "summary": "32-year-old patient with type 2 diabetes and obesity.",
        "medical_history": ["Type 2 diabetes mellitus", "Obesity"],
        "allergies": ["No known allergies"],
        "medication_history": ["Metformin 500 mg", "Atorvastatin 20 mg"],
        "social_history": ["Unemployed", "Occasional smoker"],
        "labs": ["HbA1c 8.2%", "Lipid panel (elevated LDL)"],
        "radiology": [],
        "procedures": ["Foot exam"],
    },
    "P3": {
        "summary": "60-year-old patient with chronic back pain and depression.",
        "medical_history": ["Chronic low back pain", "Major depressive disorder"],
        "allergies": ["Sulfa drugs"],
        "medication_history": ["Ibuprofen 400 mg", "Sertraline 50 mg"],
        "social_history": ["Retired", "Married"],
        "labs": ["TSH (normal)", "Vitamin D (low)"],
        "radiology": ["Lumbar spine MRI (degenerative changes)"],
        "procedures": ["Physical therapy"],
    },
}

# Item texts that do NOT appear in any synthetic patient context field
SAFE_ITEM_TEXTS = [
    "Does the patient have a history of substance abuse?",
    "Is there a record of genetic testing?",
    "Does the patient use interpreter services?",
]


@pytest.fixture
def summaries_file(tmp_path: Path) -> Path:
    """Write synthetic patient_summaries.json to a temp file and return its path."""
    path = tmp_path / "patient_summaries.json"
    path.write_text(json.dumps(SYNTHETIC_PATIENTS), encoding="utf-8")
    return path


@pytest.fixture
def builder(summaries_file: Path) -> ContextBuilder:
    """ContextBuilder loaded with synthetic patient data."""
    return ContextBuilder(summaries_path=summaries_file)


# ---------------------------------------------------------------------------
# test_shuffled_id_differs
# ---------------------------------------------------------------------------


def test_shuffled_id_differs(builder: ContextBuilder) -> None:
    """Shuffled patient IDs must differ from the correct patient ID for all pairs.

    """
    rng = np.random.default_rng(42)
    patient_ids = list(SYNTHETIC_PATIENTS.keys())

    for pid in patient_ids:
        for item_text in SAFE_ITEM_TEXTS:
            builder.build_shuffled_context(pid, item_text, rng)

    shuffled_map = builder.get_shuffled_id_map()

    assert shuffled_map, "Shuffled ID map should not be empty."
    for (correct_pid, _item), shuffled_pid in shuffled_map.items():
        assert shuffled_pid != correct_pid, (
            f"Shuffled patient ID '{shuffled_pid}' must differ from "
            f"correct patient ID '{correct_pid}'."
        )


# ---------------------------------------------------------------------------
# test_shuffled_context_differs
# ---------------------------------------------------------------------------


def test_shuffled_context_differs(builder: ContextBuilder) -> None:
    """Shuffled context text must differ from the correct context text.

    When the shuffled patient is different from the correct patient (which is
    guaranteed by the pool exclusion logic), the context texts must differ.

    """
    rng = np.random.default_rng(42)
    patient_ids = list(SYNTHETIC_PATIENTS.keys())

    for pid in patient_ids:
        item_text = SAFE_ITEM_TEXTS[0]
        correct_ctx = builder.build_correct_context(pid, item_text)
        shuffled_ctx = builder.build_shuffled_context(pid, item_text, rng)

        # The shuffled patient is always different, so the context must differ
        shuffled_map = builder.get_shuffled_id_map()
        shuffled_pid = shuffled_map[(pid, item_text)]

        if shuffled_pid != pid:
            assert shuffled_ctx != correct_ctx, (
                f"Shuffled context for patient '{pid}' (shuffled to '{shuffled_pid}') "
                "must differ from the correct context."
            )


# ---------------------------------------------------------------------------
# test_shuffled_deterministic
# ---------------------------------------------------------------------------


def test_shuffled_deterministic(summaries_file: Path) -> None:
    """Shuffled assignments must be identical across two runs with the same seed.

    """
    seed = 99

    # First run
    builder_a = ContextBuilder(summaries_path=summaries_file)
    rng_a = np.random.default_rng(seed)
    patient_ids = list(SYNTHETIC_PATIENTS.keys())
    contexts_a: list[str] = []
    for pid in patient_ids:
        for item_text in SAFE_ITEM_TEXTS:
            ctx = builder_a.build_shuffled_context(pid, item_text, rng_a)
            contexts_a.append(ctx)
    map_a = builder_a.get_shuffled_id_map()

    # Second run — fresh builder and fresh rng with the same seed
    builder_b = ContextBuilder(summaries_path=summaries_file)
    rng_b = np.random.default_rng(seed)
    contexts_b: list[str] = []
    for pid in patient_ids:
        for item_text in SAFE_ITEM_TEXTS:
            ctx = builder_b.build_shuffled_context(pid, item_text, rng_b)
            contexts_b.append(ctx)
    map_b = builder_b.get_shuffled_id_map()

    assert contexts_a == contexts_b, (
        "Shuffled context texts must be identical across two runs with the same seed."
    )
    assert map_a == map_b, (
        "Shuffled patient ID maps must be identical across two runs with the same seed."
    )


# ---------------------------------------------------------------------------
# test_no_item_in_context
# ---------------------------------------------------------------------------


def test_no_item_in_context(summaries_file: Path) -> None:
    """Item text must not appear in any patient context field.

    Uses the bundled example patient_summaries.json when available; falls back to
    synthetic data.  Verifies that build_correct_context does NOT raise
    LabelLeakageError for safe item texts (i.e., the items are genuinely
    absent from the context).

    """
    real_path = Path("examples/synthetic/patient_summaries.json")
    path = real_path if real_path.exists() else summaries_file

    cb = ContextBuilder(summaries_path=path)
    with path.open("r", encoding="utf-8") as fh:
        summaries: dict = json.load(fh)

    patient_ids = list(summaries.keys())

    for pid in patient_ids:
        for item_text in SAFE_ITEM_TEXTS:
            # build_correct_context raises LabelLeakageError if item is found
            ctx = cb.build_correct_context(pid, item_text)

            # Double-check: item text must not appear in the returned context
            assert item_text.lower() not in ctx.lower(), (
                f"Item text '{item_text}' must not appear in context for patient '{pid}'."
            )


def test_no_item_in_context_all_fields(builder: ContextBuilder) -> None:
    """Item text must not appear in any individual context field.

    """
    for pid, fields in SYNTHETIC_PATIENTS.items():
        for item_text in SAFE_ITEM_TEXTS:
            for field in CONTEXT_FIELDS:
                raw = fields.get(field, "")
                if isinstance(raw, list):
                    field_text = " ".join(raw)
                else:
                    field_text = str(raw)
                assert item_text.lower() not in field_text.lower(), (
                    f"Item '{item_text}' must not appear in field '{field}' "
                    f"of patient '{pid}'."
                )


# ---------------------------------------------------------------------------
# test_no_category_labels_in_context
# ---------------------------------------------------------------------------


def test_no_category_labels_in_context(builder: ContextBuilder) -> None:
    """Privacy_Category label names must not appear in patient context text.

    """
    patient_ids = list(SYNTHETIC_PATIENTS.keys())

    for pid in patient_ids:
        for item_text in SAFE_ITEM_TEXTS:
            ctx = builder.build_correct_context(pid, item_text)
            ctx_lower = ctx.lower()

            for label in PRIVACY_CATEGORY_LABELS:
                assert label.lower() not in ctx_lower, (
                    f"Privacy_Category label '{label}' must not appear in "
                    f"context for patient '{pid}'."
                )


def test_no_category_labels_in_real_context() -> None:
    """Privacy_Category labels must not appear in real patient_summaries.json context.

    The check uses whole-word matching so that common English words that happen
    to share a name with a category (e.g. "other") are not flagged when they
    appear as part of ordinary clinical text.  The intent of this check is that
    the context does not contain the category names *as classification labels*
    (e.g. as JSON keys or explicit schema references).

    """
    import re

    real_path = Path("examples/synthetic/patient_summaries.json")
    if not real_path.exists():
        pytest.skip("examples/synthetic/patient_summaries.json not available; skipping real-data test.")

    cb = ContextBuilder(summaries_path=real_path)
    with real_path.open("r", encoding="utf-8") as fh:
        summaries: dict = json.load(fh)

    # Multi-word / underscored labels are unambiguous; single-word labels like
    # "other", "diagnoses", "genetics", "violence", "medications" may appear in
    # ordinary clinical prose.  We check that the label does NOT appear as a
    # standalone JSON-style key (i.e. surrounded by quotes or as a bare
    # identifier followed by a colon), which is the only form that would
    # constitute actual label leakage into the prompt.
    item_text = "Does the patient have a history of genetic testing?"
    for pid in summaries:
        ctx = cb.build_correct_context(pid, item_text)
        ctx_lower = ctx.lower()

        # Unambiguous multi-word labels (contain underscores) must not appear at all
        unambiguous_labels = [
            lbl for lbl in PRIVACY_CATEGORY_LABELS if "_" in lbl
        ]
        for label in unambiguous_labels:
            assert label.lower() not in ctx_lower, (
                f"Privacy_Category label '{label}' must not appear in "
                f"context for patient '{pid}'."
            )

        # For single-word labels, check they don't appear as JSON-style keys
        # (i.e. as `"label":` or `label:` patterns that would indicate schema leakage)
        single_word_labels = [
            lbl for lbl in PRIVACY_CATEGORY_LABELS if "_" not in lbl
        ]
        for label in single_word_labels:
            # Pattern: label as a JSON key — `"label"` or `label:` at start of line
            json_key_pattern = re.compile(
                r'["\']' + re.escape(label) + r'["\']',
                re.IGNORECASE,
            )
            assert not json_key_pattern.search(ctx), (
                f"Privacy_Category label '{label}' must not appear as a JSON key "
                f"in context for patient '{pid}'."
            )


# ---------------------------------------------------------------------------
# test_leakage_detection
# ---------------------------------------------------------------------------


def test_leakage_detection(tmp_path: Path) -> None:
    """ContextBuilder must raise LabelLeakageError when item text is injected into a context field.

    """
    # Inject the item text into the 'summary' field of patient P1
    injected_item = "patient has a history of substance abuse"
    poisoned_patients = {
        "P1": {
            "summary": f"45-year-old patient. {injected_item}.",
            "medical_history": ["Hypertension"],
            "allergies": ["Penicillin"],
            "medication_history": ["Lisinopril 10 mg"],
            "social_history": ["Employed"],
            "labs": ["CBC (normal)"],
            "radiology": ["Chest X-ray (normal)"],
            "procedures": ["EKG (normal)"],
        },
        "P2": {
            "summary": "32-year-old patient with type 2 diabetes.",
            "medical_history": ["Type 2 diabetes mellitus"],
            "allergies": ["No known allergies"],
            "medication_history": ["Metformin 500 mg"],
            "social_history": ["Unemployed"],
            "labs": ["HbA1c 8.2%"],
            "radiology": [],
            "procedures": ["Foot exam"],
        },
        "P3": {
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

    path = tmp_path / "poisoned_summaries.json"
    path.write_text(json.dumps(poisoned_patients), encoding="utf-8")

    cb = ContextBuilder(summaries_path=path)

    with pytest.raises(LabelLeakageError) as exc_info:
        cb.build_correct_context("P1", injected_item)

    err = exc_info.value
    assert err.patient_id == "P1"
    assert err.field_name == "summary"
    assert injected_item.lower() in err.matched_text.lower()


def test_leakage_detection_in_list_field(tmp_path: Path) -> None:
    """LabelLeakageError must be raised when item text appears in a list-valued field.

    """
    injected_item = "Metformin 500 mg"
    patients = {
        "P1": {
            "summary": "45-year-old patient.",
            "medical_history": ["Hypertension"],
            "allergies": ["Penicillin"],
            "medication_history": [injected_item, "Lisinopril 10 mg"],
            "social_history": ["Employed"],
            "labs": ["CBC (normal)"],
            "radiology": [],
            "procedures": [],
        },
        "P2": {
            "summary": "32-year-old patient.",
            "medical_history": ["Obesity"],
            "allergies": ["No known allergies"],
            "medication_history": ["Atorvastatin 20 mg"],
            "social_history": ["Unemployed"],
            "labs": ["HbA1c 8.2%"],
            "radiology": [],
            "procedures": [],
        },
        "P3": {
            "summary": "60-year-old patient.",
            "medical_history": ["Back pain"],
            "allergies": ["Sulfa drugs"],
            "medication_history": ["Ibuprofen 400 mg"],
            "social_history": ["Retired"],
            "labs": ["TSH (normal)"],
            "radiology": [],
            "procedures": [],
        },
    }

    path = tmp_path / "list_field_summaries.json"
    path.write_text(json.dumps(patients), encoding="utf-8")

    cb = ContextBuilder(summaries_path=path)

    with pytest.raises(LabelLeakageError) as exc_info:
        cb.build_correct_context("P1", injected_item)

    err = exc_info.value
    assert err.patient_id == "P1"
    assert err.field_name == "medication_history"


def test_leakage_detection_case_insensitive(tmp_path: Path) -> None:
    """LabelLeakageError must be raised for case-insensitive item text matches.

    """
    injected_item = "HYPERTENSION"
    patients = {
        "P1": {
            "summary": "45-year-old patient.",
            "medical_history": ["hypertension"],  # lowercase in field
            "allergies": ["Penicillin"],
            "medication_history": ["Lisinopril 10 mg"],
            "social_history": ["Employed"],
            "labs": ["CBC (normal)"],
            "radiology": [],
            "procedures": [],
        },
        "P2": {
            "summary": "32-year-old patient.",
            "medical_history": ["Obesity"],
            "allergies": ["No known allergies"],
            "medication_history": ["Metformin 500 mg"],
            "social_history": ["Unemployed"],
            "labs": ["HbA1c 8.2%"],
            "radiology": [],
            "procedures": [],
        },
        "P3": {
            "summary": "60-year-old patient.",
            "medical_history": ["Back pain"],
            "allergies": ["Sulfa drugs"],
            "medication_history": ["Ibuprofen 400 mg"],
            "social_history": ["Retired"],
            "labs": ["TSH (normal)"],
            "radiology": [],
            "procedures": [],
        },
    }

    path = tmp_path / "case_insensitive_summaries.json"
    path.write_text(json.dumps(patients), encoding="utf-8")

    cb = ContextBuilder(summaries_path=path)

    with pytest.raises(LabelLeakageError):
        cb.build_correct_context("P1", injected_item)
