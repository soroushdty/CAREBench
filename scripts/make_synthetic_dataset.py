#!/usr/bin/env python3
"""Generate the synthetic example dataset in the paired-context data format.

Every patient, clinical snapshot, and label in this dataset is fictional and
generated deterministically by this script. It exists so the pipelines, tests,
and documentation have a runnable example; it carries no scientific meaning.

Outputs (default directory: examples/synthetic/):
    dataset.xlsx            — sheets ``train``, ``test`` (context-free reference
                              condition), ``interview`` (correct-context
                              reference condition)
    patient_summaries.json  — one clinical snapshot per patient
    mapping.json            — canonical item string → raw item variants

Usage:
    python scripts/make_synthetic_dataset.py [--out examples/synthetic] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES: list[str] = [
    "Behavioral health",
    "Diagnoses",
    "Disabilities",
    "Infectious diseases",
    "Genetics",
    "Medications",
    "Sexual and reproductive health",
    "Social determinants of health",
    "Violence",
    "Other",
]

# Short aliases used in the tables below.
_BH, _DX, _DIS, _ID, _GEN, _MED, _SRH, _SDOH, _VIO, _OTH = CATEGORIES

# Generic EHR item strings with their context-free category assignment.
ITEM_POOL: dict[str, list[str]] = {
    "Essential hypertension": [_DX],
    "Type 2 diabetes mellitus without complications": [_DX],
    "Hyperlipidemia, unspecified": [_DX],
    "Seasonal allergic rhinitis": [_DX],
    "Gastroesophageal reflux disease": [_DX],
    "Osteoarthritis of right knee": [_DX],
    "Chronic low back pain": [_DX],
    "Migraine without aura": [_DX],
    "Iron deficiency anemia": [_DX],
    "Vitamin D deficiency": [_DX],
    "Hypothyroidism": [_DX],
    "Asthma, mild intermittent": [_DX],
    "Generalized anxiety disorder": [_BH, _DX],
    "Insomnia disorder": [_BH, _DX],
    "Alcohol use, unspecified": [_BH],
    "Tobacco use": [_BH],
    "Hearing loss, bilateral": [_DIS, _DX],
    "Uses a wheelchair": [_DIS],
    "Low vision, left eye": [_DIS, _DX],
    "Hepatitis C antibody test": [_ID],
    "HIV-1/2 antigen/antibody screen": [_ID, _SRH],
    "Influenza vaccine administered": [_ID, _OTH],
    "Latent tuberculosis infection": [_ID, _DX],
    "BRCA1/2 genetic testing": [_GEN],
    "Family history of Huntington disease": [_GEN],
    "Carrier screening panel": [_GEN, _SRH],
    "Lisinopril 10 mg tablet": [_MED],
    "Metformin 500 mg tablet": [_MED],
    "Atorvastatin 20 mg tablet": [_MED],
    "Omeprazole 20 mg capsule": [_MED],
    "Ibuprofen 400 mg tablet": [_MED],
    "Levothyroxine 50 mcg tablet": [_MED],
    "Albuterol inhaler 90 mcg": [_MED],
    "Sertraline 50 mg tablet": [_MED, _BH],
    "Buprenorphine-naloxone 8 mg-2 mg film": [_MED, _BH],
    "Oral contraceptive pill": [_MED, _SRH],
    "Pregnancy test, urine": [_SRH],
    "Chlamydia NAAT": [_SRH, _ID],
    "Erectile dysfunction": [_SRH, _DX],
    "Lives alone": [_SDOH],
    "Food insecurity screen positive": [_SDOH],
    "Unstable housing": [_SDOH],
    "Unemployed": [_SDOH],
    "Intimate partner violence screen": [_VIO],
    "Assault by unspecified means": [_VIO],
    "Complete blood count": [_OTH],
    "Basic metabolic panel": [_OTH],
    "Chest X-ray, two views (normal)": [_OTH],
    "Electrocardiogram (normal sinus rhythm)": [_OTH],
    "Hemoglobin A1c 7.2%": [_OTH, _DX],
    "Alanine aminotransferase 88 U/L (elevated)": [_OTH],
    "Urine drug screen": [_OTH, _BH],
    "Gamma-glutamyl transferase 120 U/L (elevated)": [_OTH],
    "Thyroid stimulating hormone": [_OTH],
    "Lipid panel": [_OTH],
    "Abdominal ultrasound (normal)": [_OTH],
}

# Fictional patients. ``test_items`` are rated in both reference conditions;
# ``context_shift`` lists categories that the patient's snapshot makes
# relevant for specific test items in the correct-context condition.
PATIENTS: list[dict] = [
    {
        "summary": "46-year-old woman with a history of opioid use disorder in sustained remission.",
        "medical_history": ["Chronic low back pain", "Hepatitis C, treated"],
        "allergies": ["Penicillin"],
        "medication_history": ["Naloxone nasal spray 4 mg"],
        "social_history": ["Works as a cashier", "Attends weekly peer support group"],
        "labs": ["Hepatitis C RNA undetectable"],
        "radiology": ["Lumbar spine MRI (mild degenerative changes)"],
        "procedures": ["Physical therapy evaluation"],
        "test_items": {
            "Buprenorphine-naloxone 8 mg-2 mg film": [],
            "Urine drug screen": [_BH],
            "Alanine aminotransferase 88 U/L (elevated)": [_ID, _BH],
            "Ibuprofen 400 mg tablet": [],
            "Lives alone": [],
        },
    },
    {
        "summary": "31-year-old man with a history of HIV infection on antiretroviral therapy.",
        "medical_history": ["Latent syphilis, treated", "Seasonal allergic rhinitis"],
        "allergies": ["No known allergies"],
        "medication_history": ["Bictegravir/emtricitabine/tenofovir alafenamide"],
        "social_history": ["Graduate student", "In a monogamous relationship"],
        "labs": ["CD4 count 640 cells/uL", "HIV viral load undetectable"],
        "radiology": ["No imaging on file"],
        "procedures": ["Routine vaccination review"],
        "test_items": {
            "Chlamydia NAAT": [],
            "Complete blood count": [_ID],
            "Hepatitis C antibody test": [_SRH],
            "Influenza vaccine administered": [],
            "Basic metabolic panel": [_ID, _MED],
        },
    },
    {
        "summary": "58-year-old man with alcohol use disorder and recent job loss.",
        "medical_history": ["Essential hypertension", "Alcoholic fatty liver"],
        "allergies": ["Sulfa drugs"],
        "medication_history": ["Naltrexone 50 mg", "Thiamine 100 mg"],
        "social_history": ["Recently laid off from construction work", "Divorced"],
        "labs": ["Mean corpuscular volume 104 fL (elevated)"],
        "radiology": ["Liver ultrasound (increased echogenicity)"],
        "procedures": ["Alcohol withdrawal assessment"],
        "test_items": {
            "Gamma-glutamyl transferase 120 U/L (elevated)": [_BH],
            "Unemployed": [_BH],
            "Lisinopril 10 mg tablet": [],
            "Abdominal ultrasound (normal)": [],
            "Insomnia disorder": [],
        },
    },
    {
        "summary": "27-year-old woman, 10 weeks pregnant, with a history of depression.",
        "medical_history": ["Major depressive disorder, recurrent, in partial remission", "Asthma, mild intermittent"],
        "allergies": ["Latex"],
        "medication_history": ["Prenatal vitamin", "Escitalopram 10 mg"],
        "social_history": ["Lives with partner", "Reports partner monitors her phone"],
        "labs": ["Blood type O positive"],
        "radiology": ["First-trimester ultrasound (single live intrauterine pregnancy)"],
        "procedures": ["Prenatal intake visit"],
        "test_items": {
            "Sertraline 50 mg tablet": [_SRH],
            "Intimate partner violence screen": [_SRH],
            "Carrier screening panel": [],
            "Albuterol inhaler 90 mcg": [],
            "Thyroid stimulating hormone": [_SRH],
        },
    },
    {
        "summary": "67-year-old man with a family history of hereditary cancer syndromes.",
        "medical_history": ["Type 2 diabetes mellitus", "Hyperlipidemia"],
        "allergies": ["No known allergies"],
        "medication_history": ["Metformin 1000 mg", "Atorvastatin 40 mg"],
        "social_history": ["Retired teacher", "Lives with spouse"],
        "labs": ["Prostate-specific antigen 1.1 ng/mL"],
        "radiology": ["Colonoscopy referral pending"],
        "procedures": ["Genetic counseling visit"],
        "test_items": {
            "BRCA1/2 genetic testing": [],
            "Hemoglobin A1c 7.2%": [],
            "Erectile dysfunction": [_MED],
            "Lipid panel": [],
            "Family history of Huntington disease": [_BH],
        },
    },
    {
        "summary": "39-year-old woman with spinal cord injury after an assault three years ago.",
        "medical_history": ["Incomplete paraplegia", "Post-traumatic stress disorder"],
        "allergies": ["Codeine"],
        "medication_history": ["Baclofen 10 mg", "Prazosin 2 mg"],
        "social_history": ["Receives disability benefits", "Housing assistance application pending"],
        "labs": ["Urinalysis (normal)"],
        "radiology": ["Thoracic spine CT (stable post-surgical changes)"],
        "procedures": ["Wheelchair seating evaluation"],
        "test_items": {
            "Uses a wheelchair": [_VIO],
            "Assault by unspecified means": [_BH],
            "Unstable housing": [_DIS],
            "Insomnia disorder": [_VIO],
            "Chest X-ray, two views (normal)": [],
        },
    },
]

TRAIN_ITEMS_PER_PATIENT = 16
FLIP_PROB = 0.06  # per-observer, per-label disagreement noise


def _rate(categories: list[str], rng: random.Random) -> dict[str, int]:
    """One observer's binary labels for an item: base categories plus noise."""
    labels = {c: int(c in categories) for c in CATEGORIES}
    for c in CATEGORIES:
        if rng.random() < FLIP_PROB:
            labels[c] = 1 - labels[c]
    return labels


def build(seed: int) -> tuple[dict[str, list[dict]], dict, dict]:
    rng = random.Random(seed)
    sheets: dict[str, list[dict]] = {"train": [], "test": [], "interview": []}
    summaries: dict[str, dict] = {}

    for idx, patient in enumerate(PATIENTS, start=1):
        observers = (2 * idx - 1, 2 * idx)
        summaries[str(idx)] = {
            k: v for k, v in patient.items() if k != "test_items"
        }

        test_items = patient["test_items"]
        train_pool = sorted(i for i in ITEM_POOL if i not in test_items)
        train_items = rng.sample(train_pool, TRAIN_ITEMS_PER_PATIENT)

        for item in train_items:
            for obs in observers:
                sheets["train"].append(
                    {"Patient": idx, "Physician": obs, "Item": item, **_rate(ITEM_POOL[item], rng)}
                )
        for item, shift in test_items.items():
            base = ITEM_POOL[item]
            for obs in observers:
                sheets["test"].append(
                    {"Patient": idx, "Physician": obs, "Item": item, **_rate(base, rng)}
                )
                sheets["interview"].append(
                    {"Patient": idx, "Physician": obs, "Item": item,
                     **_rate(sorted(set(base) | set(shift)), rng)}
                )

    mapping = {item.lower(): [item] for item in sorted(ITEM_POOL)}
    return sheets, summaries, mapping


def _check_no_label_leakage(sheets: dict[str, list[dict]], summaries: dict) -> None:
    """Item text must not appear verbatim in its own patient's snapshot."""
    for row in sheets["test"]:
        record = summaries[str(row["Patient"])]
        for field, value in record.items():
            text = " ".join(value) if isinstance(value, list) else str(value)
            if row["Item"].lower() in text.lower():
                raise ValueError(f"Item {row['Item']!r} leaks into patient {row['Patient']} field {field!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="examples/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args(argv)

    import pandas as pd

    sheets, summaries, mapping = build(args.seed)
    _check_no_label_leakage(sheets, summaries)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["Patient", "Physician", "Item", *CATEGORIES]
    with pd.ExcelWriter(out / "dataset.xlsx", engine="openpyxl") as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows, columns=columns).to_excel(writer, sheet_name=name, index=False)
    (out / "patient_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    for name, rows in sheets.items():
        print(f"{name}: {len(rows)} rows")
    print(f"Wrote {out}/dataset.xlsx, patient_summaries.json, mapping.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
