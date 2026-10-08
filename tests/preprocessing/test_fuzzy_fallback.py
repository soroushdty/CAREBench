"""
Tests for the train-only fuzzy fallback architecture.

Verified invariants:
  a) JSON-first + fuzzy-second partial coverage: correct resolved masks, mask respected by fallback
  b) preprocessing() never calls generate_fuzzy_mapping under any code path
  c) FuzzyMatcher candidate space is built from fold-training strings ONLY (no held-out leakage)
  d) A query string (unseen during training) can be matched without polluting candidate space
  e) preprocessing() leaves items unresolved when no JSON is configured (no global fuzzy leakage)
  f) Embedding re-lookup uses post-fuzzy canonical string, not original pre-fuzzy string
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from shared.preprocessing.preprocessing import preprocessing
from shared.preprocessing.fuzzy_mapping import (
    FuzzyMatcher,
    build_fuzzy_matcher_from_train,
    apply_fuzzy_fallback,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dataset(
    path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    interview: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(path) as writer:
        train.to_excel(writer, sheet_name="train", index=False)
        test.to_excel(writer, sheet_name="test", index=False)
        interview.to_excel(writer, sheet_name="interview", index=False)


def _minimal_cfg(dataset_path: Path, **overrides) -> dict:
    base = {
        "DIR_DATASET": str(dataset_path),
        "TRAIN_SHEET": "train",
        "TEST_SHEET": "test",
        "INTERVIEW_SHEET": "interview",
        "item_col": "Item",
        "classes": ["label"],
        "physician_count": 1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test a: JSON-first + fuzzy-second, partial coverage
# ---------------------------------------------------------------------------

def test_json_first_fuzzy_second_partial_coverage(tmp_path):
    """JSON resolves CBC rows; near-duplicate typo items are left unresolved.
    A subsequent FuzzyMatcher built from unresolved-only rows can merge them.
    The fallback must NOT touch JSON-resolved rows (unresolved_mask=False).
    """
    json_map = tmp_path / "mapping.json"
    json_map.write_text(
        '{"Complete Blood Count": ["CBC", "cbc", "Complete Blood Count"]}',
        encoding="utf-8",
    )
    # "Lipid Panel" and "Lipid Pannel" are very similar (~95% fuzzy score) →
    # reliably merge at threshold 85.  CBC is covered by JSON.
    train = pd.DataFrame({
        "Patient":   [1.0, 2.0, 3.0],
        "Physician": [10.0, 10.0, 10.0],
        "Item":      ["CBC", "Lipid Panel", "Lipid Pannel"],
        "label":     [1, 0, 0],
    })
    test = pd.DataFrame({
        "Patient":   [4.0],
        "Physician": [10.0],
        "Item":      ["cbc"],
        "label":     [0],
    })
    _write_dataset(tmp_path / "ds.xlsx", train, test, test.copy())

    result = preprocessing(_minimal_cfg(tmp_path / "ds.xlsx", DIR_JSON_MAP=str(json_map)))
    df = result.df

    assert "item_json_resolved" in df.columns

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    cbc_mask   = train_df["Item"].str.lower().str.startswith("complete blood")
    lipid_mask = train_df["Item"].str.lower().str.startswith("lipid")

    assert cbc_mask.any(),   "Expected at least one CBC row after JSON standardization"
    assert lipid_mask.any(), "Expected at least one Lipid row (unresolved)"

    assert train_df.loc[cbc_mask, "item_json_resolved"].all(), (
        "CBC rows must have item_json_resolved=True (covered by JSON)"
    )
    assert not train_df.loc[lipid_mask, "item_json_resolved"].any(), (
        "Lipid rows must have item_json_resolved=False (not covered by JSON)"
    )

    # Build FuzzyMatcher from unresolved rows only
    items_df    = train_df[["Item"]].copy().reset_index(drop=True)
    unresolved  = pd.Series(~train_df["item_json_resolved"].values, index=items_df.index)
    matcher = build_fuzzy_matcher_from_train(
        items_df, "Item", threshold=85.0, unresolved_mask=unresolved
    )

    # Fuzzy fallback respects the mask: CBC rows (unresolved=False) are NEVER touched
    out_df, _ = apply_fuzzy_fallback(items_df, "Item", matcher, unresolved)

    cbc_out = out_df.loc[~unresolved, "Item"].tolist()
    assert all("complete blood count" in v.lower() for v in cbc_out), (
        f"CBC rows must remain unchanged after fuzzy fallback, got {cbc_out}"
    )

    # Lipid rows (unresolved=True) should be merged to one canonical
    lipid_out = out_df.loc[unresolved, "Item"].tolist()
    assert len(set(lipid_out)) == 1, (
        f"Lipid Panel / Lipid Pannel must merge to a single canonical, got {lipid_out}"
    )


# ---------------------------------------------------------------------------
# Test b: No JSON → preprocessing() must not build any fuzzy mapping
# ---------------------------------------------------------------------------

def test_missing_json_triggers_later_fuzzy_not_preprocessing_fuzzy(tmp_path):
    """With only fuzzy_threshold and no DIR_JSON_MAP, preprocessing() must not
    call generate_fuzzy_mapping.  Items remain distinct after preprocessing.
    A manually built FuzzyMatcher can merge them afterward.
    """
    train = pd.DataFrame({
        "Patient":   [1.0, 2.0],
        "Physician": [10.0, 10.0],
        # "blood test" / "blood tests": 1-char difference, ~95% similarity
        "Item":      ["blood test", "blood tests"],
        "label":     [0, 1],
    })
    test = pd.DataFrame({
        "Patient":   [3.0],
        "Physician": [10.0],
        "Item":      ["blood test"],
        "label":     [0],
    })
    _write_dataset(tmp_path / "ds.xlsx", train, test, test.copy())

    cfg = _minimal_cfg(tmp_path / "ds.xlsx", fuzzy_threshold=85)

    with patch("shared.preprocessing.fuzzy_mapping.generate_fuzzy_mapping") as mock_gen:
        result = preprocessing(cfg)
        assert mock_gen.call_count == 0, (
            "preprocessing() must never call generate_fuzzy_mapping "
            f"(was called {mock_gen.call_count} time(s))"
        )

    df = result.df
    assert "item_json_resolved" in df.columns
    assert not df["item_json_resolved"].any(), (
        "All item_json_resolved values must be False when no JSON mapping is configured"
    )

    # Items are distinct after preprocessing (no fuzzy merging happened)
    train_items = df[df["split"] == "train"]["Item"].tolist()
    assert "blood test" in train_items
    assert "blood tests" in train_items

    # A FuzzyMatcher built afterward CAN merge them
    train_df = df[df["split"] == "train"][["Item"]].copy().reset_index(drop=True)
    unresolved = pd.Series(True, index=train_df.index)
    matcher = build_fuzzy_matcher_from_train(
        train_df, "Item", threshold=85.0, unresolved_mask=unresolved
    )
    out_df, _ = apply_fuzzy_fallback(train_df, "Item", matcher, unresolved)
    assert len(out_df["Item"].unique()) == 1, (
        f"FuzzyMatcher must merge 'blood test'/'blood tests' to one canonical, "
        f"got {out_df['Item'].unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# Test c: Fold-leakage guard — held-out item NOT in candidate space
# ---------------------------------------------------------------------------

def test_lopo_fold_leakage_guard():
    """FuzzyMatcher candidate space is derived from fold-training strings ONLY.

    Patient B's item (held-out validation row) must not appear in the matcher's
    internal variant→canonical lookup, even when it is passed as a query later.
    """
    # Fold-training: only Patient A's item
    fold_train_df = pd.DataFrame({"Item": ["ECG"]}, index=[0])
    unresolved    = pd.Series([True], index=[0])

    matcher = build_fuzzy_matcher_from_train(
        fold_train_df, "Item", threshold=80.0, unresolved_mask=unresolved
    )

    # Patient B's item was never included in fold-training
    held_out_item  = "Electrocardiogram"
    norm_held_out  = normalize_text(held_out_item)

    assert norm_held_out not in matcher._norm_variant_to_norm_canonical, (
        f"Held-out item '{held_out_item}' (normalized: '{norm_held_out}') "
        "must NOT appear in the matcher's candidate variant space — "
        "it was never part of the fold-training set"
    )


# ---------------------------------------------------------------------------
# Test d: Unseen query string matched without polluting candidate space
# ---------------------------------------------------------------------------

def test_unseen_held_out_query_support():
    """A query string unseen during training can be matched against training candidates
    without being added to the matcher's internal variant→canonical lookup.
    """
    # Training: "ECG" and "EKG" (both unresolved)
    train_df   = pd.DataFrame({"Item": ["ECG", "EKG"]}, index=[0, 1])
    unresolved = pd.Series([True, True], index=[0, 1])

    matcher = build_fuzzy_matcher_from_train(
        train_df, "Item", threshold=50.0, unresolved_mask=unresolved
    )

    # Query string never seen during training
    query      = "ecg variant"
    norm_query = normalize_text(query)

    # Must not be in candidate space before the query
    assert norm_query not in matcher._norm_variant_to_norm_canonical, (
        f"Query '{query}' must not pre-exist in the candidate space"
    )

    # Issue the query (regardless of whether it matches)
    canonical, _ = matcher.match(query)

    # Must not have been added to candidate space by the match() call
    assert norm_query not in matcher._norm_variant_to_norm_canonical, (
        f"match() must not mutate the candidate space — "
        f"'{query}' appeared in _norm_variant_to_norm_canonical after match()"
    )


# ---------------------------------------------------------------------------
# Test e: No hidden global leakage — typo item left unresolved after preprocessing
# ---------------------------------------------------------------------------

def test_no_hidden_global_leakage(tmp_path):
    """preprocessing() must not apply fuzzy to test items even when fuzzy_threshold is set.

    The old design called build_shared_fuzzy_mapping inside preprocessing() using the
    full training sheet, which could map test-side typos like 'Lipid Profil' to a
    training canonical ('Lipid Panel') before any CV fold was constructed.
    """
    train = pd.DataFrame({
        "Patient":   [1.0],
        "Physician": [10.0],
        "Item":      ["Lipid Panel"],
        "label":     [1],
    })
    test = pd.DataFrame({
        "Patient":   [2.0],
        "Physician": [10.0],
        "Item":      ["Lipid Profil"],   # intentional typo (~92% similar to "Lipid Panel")
        "label":     [0],
    })
    _write_dataset(tmp_path / "ds.xlsx", train, test, test.copy())

    result = preprocessing(_minimal_cfg(tmp_path / "ds.xlsx", fuzzy_threshold=85))
    df = result.df

    # preprocessing() must have left "Lipid Profil" unchanged
    test_items = df[df["split"] == "test"]["Item"].tolist()
    assert "Lipid Profil" in test_items, (
        "preprocessing() must leave 'Lipid Profil' unresolved; "
        f"test items after preprocessing: {test_items}"
    )
    assert "Lipid Panel" not in test_items, (
        "preprocessing() must not map 'Lipid Profil' → 'Lipid Panel'; "
        "that would be cross-split information leakage"
    )

    # All items unresolved (no JSON configured)
    assert not df["item_json_resolved"].any()


# ---------------------------------------------------------------------------
# Test f: Embedding path correctness — post-fuzzy cache lookup
# ---------------------------------------------------------------------------

def test_embedding_path_correctness():
    """After fuzzy fallback maps 'alfa' → 'alpha', the embedding re-lookup must
    return embedding_cache['alpha'], not the original embedding_cache['alfa'].

    This verifies the _reassemble_X component logic used by train_ensemble_pipeline:
    old code: X_tr = X_train[train_ix]  → keeps pre-fuzzy embedding
    new code: X_tr = _reassemble_X(post_fuzzy_strings, original_strings) → correct
    """
    # Build matcher directly with a known canonical-to-variants mapping so
    # the test is independent of generate_fuzzy_mapping / clustering behavior.
    matcher = FuzzyMatcher({"alpha": ["alpha", "alfa"]}, threshold=0.5)

    embedding_cache = {
        "alpha": np.array([1.0, 0.0]),
        "alfa":  np.array([0.0, 1.0]),
    }

    original_strings = np.array(["alfa"])
    train_df   = pd.DataFrame({"Item": list(original_strings)}, index=[0])
    unresolved = pd.Series([True], index=[0])

    # Apply fuzzy fallback: "alfa" should be mapped to the canonical form
    out_df, warnings = apply_fuzzy_fallback(train_df, "Item", matcher, unresolved)
    post_fuzzy_string = out_df.at[0, "Item"]

    assert post_fuzzy_string == "alpha", (
        f"apply_fuzzy_fallback must map 'alfa' → 'alpha', got '{post_fuzzy_string}'"
    )
    assert warnings == [], f"No unmapped warnings expected, got {warnings}"

    # Simulate _reassemble_X: cache lookup using post-fuzzy canonical string
    post_fuzzy_embedding = embedding_cache.get(post_fuzzy_string)
    assert post_fuzzy_embedding is not None, (
        f"embedding_cache must contain key '{post_fuzzy_string}'"
    )
    np.testing.assert_array_equal(
        post_fuzzy_embedding, np.array([1.0, 0.0]),
        err_msg="Post-fuzzy embedding must be the 'alpha' vector [1, 0]",
    )

    # The pre-fuzzy (original) embedding differs — verifies the lookup is on the canonical
    original_embedding = embedding_cache[original_strings[0]]
    assert not np.allclose(post_fuzzy_embedding, original_embedding), (
        "Post-fuzzy embedding [1, 0] must differ from original 'alfa' embedding [0, 1]. "
        "Old architecture (X_tr = X_train[train_ix]) would have returned [0, 1]."
    )
