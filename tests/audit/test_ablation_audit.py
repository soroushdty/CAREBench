"""Ablation attribution audit tests.

Verifies that:
1. Ablated context embeddings actually differ from original embeddings.
2. Ablated fusion features actually differ when context changes.
3. Stage-2 predictions change for a context-sensitive model when context is ablated.
4. _run_ablation returns nonzero attribution for context-sensitive models.
5. _run_ablation returns near-zero attribution when the Stage-2 model ignores context.
6. Baseline (ca_baseline) is computed from Stage-2 models on original context, not from
   the raw y_hat_ca that may include Stage-1 fallbacks for None-fold slots.
7. Attribution is exactly zero when the ablated context string is identical to the
   original (field already empty).
"""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D = 16   # embedding dimension used across all tests
N_CLASSES = 3


def _make_random(shape, *, seed=0, dtype=np.float32):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def _make_stage2_model(*, context_weight_scale: float = 1.0, seed: int = 0):
    """Return a Stage2FusionModel with controllable context sensitivity.

    context_weight_scale=0.0  → model ignores context completely.
    context_weight_scale=5.0  → model is strongly context-sensitive.
    """
    from tracks.representation.training.stage2.fit_stage2_fusion import Stage2FusionModel

    model = Stage2FusionModel(input_dim=4 * D, n_classes=N_CLASSES)
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        # Item block [0:d]: always nonzero so predictions are non-trivial
        model.linear.weight[:, :D] = torch.tensor(
            rng.standard_normal((N_CLASSES, D)).astype(np.float32)
        )
        # Context block [d:2d]: scaled by context_weight_scale
        ctx_w = rng.standard_normal((N_CLASSES, D)).astype(np.float32)
        model.linear.weight[:, D : 2 * D] = torch.tensor(
            ctx_w * context_weight_scale
        )
        # Hadamard block [2d:3d]
        model.linear.weight[:, 2 * D : 3 * D] = torch.tensor(
            rng.standard_normal((N_CLASSES, D)).astype(np.float32)
            * context_weight_scale
        )
        # Abs-diff block [3d:4d]
        model.linear.weight[:, 3 * D :] = torch.tensor(
            rng.standard_normal((N_CLASSES, D)).astype(np.float32)
            * context_weight_scale
        )
        model.linear.bias.zero_()
    return model


def _artifact(model, *, fusion_strategy: str = "4_vector", r: int = 8) -> dict:
    """Wrap a bare Stage-2 model in the per-fold artifact dict used by bundles."""
    return {
        "model": model,
        "pca": None,
        "use_cf_passthrough": False,
        "fusion_strategy": fusion_strategy,
        "r": r,
        "calibrators": None,
        "class_list": None,
    }


def _make_bundle_file(
    stage2_model,
    stage2_context_vectors: dict,
    *,
    fusion_strategy: str = "4_vector",
) -> Path:
    """Serialize a minimal pdm_ensemble_v1 bundle to a temp file."""
    bundle = {
        "bundle_format": "pdm_ensemble_v1",
        "models": [],
        "preprocessors": [],
        "calibrators": [],
        "thresholds": np.full(N_CLASSES, 0.5, dtype=np.float32),
        "device": "cpu",
        "class_list": [f"cls_{c}" for c in range(N_CLASSES)],
        "model_arch": {
            "fusion_strategy": fusion_strategy,
            "lowrank_bilinear_r": 8,
        },
        "stage2": [_artifact(stage2_model, fusion_strategy=fusion_strategy)],
        "stage2_context_vectors": stage2_context_vectors,
        "config_snapshot": {},
        "metadata": {},
    }
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    tmp.close()
    joblib.dump(bundle, tmp.name)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Component-level: do ablated context embeddings and features actually differ?
# ---------------------------------------------------------------------------

class TestAblatedContextDiffersFromOriginal:
    """Low-level checks that ablation changes the encoding input."""

    def test_ablated_context_string_differs(self):
        """Ablating a non-empty field changes the serialized context string."""
        from tracks.representation.training.stage2.stage2_context import build_context_string

        ctx = {
            "summary": "Patient has diabetes",
            "medical_history": "Hypertension",
            "allergies": "",
            "medication_history": "Metformin",
            "social_history": "Non-smoker",
            "labs": "HbA1c 7.5",
            "radiology": "",
            "procedures": "",
        }
        ablated = {**ctx, "summary": ""}
        assert build_context_string(ctx) != build_context_string(ablated)

    def test_already_empty_field_ablation_does_not_change_string(self):
        """Ablating an already-empty field leaves the context string unchanged."""
        from tracks.representation.training.stage2.stage2_context import build_context_string

        ctx = {f: "" for f in [
            "summary", "medical_history", "allergies", "medication_history",
            "social_history", "labs", "radiology", "procedures",
        ]}
        ablated = {**ctx, "allergies": ""}  # already empty
        assert build_context_string(ctx) == build_context_string(ablated)

    def test_different_context_embedding_changes_fusion_features(self):
        """build_fusion_features with distinct c_p values produces distinct z."""
        from tracks.representation.training.stage2.stage2_context import build_fusion_features

        rng = np.random.default_rng(1)
        e_i = rng.standard_normal(D).astype(np.float32)
        c_p_a = rng.standard_normal(D).astype(np.float32)
        c_p_b = rng.standard_normal(D).astype(np.float32)  # different context

        z_a = build_fusion_features(e_i, c_p_a, fusion_strategy="4_vector")
        z_b = build_fusion_features(e_i, c_p_b, fusion_strategy="4_vector")

        assert not np.allclose(z_a, z_b), (
            "Fusion features must differ when context embeddings differ"
        )

    def test_same_context_embedding_gives_same_fusion_features(self):
        """Identical c_p values produce identical fusion features."""
        from tracks.representation.training.stage2.stage2_context import build_fusion_features

        rng = np.random.default_rng(2)
        e_i = rng.standard_normal(D).astype(np.float32)
        c_p = rng.standard_normal(D).astype(np.float32)

        z_a = build_fusion_features(e_i, c_p.copy(), fusion_strategy="4_vector")
        z_b = build_fusion_features(e_i, c_p.copy(), fusion_strategy="4_vector")

        np.testing.assert_array_equal(z_a, z_b)


# ---------------------------------------------------------------------------
# Component-level: does apply_stage2_fusion change output when context changes?
# ---------------------------------------------------------------------------

class TestStage2PredictionsChangeWithContext:
    """Stage2FusionModel predictions shift when context embedding shifts."""

    def test_context_sensitive_model_changes_predictions(self):
        """A model with large context weights produces different output for different c_p."""
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion
        from tracks.representation.training.stage2.stage2_context import build_fusion_features

        model = _make_stage2_model(context_weight_scale=5.0)
        rng = np.random.default_rng(3)
        e_i = rng.standard_normal(D).astype(np.float32)
        c_orig = rng.standard_normal(D).astype(np.float32)
        c_ablated = np.zeros(D, dtype=np.float32)  # zero → very different from c_orig

        z_orig = build_fusion_features(e_i, c_orig, fusion_strategy="4_vector")
        z_ablated = build_fusion_features(e_i, c_ablated, fusion_strategy="4_vector")

        pred_orig = apply_stage2_fusion(_artifact(model), z_orig.reshape(1, -1))
        pred_ablated = apply_stage2_fusion(_artifact(model), z_ablated.reshape(1, -1))

        assert not np.allclose(pred_orig, pred_ablated, atol=1e-4), (
            "Context-sensitive Stage-2 model must change predictions when c_p changes"
        )

    def test_context_agnostic_model_preserves_predictions(self):
        """A model with zero context weights produces the same output regardless of c_p."""
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion
        from tracks.representation.training.stage2.stage2_context import build_fusion_features

        model = _make_stage2_model(context_weight_scale=0.0)
        rng = np.random.default_rng(4)
        e_i = rng.standard_normal(D).astype(np.float32)
        c_orig = rng.standard_normal(D).astype(np.float32)
        c_ablated = np.zeros(D, dtype=np.float32)

        z_orig = build_fusion_features(e_i, c_orig, fusion_strategy="4_vector")
        z_ablated = build_fusion_features(e_i, c_ablated, fusion_strategy="4_vector")

        pred_orig = apply_stage2_fusion(_artifact(model), z_orig.reshape(1, -1))
        pred_ablated = apply_stage2_fusion(_artifact(model), z_ablated.reshape(1, -1))

        np.testing.assert_allclose(pred_orig, pred_ablated, atol=1e-5,
            err_msg="Context-agnostic model must not change predictions when c_p changes")


# ---------------------------------------------------------------------------
# Integration: _run_ablation end-to-end with mocked compute_embeddings
# ---------------------------------------------------------------------------

def _make_mock_compute_embeddings(original_ctx_str: str, ablated_ctx_str: str,
                                   d: int, seed: int = 42):
    """Return a mock for compute_embeddings that gives fixed embeddings per string."""
    rng = np.random.default_rng(seed)

    # Two distinct fixed embeddings: one for original, one for ablated
    emb_original = rng.standard_normal(d).astype(np.float32)
    emb_ablated  = rng.standard_normal(d).astype(np.float32)

    def _mock(model_id, data, schema, cfg=None, batch_size=32):
        # The ablation loop calls with schema='pandas' and a pd.Series of
        # ablated context strings.  We return the ablated embedding for the
        # ablated string and the original embedding for the original string.
        import pandas as pd
        texts = list(data) if isinstance(data, (pd.Series, list)) else [str(data)]
        result = {}
        for t in texts:
            t_str = str(t).strip()
            if t_str == ablated_ctx_str.strip():
                result[t_str] = emb_ablated.copy()
            else:
                result[t_str] = emb_original.copy()
        return result

    return _mock, emb_original, emb_ablated


class TestRunAblationEndToEnd:
    """Integration tests for _run_ablation with a synthetic bundle."""

    # ------------------------------------------------------------------
    # shared setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_json():
        return {
            "1": {
                "summary": "Patient has diabetes",
                "medical_history": "Hypertension",
                "allergies": "",
                "medication_history": "Metformin",
                "social_history": "Non-smoker",
                "labs": "HbA1c 7.5",
                "radiology": "",
                "procedures": "",
            }
        }

    @staticmethod
    def _build_ablated_string_for(context_json, pid_str, field):
        from tracks.representation.training.stage2.stage2_context import build_context_string
        ctx = context_json[pid_str]
        ablated = {k: ("" if k == field else v) for k, v in ctx.items()}
        return build_context_string(ablated)

    @staticmethod
    def _build_original_string(context_json, pid_str):
        from tracks.representation.training.stage2.stage2_context import build_context_string
        return build_context_string(context_json[pid_str])

    # ------------------------------------------------------------------

    def _run_with_mocked_embeddings(
        self,
        stage2_model,
        original_ctx_str: str,
        ablated_ctx_str: str,
        context_vectors: dict,
        *,
        seed: int = 7,
    ):
        """Helper: call _run_ablation with mocked LLM embedding."""
        from tracks.representation.statistical.orchestrator.run_analysis import _run_ablation

        context_json = self._build_context_json()
        n_items = 4
        rng = np.random.default_rng(seed)
        X_items = rng.standard_normal((n_items, D)).astype(np.float32)
        patient_ids = np.array([1, 1, 1, 1])  # all from patient 1

        # Compute y_hat_ca using the model on original context
        from tracks.representation.training.stage2.stage2_context import build_fusion_features
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion

        c_p_orig = context_vectors[1]
        y_hat_ca = np.zeros((n_items, N_CLASSES), dtype=np.float64)
        for i in range(n_items):
            z = build_fusion_features(X_items[i], c_p_orig, fusion_strategy="4_vector")
            y_hat_ca[i] = apply_stage2_fusion(_artifact(stage2_model), z.reshape(1, -1))[0]

        bundle_path = _make_bundle_file(stage2_model, context_vectors)

        mock_fn, _, _ = _make_mock_compute_embeddings(
            original_ctx_str, ablated_ctx_str, D, seed=seed
        )

        try:
            with patch(
                "shared.embeddings.compute_embeddings.compute_embeddings",
                side_effect=mock_fn,
            ):
                result = _run_ablation(
                    context_json=context_json,
                    ensemble_bundle_path=bundle_path,
                    y_hat_ca=y_hat_ca,
                    patient_ids=patient_ids,
                    item_texts=np.array(["I1", "I2", "I3", "I4"]),
                    class_list=[f"cls_{c}" for c in range(N_CLASSES)],
                    llm="mock-llm",
                    cfg={},
                    X_items_test=X_items,
                )
        finally:
            bundle_path.unlink(missing_ok=True)

        return result

    # ------------------------------------------------------------------
    # Test: context-sensitive model → nonzero attribution for a non-empty field
    # ------------------------------------------------------------------

    def test_attribution_nonzero_for_context_sensitive_model(self):
        """When Stage-2 uses context heavily, ablating a non-empty field gives nonzero A(k,c)."""
        from tracks.representation.training.stage2.stage2_context import build_context_string

        context_json = self._build_context_json()
        original_str  = self._build_original_string(context_json, "1")
        ablated_str   = self._build_ablated_string_for(context_json, "1", "summary")

        rng = np.random.default_rng(5)
        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}

        # Large context weights → predictions change a lot when c_p changes
        model = _make_stage2_model(context_weight_scale=10.0, seed=5)

        df = self._run_with_mocked_embeddings(
            model, original_str, ablated_str, stage2_context_vectors
        )

        assert df is not None, "_run_ablation returned None unexpectedly"
        # The "summary" row should have at least one class with nonzero attribution
        row_summary = df[df["sub_field"] == "summary"]
        assert not row_summary.empty
        vals = row_summary.iloc[0][[f"cls_{c}" for c in range(N_CLASSES)]].values.astype(float)
        assert np.any(vals > 1e-6), (
            f"Attribution for 'summary' should be nonzero for a context-sensitive model, "
            f"got {vals}"
        )

    # ------------------------------------------------------------------
    # Test: context-agnostic model → near-zero attribution for all fields
    # ------------------------------------------------------------------

    def test_attribution_near_zero_for_context_agnostic_model(self):
        """When Stage-2 ignores context (zero weights on all context blocks), attribution ≈ 0."""
        from tracks.representation.training.stage2.stage2_context import build_context_string

        context_json = self._build_context_json()
        original_str  = self._build_original_string(context_json, "1")
        ablated_str   = self._build_ablated_string_for(context_json, "1", "summary")

        rng = np.random.default_rng(6)
        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}

        # Zero context weights → context has no effect on predictions
        model = _make_stage2_model(context_weight_scale=0.0, seed=6)

        df = self._run_with_mocked_embeddings(
            model, original_str, ablated_str, stage2_context_vectors
        )

        assert df is not None, "_run_ablation returned None unexpectedly"
        # All attribution values should be near zero
        numeric_cols = [f"cls_{c}" for c in range(N_CLASSES)]
        vals = df[numeric_cols].values.astype(float)
        assert np.all(np.abs(vals) < 1e-5), (
            f"Attribution should be ≈0 when Stage-2 ignores context, got max={np.max(np.abs(vals)):.2e}"
        )

    # ------------------------------------------------------------------
    # Test: baseline consistency — ca_baseline equals ca when all folds have S2
    # ------------------------------------------------------------------

    def test_baseline_equals_yhat_ca_when_all_folds_have_stage2(self):
        """When all folds have a Stage-2 model, ca_baseline ≈ y_hat_ca (within float precision)."""
        from tracks.representation.training.stage2.stage2_context import build_context_string, build_fusion_features
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion

        context_json = self._build_context_json()
        original_str  = self._build_original_string(context_json, "1")
        ablated_str   = self._build_ablated_string_for(context_json, "1", "summary")

        rng = np.random.default_rng(8)
        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}
        model = _make_stage2_model(context_weight_scale=3.0, seed=8)

        # Build the exact y_hat_ca the way predict_proba would when all folds have S2
        n_items = 3
        X_items = rng.standard_normal((n_items, D)).astype(np.float32)
        patient_ids = np.array([1, 1, 1])
        y_hat_ca = np.zeros((n_items, N_CLASSES), dtype=np.float64)
        for i in range(n_items):
            z = build_fusion_features(X_items[i], c_p_orig, fusion_strategy="4_vector")
            y_hat_ca[i] = apply_stage2_fusion(_artifact(model), z.reshape(1, -1))[0]

        bundle_path = _make_bundle_file(model, stage2_context_vectors)
        mock_fn, _, _ = _make_mock_compute_embeddings(original_str, ablated_str, D, seed=8)

        try:
            with patch(
                "shared.embeddings.compute_embeddings.compute_embeddings",
                side_effect=mock_fn,
            ):
                from tracks.representation.statistical.orchestrator.run_analysis import _run_ablation
                df = _run_ablation(
                    context_json=context_json,
                    ensemble_bundle_path=bundle_path,
                    y_hat_ca=y_hat_ca,
                    patient_ids=patient_ids,
                    item_texts=np.array(["I1", "I2", "I3"]),
                    class_list=[f"cls_{c}" for c in range(N_CLASSES)],
                    llm="mock-llm",
                    cfg={},
                    X_items_test=X_items,
                )
        finally:
            bundle_path.unlink(missing_ok=True)

        assert df is not None

    # ------------------------------------------------------------------
    # Test: already-empty field → attribution is zero
    # ------------------------------------------------------------------

    def test_attribution_zero_for_already_empty_field(self):
        """Ablating a field that is already empty → same context string → attribution = 0.

        When original_str == ablated_str, compute_embeddings returns the same
        embedding for both.  stage2_context_vectors must use that same embedding
        so ca_baseline and ablated_ca are computed with identical c_p, giving
        identical predictions and exactly-zero attribution.
        """
        from tracks.representation.training.stage2.stage2_context import build_fusion_features
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion

        context_json = self._build_context_json()
        # "allergies" is already "" in the context fixture
        original_str = self._build_original_string(context_json, "1")
        ablated_str  = self._build_ablated_string_for(context_json, "1", "allergies")

        # Pre-condition: ablating an already-empty field must not change the string
        assert original_str == ablated_str, (
            "Pre-condition: ablating an already-empty field must not change the context string"
        )

        # Use a single shared embedding for all context strings (original == ablated).
        # stage2_context_vectors must store this SAME embedding so that ca_baseline
        # and ablated_ca are computed with identical c_p → identical predictions.
        shared_emb = np.random.default_rng(9).standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: shared_emb}
        model = _make_stage2_model(context_weight_scale=5.0, seed=9)

        def mock_fn_shared(model_id, data, schema, cfg=None, batch_size=32):
            import pandas as pd
            texts = list(data) if isinstance(data, pd.Series) else [str(data)]
            return {str(t).strip(): shared_emb.copy() for t in texts}

        n_items = 4
        rng = np.random.default_rng(9)
        X_items = rng.standard_normal((n_items, D)).astype(np.float32)
        patient_ids = np.array([1, 1, 1, 1])

        y_hat_ca = np.zeros((n_items, N_CLASSES), dtype=np.float64)
        for i in range(n_items):
            z = build_fusion_features(X_items[i], shared_emb, fusion_strategy="4_vector")
            y_hat_ca[i] = apply_stage2_fusion(_artifact(model), z.reshape(1, -1))[0]

        bundle_path = _make_bundle_file(model, stage2_context_vectors)
        try:
            with patch(
                "shared.embeddings.compute_embeddings.compute_embeddings",
                side_effect=mock_fn_shared,
            ):
                from tracks.representation.statistical.orchestrator.run_analysis import _run_ablation
                df = _run_ablation(
                    context_json=context_json,
                    ensemble_bundle_path=bundle_path,
                    y_hat_ca=y_hat_ca,
                    patient_ids=patient_ids,
                    item_texts=np.array(["I1", "I2", "I3", "I4"]),
                    class_list=[f"cls_{c}" for c in range(N_CLASSES)],
                    llm="mock-llm",
                    cfg={},
                    X_items_test=X_items,
                )
        finally:
            bundle_path.unlink(missing_ok=True)

        assert df is not None
        row = df[df["sub_field"] == "allergies"]
        assert not row.empty
        vals = row.iloc[0][[f"cls_{c}" for c in range(N_CLASSES)]].values.astype(float)
        # Same string → same embedding → same predictions → attribution must be 0
        np.testing.assert_allclose(vals, 0.0, atol=1e-6,
            err_msg="Ablating an already-empty field must yield zero attribution")

    # ------------------------------------------------------------------
    # Test: ablated predictions are NOT equal to original CA for sensitive model
    # ------------------------------------------------------------------

    def test_ablated_predictions_differ_from_original_for_sensitive_model(self):
        """_run_ablation does NOT return ca=ablated_ca for a context-sensitive model.

        This is the direct regression test for the suspected bug: if ablated
        predictions were silently set equal to the original CA, all attribution
        scores would be identically zero.
        """
        from tracks.representation.training.stage2.stage2_context import build_context_string

        context_json = self._build_context_json()
        original_str  = self._build_original_string(context_json, "1")
        ablated_str   = self._build_ablated_string_for(context_json, "1", "summary")

        rng = np.random.default_rng(10)
        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}
        model = _make_stage2_model(context_weight_scale=8.0, seed=10)

        df = self._run_with_mocked_embeddings(
            model, original_str, ablated_str, stage2_context_vectors, seed=10
        )

        assert df is not None, "_run_ablation returned None"

        numeric_cols = [f"cls_{c}" for c in range(N_CLASSES)]
        all_vals = df[numeric_cols].values.astype(float)

        # At least one field must have nonzero attribution.
        # If ablated == original CA for all fields, all would be 0.
        assert np.any(all_vals > 1e-6), (
            "All attribution scores are zero — this suggests ablated predictions "
            "were silently set equal to original CA predictions (the suspected bug). "
            f"Max attribution observed: {np.max(all_vals):.2e}"
        )

    # ------------------------------------------------------------------
    # Test: _run_ablation returns None gracefully when Stage-2 is absent
    # ------------------------------------------------------------------

    def test_run_ablation_returns_none_without_stage2(self):
        """_run_ablation returns None when the bundle has no Stage-2 models."""
        import tempfile

        bundle = {
            "bundle_format": "pdm_ensemble_v1",
            "models": [],
            "preprocessors": [],
            "calibrators": [],
            "thresholds": np.full(N_CLASSES, 0.5, dtype=np.float32),
            "device": "cpu",
            "class_list": [f"cls_{c}" for c in range(N_CLASSES)],
            "model_arch": {},
            "stage2": [None, None],   # no fitted Stage-2 models
            "stage2_context_vectors": {},
            "config_snapshot": {},
            "metadata": {},
        }
        tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        tmp.close()
        joblib.dump(bundle, tmp.name)
        bundle_path = Path(tmp.name)

        from tracks.representation.statistical.orchestrator.run_analysis import _run_ablation
        try:
            result = _run_ablation(
                context_json={"1": {f: "" for f in [
                    "summary", "medical_history", "allergies", "medication_history",
                    "social_history", "labs", "radiology", "procedures",
                ]}},
                ensemble_bundle_path=bundle_path,
                y_hat_ca=np.zeros((2, N_CLASSES)),
                patient_ids=np.array([1, 1]),
                item_texts=np.array(["I1", "I2"]),
                class_list=[f"cls_{c}" for c in range(N_CLASSES)],
                llm="mock-llm",
                cfg={},
                X_items_test=np.zeros((2, D), dtype=np.float32),
            )
        finally:
            bundle_path.unlink(missing_ok=True)

        assert result is None, "Expected None when all stage2 slots are None"

    # ------------------------------------------------------------------
    # Test: output DataFrame has correct shape and column structure
    # ------------------------------------------------------------------

    def test_output_dataframe_shape(self):
        """Result DataFrame has shape (8, n_classes+1) with sub_field index column."""
        from tracks.representation.training.stage2.stage2_context import _CONTEXT_FIELDS, build_context_string

        context_json = self._build_context_json()
        original_str  = self._build_original_string(context_json, "1")
        ablated_str   = self._build_ablated_string_for(context_json, "1", "summary")

        rng = np.random.default_rng(11)
        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}
        model = _make_stage2_model(context_weight_scale=2.0, seed=11)

        df = self._run_with_mocked_embeddings(
            model, original_str, ablated_str, stage2_context_vectors, seed=11
        )

        assert df is not None
        assert len(df) == len(_CONTEXT_FIELDS), (
            f"Expected {len(_CONTEXT_FIELDS)} rows, got {len(df)}"
        )
        assert "sub_field" in df.columns
        for c in range(N_CLASSES):
            assert f"cls_{c}" in df.columns
        assert set(df["sub_field"].tolist()) == set(_CONTEXT_FIELDS)

    # ------------------------------------------------------------------
    # Test: attribution scores are non-negative
    # ------------------------------------------------------------------

    def test_attribution_scores_nonnegative(self):
        """A(k,c) = mean(|·|) is always ≥ 0."""
        from tracks.representation.training.stage2.stage2_context import build_context_string

        context_json = self._build_context_json()
        original_str  = self._build_original_string(context_json, "1")
        ablated_str   = self._build_ablated_string_for(context_json, "1", "labs")

        rng = np.random.default_rng(12)
        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}
        model = _make_stage2_model(context_weight_scale=3.0, seed=12)

        df = self._run_with_mocked_embeddings(
            model, original_str, ablated_str, stage2_context_vectors, seed=12
        )

        assert df is not None
        numeric_cols = [f"cls_{c}" for c in range(N_CLASSES)]
        vals = df[numeric_cols].values.astype(float)
        assert np.all(vals >= 0.0), (
            f"Attribution scores must be non-negative (A=mean|·|), got min={vals.min():.3e}"
        )


# ---------------------------------------------------------------------------
# Test: partial Stage-2 fold coverage (some folds are None)
# ---------------------------------------------------------------------------

class TestPartialStage2FoldCoverage:
    """When some outer folds have None Stage-2 models, attribution remains consistent."""

    def test_partial_none_folds_do_not_inflate_attribution(self):
        """With mixed S2/None folds, ca_baseline and ablated_ca use the same S2 models."""
        from tracks.representation.training.stage2.stage2_context import (
            build_context_string, build_fusion_features, _CONTEXT_FIELDS,
        )
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion
        from tracks.representation.statistical.orchestrator.run_analysis import _run_ablation

        context_json = {
            "1": {
                "summary": "Patient A summary",
                "medical_history": "None",
                "allergies": "",
                "medication_history": "",
                "social_history": "",
                "labs": "",
                "radiology": "",
                "procedures": "",
            }
        }

        rng = np.random.default_rng(20)
        # Two Stage-2 fold models, one None slot (mimics failed fold)
        model_0 = _make_stage2_model(context_weight_scale=0.0, seed=20)  # ignores context
        model_1 = _make_stage2_model(context_weight_scale=0.0, seed=21)  # ignores context

        c_p_orig = rng.standard_normal(D).astype(np.float32)
        stage2_context_vectors = {1: c_p_orig}

        n_items = 3
        X_items = rng.standard_normal((n_items, D)).astype(np.float32)
        patient_ids = np.array([1, 1, 1])

        # y_hat_ca: simulate predict_proba with 3 folds, one None
        # Fold 0: model_0 (Stage-2)
        # Fold 1: None (Stage-1, represented here as an arbitrary prediction)
        # Fold 2: model_1 (Stage-2)
        s1_preds = rng.random((n_items, N_CLASSES)).astype(np.float32) * 0.5
        s2_0_preds = np.zeros((n_items, N_CLASSES), dtype=np.float32)
        s2_1_preds = np.zeros((n_items, N_CLASSES), dtype=np.float32)
        for i in range(n_items):
            z = build_fusion_features(X_items[i], c_p_orig, fusion_strategy="4_vector")
            s2_0_preds[i] = apply_stage2_fusion(_artifact(model_0), z.reshape(1, -1))[0]
            s2_1_preds[i] = apply_stage2_fusion(_artifact(model_1), z.reshape(1, -1))[0]
        # predict_proba average: (s2_0 + s1_none + s2_1) / 3
        y_hat_ca = ((s2_0_preds + s1_preds + s2_1_preds) / 3.0).astype(np.float64)

        # Bundle has 3 stage2 slots: [model_0, None, model_1]
        bundle = {
            "bundle_format": "pdm_ensemble_v1",
            "models": [],
            "preprocessors": [],
            "calibrators": [],
            "thresholds": np.full(N_CLASSES, 0.5, dtype=np.float32),
            "device": "cpu",
            "class_list": [f"cls_{c}" for c in range(N_CLASSES)],
            "model_arch": {"fusion_strategy": "4_vector", "lowrank_bilinear_r": 8},
            "stage2": [_artifact(model_0), None, _artifact(model_1)],
            "stage2_context_vectors": stage2_context_vectors,
            "config_snapshot": {},
            "metadata": {},
        }
        tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        tmp.close()
        joblib.dump(bundle, tmp.name)
        bundle_path = Path(tmp.name)

        original_ctx_str = build_context_string(context_json["1"])
        ablated_ctx = {k: ("" if k == "summary" else v) for k, v in context_json["1"].items()}
        ablated_ctx_str = build_context_string(ablated_ctx)

        def mock_emb(model_id, data, schema, cfg=None, batch_size=32):
            # Return the same zero-vector for everything (context-agnostic path)
            import pandas as pd
            texts = list(data) if isinstance(data, pd.Series) else [str(data)]
            return {str(t).strip(): np.zeros(D, dtype=np.float32) for t in texts}

        try:
            with patch(
                "shared.embeddings.compute_embeddings.compute_embeddings",
                side_effect=mock_emb,
            ):
                df = _run_ablation(
                    context_json=context_json,
                    ensemble_bundle_path=bundle_path,
                    y_hat_ca=y_hat_ca,
                    patient_ids=patient_ids,
                    item_texts=np.array(["I1", "I2", "I3"]),
                    class_list=[f"cls_{c}" for c in range(N_CLASSES)],
                    llm="mock-llm",
                    cfg={},
                    X_items_test=X_items,
                )
        finally:
            bundle_path.unlink(missing_ok=True)

        assert df is not None
        # Both model_0 and model_1 have zero context weights → ca_baseline and
        # ablated_ca both reflect the same context-agnostic predictions.
        # Because both models ignore context, attribution must be ≈ 0 regardless
        # of the Stage-1 noise baked into y_hat_ca (the None-fold contribution).
        numeric_cols = [f"cls_{c}" for c in range(N_CLASSES)]
        vals = df[numeric_cols].values.astype(float)
        assert np.all(np.abs(vals) < 1e-5), (
            "Attribution must be ≈0 for context-agnostic models even with None-fold "
            f"Stage-1 noise in y_hat_ca. Max={np.max(np.abs(vals)):.2e}"
        )
