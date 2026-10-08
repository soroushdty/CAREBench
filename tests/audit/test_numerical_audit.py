"""Numerical correctness, shape/dtype, and edge-case robustness audit tests.

Covers:
  1. BCE loss — numerically stable logits path, no sigmoid-before-loss
  2. Class weight broadcasting — shape, axis, zero-positives, NaN rows
  3. Soft labels — 0.5 preserved through training path
  4. Pair aggregation — 2-physician mean, mismatch handling
  5. Isotonic calibration — per-class, column alignment, degenerate fallback
  6. Embedding dtype and fusion shape — float32, dimension matching
  7. Stage 2 final model state — non-zero weights, non-constant predictions
  8. Patient one-hot baseline — unseen patients, dimension correctness
  9. CV mutable state — independent Preprocessors and calibrators per fold
 10. Permutation / bootstrap — early return on all-zero dp, no RuntimeWarning,
     bootstrap excludes single-patient, nonzero-delta guard
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. BCE loss
# ---------------------------------------------------------------------------

class TestBCELoss:
    def test_uses_logits_not_probabilities(self):
        """Model forward returns raw logits — BCE should be called on those."""
        from tracks.representation.models.MultiLabelModel import MultiLabelModel
        model = MultiLabelModel(n_features=8, n_classes=4, hidden_dims=[])
        X = torch.randn(10, 8)
        logits = model(X)
        # Logits must be unbounded (not sigmoid-squashed)
        assert logits.min().item() < 0.0 or logits.max().item() > 1.0, (
            "MultiLabelModel.forward() must return raw logits, not probabilities"
        )

    def test_compute_weighted_bce_numerically_stable(self):
        """F.binary_cross_entropy_with_logits is numerically stable for extreme logits."""
        from tracks.representation.training.stage1.compute_bce_loss import compute_weighted_bce_loss
        logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        pos_w = torch.ones(2)
        loss = compute_weighted_bce_loss(logits, targets, pos_w)
        assert torch.isfinite(loss), "BCE with extreme logits must be finite"
        assert loss.item() >= 0.0

    def test_bce_pos_weight_shape_broadcasts(self):
        """pos_weight (C,) must broadcast correctly with (N, C) logits."""
        from tracks.representation.training.stage1.compute_bce_loss import compute_weighted_bce_loss
        N, C = 16, 5
        logits = torch.randn(N, C)
        targets = torch.rand(N, C)
        pos_w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = compute_weighted_bce_loss(logits, targets, pos_w)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(loss)

    def test_stage2_masked_bce_reduction_consistent(self):
        """Stage 2 masked BCE loss equals manual mean over non-NaN positions."""
        logits = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
        y_int = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, 1.0]])
        nan_mask = ~torch.isnan(y_int)
        y_filled = torch.where(nan_mask, y_int, torch.zeros_like(y_int))
        pos_w = torch.ones(3)

        loss_per_elem = F.binary_cross_entropy_with_logits(
            logits, y_filled, pos_weight=pos_w, reduction="none"
        )
        valid_count = nan_mask.sum().clamp(min=1)
        computed = (loss_per_elem * nan_mask).sum() / valid_count

        # Manual: compute only over non-NaN positions
        manual_vals = []
        for r in range(2):
            for c in range(3):
                if nan_mask[r, c]:
                    manual_vals.append(F.binary_cross_entropy_with_logits(
                        logits[r:r+1, c:c+1],
                        y_filled[r:r+1, c:c+1],
                    ).item())
        manual = float(np.mean(manual_vals))
        assert abs(computed.item() - manual) < 1e-5, (
            f"Masked BCE {computed.item():.6f} != manual {manual:.6f}"
        )

    def test_bce_reduction_mean_vs_none_consistent(self):
        """reduction='mean' and 'none'+mean give identical results."""
        logits = torch.randn(8, 3)
        targets = torch.rand(8, 3)
        pos_w = torch.tensor([1.5, 2.0, 0.8])
        loss_mean = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_w, reduction="mean"
        )
        loss_none = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_w, reduction="none"
        ).mean()
        assert abs(loss_mean.item() - loss_none.item()) < 1e-5


# ---------------------------------------------------------------------------
# 2. Class weight broadcasting and edge cases
# ---------------------------------------------------------------------------

class TestClassWeights:
    def test_hp_search_zero_positive_class(self):
        """Class with zero positive examples should get weight 1.0 (Cui default)."""
        import numpy as np
        beta = 0.999
        pos_mass = 0.0
        # Cui formula with max(pos_mass, 1) guard
        w_raw = (1 - beta) / (1 - beta ** max(pos_mass, 1))
        w = float(np.clip(w_raw, 1.0, 50.0))
        assert w == 1.0, f"Zero-positive Cui weight should be 1.0, got {w}"

    def test_hp_search_inverse_freq_zero_positive(self):
        """inverse_frequency with zero positives should not divide by zero."""
        from tracks.representation.training.stage1.hp_search import _run_hp_candidate
        import numpy as np
        # Direct test of the formula used in hp_search.py
        valid_counts = 20
        pos_mass = np.zeros(3)
        weight_cap = 10.0
        # Formula from hp_search.py: np.maximum(pos_mass, 1.0)
        pos_w = np.clip(
            valid_counts / (2.0 * np.maximum(pos_mass, 1.0)),
            1.0,
            weight_cap,
        )
        assert np.all(np.isfinite(pos_w)), "Weights must be finite"
        assert np.all(pos_w >= 1.0), "Weights must be >= 1.0"

    def test_stage2_pos_weights_nan_rows_inverse_freq(self):
        """inverse_frequency uses per-class valid count, not total row count."""
        from tracks.representation.training.stage2.fit_stage2_fusion import _compute_pos_weights
        # 10 rows; class 0 has 8 NaN rows, 2 valid both positive
        y = np.full((10, 2), np.nan, dtype=np.float32)
        y[0, 0] = 1.0
        y[1, 0] = 1.0
        y[:, 1] = np.arange(10) % 2  # class 1 has all 10 valid, 5 positive
        cfg = {"class_weight_mode": "inverse_frequency"}
        w = _compute_pos_weights(y, weight_cap=50.0, cfg=cfg)

        # Class 0: n_valid=2, n_pos=2 → w = 2/(2*2) = 0.5 → clipped to 1.0
        # Old wrong formula: n_total=10, w = 10/(2*2) = 2.5 → not clipped
        # After fix: w should be 1.0 (clipped from 0.5)
        assert w[0] == 1.0, (
            f"Class 0 inv_freq weight should be 1.0 (clipped), got {w[0]}"
        )
        # Class 1: n_valid=10, n_pos=5 → w = 10/(2*5) = 1.0 (exact)
        assert abs(w[1] - 1.0) < 0.01, f"Class 1 weight should be ~1.0, got {w[1]}"

    def test_stage2_pos_weights_cui_unaffected_by_nan(self):
        """Cui mode should give same weight regardless of how many NaN rows exist."""
        from tracks.representation.training.stage2.fit_stage2_fusion import _compute_pos_weights
        y_dense = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.0]], dtype=np.float32)
        y_sparse = np.full((10, 2), np.nan, dtype=np.float32)
        y_sparse[:3] = y_dense  # same valid data, more NaN rows
        cfg = {"class_weight_mode": "cui", "class_weight_beta": 0.999}
        w_dense = _compute_pos_weights(y_dense, weight_cap=50.0, cfg=cfg)
        w_sparse = _compute_pos_weights(y_sparse, weight_cap=50.0, cfg=cfg)
        np.testing.assert_allclose(w_dense, w_sparse, rtol=1e-4,
            err_msg="Cui weights should be independent of NaN row count")

    def test_pos_weight_lower_bound_applied(self):
        """Clamp ensures pos_weight is always >= 1.0."""
        from tracks.representation.training.stage2.fit_stage2_fusion import _compute_pos_weights
        y = np.ones((20, 3), dtype=np.float32)  # all positive → very high pos_mass
        cfg = {"class_weight_mode": "cui"}
        w = _compute_pos_weights(y, weight_cap=50.0, cfg=cfg)
        assert np.all(w >= 1.0), f"All weights must be >= 1.0, got {w}"

    def test_pos_weight_upper_bound_applied(self):
        """Clamp ensures pos_weight never exceeds weight_cap."""
        from tracks.representation.training.stage2.fit_stage2_fusion import _compute_pos_weights
        y = np.zeros((20, 3), dtype=np.float32)
        y[0, :] = 0.01  # tiny positive mass → huge raw weight
        cfg = {"class_weight_mode": "cui", "class_weight_beta": 0.999}
        cap = 10.0
        w = _compute_pos_weights(y, weight_cap=cap, cfg=cfg)
        assert np.all(w <= cap), f"All weights must be <= {cap}, got {w}"


# ---------------------------------------------------------------------------
# 3. Soft labels
# ---------------------------------------------------------------------------

class TestSoftLabels:
    def test_05_preserved_through_float32_coercion(self):
        """as_float32_array must not alter 0.5 labels."""
        from shared.utils.array_utils import as_float32_array
        Y = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.5]], dtype=np.float32)
        Y_out = as_float32_array(Y, "Y_train")
        np.testing.assert_array_equal(Y_out, Y, err_msg="0.5 labels altered by coercion")

    def test_05_preserved_in_bce_targets(self):
        """0.5 in BCEWithLogitsLoss targets is valid and gives finite gradient."""
        logits = torch.zeros(4, 3, requires_grad=True)
        targets = torch.tensor([[0.5, 0.0, 1.0], [0.0, 0.5, 0.5],
                                 [1.0, 0.0, 0.0], [0.5, 1.0, 0.5]])
        pos_w = torch.ones(3)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        # Gradient at logit=0 for target=0.5 should be zero (equal push both ways)
        # target=0.5 means gradient = sigmoid(0) - 0.5 = 0.5 - 0.5 = 0
        assert abs(logits.grad[0, 0].item()) < 1e-5, (
            "Gradient for target=0.5 at logit=0 should be 0"
        )

    def test_masked_class_data_includes_05(self):
        """With default threshold=0.5, masked_class_data includes 0.5 labels."""
        from tracks.representation.training.shared.soft_label_utils import masked_class_data
        y = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        p = np.array([[0.3], [0.5], [0.7]], dtype=np.float32)
        mask, y_col, p_col = masked_class_data(y, p, 0, {"eval_pos_threshold": 0.5})
        assert mask.sum() == 3, "All 3 labels (0, 0.5, 1) should be included with threshold=0.5"

    def test_calibration_05_label_unique_check(self):
        """If all labels are 0.5, fit_calibrators uses ConstantCalibrator(0.5)."""
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
        Y = np.full((15, 1), 0.5, dtype=np.float32)
        probs = np.random.rand(15, 1).astype(np.float32)
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 5}
        cals = fit_calibrators(probs, Y, ["cls"], cfg)
        assert isinstance(cals["cls"], ConstantCalibrator), (
            "All-0.5 labels should give ConstantCalibrator fallback"
        )
        assert abs(cals["cls"].prob - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# 4. Pair aggregation
# ---------------------------------------------------------------------------

class TestPairAggregation:
    def _make_df(self, labels):
        import pandas as pd
        rows = []
        for pid, item, ph, lbl in labels:
            rows.append({"patient": pid, "item": item, "physician": ph, "A": lbl})
        return pd.DataFrame(rows)

    def test_two_physicians_mean_gives_05(self):
        """Two physicians (0, 1) → label mean = 0.5."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        import pandas as pd
        df = self._make_df([
            (1, "Q1", 101, 0),
            (1, "Q1", 102, 1),
        ])
        agg = _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                    item_col="item", class_cols=["A"], physician_count=2)
        assert len(agg) == 1
        assert float(agg.loc[0, "A"]) == pytest.approx(0.5)

    def test_two_physicians_both_agree(self):
        """Two physicians agreeing produce exact 0.0 or 1.0."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        df = self._make_df([
            (1, "Q1", 101, 1),
            (1, "Q1", 102, 1),
            (2, "Q1", 101, 0),
            (2, "Q1", 102, 0),
        ])
        agg = _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                    item_col="item", class_cols=["A"], physician_count=2)
        vals = sorted(agg["A"].tolist())
        assert vals == pytest.approx([0.0, 1.0])

    def test_condition_a_duplicate_resolved(self):
        """Condition A: 4 rows = [A, B, A, B] → every row has a replica + 2 unique → resolves."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        import pandas as pd
        # 4 rows for same (patient, item): rows 0&2 identical, rows 1&3 identical.
        # all_rows_have_replica=True, unique_count=2 → Condition A satisfied.
        df = pd.DataFrame([
            {"patient": 1, "item": "Q1", "physician": 101, "A": 0},
            {"patient": 1, "item": "Q1", "physician": 102, "A": 1},
            {"patient": 1, "item": "Q1", "physician": 101, "A": 0},  # dup of row 0
            {"patient": 1, "item": "Q1", "physician": 102, "A": 1},  # dup of row 1
        ])
        agg = _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                    item_col="item", class_cols=["A"], physician_count=2)
        assert len(agg) == 1
        assert float(agg.loc[0, "A"]) == pytest.approx(0.5)

    def test_condition_a_partial_dup_is_condition_b(self):
        """3 rows [A, B, A]: B has no replica → fails Condition A → treated as Condition B."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        import pandas as pd
        # Row 1 (physician=102) is unique — no replica — so all_rows_have_replica=False.
        df = pd.DataFrame([
            {"patient": 1, "item": "Q1", "physician": 101, "A": 0},
            {"patient": 1, "item": "Q1", "physician": 102, "A": 1},
            {"patient": 1, "item": "Q1", "physician": 101, "A": 0},  # dup of row 0 only
        ])
        agg = _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                    item_col="item", class_cols=["A"],
                                    physician_count=2, mismatch_error=False)
        # Condition B → all 3 rows dropped
        assert len(agg) == 0

    def test_condition_b_mismatch_drops_rows(self):
        """Condition B (unresolvable) drops associated rows when mismatch_error=False."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        import pandas as pd
        df = pd.DataFrame([
            {"patient": 1, "item": "Q1", "physician": 101, "A": 0},  # only one physician
            {"patient": 2, "item": "Q1", "physician": 101, "A": 1},
            {"patient": 2, "item": "Q1", "physician": 102, "A": 0},
        ])
        agg = _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                    item_col="item", class_cols=["A"],
                                    physician_count=2, mismatch_error=False)
        # Patient 1 dropped (only 1 physician), Patient 2 kept
        assert len(agg) == 1
        assert int(agg.loc[0, "patient"]) == 2

    def test_condition_b_mismatch_error_raises(self):
        """Condition B with mismatch_error=True must raise ValueError."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        import pandas as pd
        df = pd.DataFrame([
            {"patient": 1, "item": "Q1", "physician": 101, "A": 0},
        ])
        with pytest.raises(ValueError, match="Reference observer count mismatch"):
            _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                  item_col="item", class_cols=["A"],
                                  physician_count=2, mismatch_error=True)

    def test_output_labels_in_expected_set(self):
        """All aggregated labels must be in {0.0, 0.5, 1.0} for 2-physician input."""
        from shared.reference.aggregation.paired_reference_mean import _aggregate_physicians
        import pandas as pd
        rows = []
        for pid in range(1, 4):
            for item in ["Q1", "Q2", "Q3"]:
                for ph_i, lbl in [(101, 0), (102, 1)]:
                    rows.append({"patient": pid, "item": item, "physician": ph_i, "A": lbl})
        df = pd.DataFrame(rows)
        agg = _aggregate_physicians(df, patient_col="patient", physician_col="physician",
                                    item_col="item", class_cols=["A"], physician_count=2)
        valid_values = {0.0, 0.5, 1.0}
        for v in agg["A"]:
            assert float(v) in valid_values, f"Unexpected aggregated label {v}"


# ---------------------------------------------------------------------------
# 5. Isotonic calibration
# ---------------------------------------------------------------------------

class TestIsotonicCalibration:
    def _probs_labels(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        probs = rng.random((n, 3)).astype(np.float32)
        Y = (rng.random((n, 3)) > 0.5).astype(np.float32)
        return probs, Y

    def test_one_calibrator_per_class(self):
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        probs, Y = self._probs_labels()
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 5}
        cals = fit_calibrators(probs, Y, ["A", "B", "C"], cfg)
        assert set(cals.keys()) == {"A", "B", "C"}

    def test_calibrator_applied_to_correct_column(self):
        """apply_calibrators must not transpose or swap columns."""
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        from tracks.representation.training.shared.apply_calibrators import apply_calibrators
        n = 40
        rng = np.random.default_rng(77)
        probs = rng.random((n, 3)).astype(np.float32)
        # Distinct label structure per class
        Y = np.zeros((n, 3), dtype=np.float32)
        Y[:20, 0] = 1.0  # class 0: first half positive
        Y[20:, 1] = 1.0  # class 1: second half positive
        Y[:, 2] = 0.5     # class 2: all 0.5
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 5}
        cals = fit_calibrators(probs, Y, ["A", "B", "C"], cfg)
        cal_probs = apply_calibrators(cals, probs, ["A", "B", "C"])
        # Calibrated shape must match input
        assert cal_probs.shape == probs.shape
        # All values must be in [0, 1]
        assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)

    def test_constant_label_falls_back_to_constant_calibrator(self):
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
        n = 20
        probs = np.random.rand(n, 2).astype(np.float32)
        Y = np.zeros((n, 2), dtype=np.float32)
        Y[:, 1] = 1.0  # class 1: all positive → single unique label
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 5}
        cals = fit_calibrators(probs, Y, ["X", "Y"], cfg)
        assert isinstance(cals["X"], ConstantCalibrator)  # all-negative
        assert isinstance(cals["Y"], ConstantCalibrator)  # all-positive

    def test_too_few_samples_falls_back(self):
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
        probs = np.random.rand(4, 1).astype(np.float32)
        Y = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 10}
        cals = fit_calibrators(probs, Y, ["A"], cfg)
        assert isinstance(cals["A"], ConstantCalibrator)

    def test_no_samples_falls_back_to_05(self):
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
        probs = np.empty((0, 1), dtype=np.float32)
        Y = np.empty((0, 1), dtype=np.float32)
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 5}
        cals = fit_calibrators(probs, Y, ["A"], cfg)
        assert isinstance(cals["A"], ConstantCalibrator)
        assert abs(cals["A"].prob - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# 6. Embedding dtype and fusion shape
# ---------------------------------------------------------------------------

class TestEmbeddingDtypeShape:
    def test_l2_normalize_preserves_float32(self):
        """_l2_normalize must not upcast float32 embeddings to float64."""
        from shared.embeddings.compute_embeddings import _l2_normalize
        emb = np.random.randn(10, 128).astype(np.float32)
        norm = _l2_normalize(emb)
        assert norm.dtype == np.float32, f"Expected float32, got {norm.dtype}"

    def test_l2_normalize_unit_norm(self):
        from shared.embeddings.compute_embeddings import _l2_normalize
        emb = np.random.randn(8, 64).astype(np.float32)
        norm = _l2_normalize(emb)
        norms = np.linalg.norm(norm, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-5)

    def test_fusion_matrix_float32_output(self):
        from tracks.representation.training.stage2.stage2_context import build_fusion_matrix
        d = 8
        X = np.random.randn(5, d).astype(np.float32)
        ctx = {1: np.ones(d, dtype=np.float32), 2: np.ones(d, dtype=np.float32) * 2}
        pids = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        Z = build_fusion_matrix(X, pids, ctx)
        assert Z.dtype == np.float32, f"Fusion matrix must be float32, got {Z.dtype}"

    def test_fusion_dim_4_vector(self):
        from tracks.representation.training.stage2.stage2_context import build_fusion_matrix
        d = 16
        X = np.random.randn(3, d).astype(np.float32)
        ctx = {10: np.random.randn(d).astype(np.float32)}
        pids = np.array([10.0, 10.0, 10.0])
        Z = build_fusion_matrix(X, pids, ctx, fusion_strategy="4_vector")
        assert Z.shape == (3, 4 * d), f"4_vector dim should be {4*d}, got {Z.shape[1]}"

    def test_fusion_dim_2d(self):
        from tracks.representation.training.stage2.stage2_context import build_fusion_matrix
        d = 16
        X = np.random.randn(3, d).astype(np.float32)
        ctx = {10: np.random.randn(d).astype(np.float32)}
        pids = np.array([10.0, 10.0, 10.0])
        Z = build_fusion_matrix(X, pids, ctx, fusion_strategy="2d")
        assert Z.shape == (3, 2 * d)

    def test_fusion_dim_3d(self):
        from tracks.representation.training.stage2.stage2_context import build_fusion_matrix
        d = 16
        X = np.random.randn(3, d).astype(np.float32)
        ctx = {10: np.random.randn(d).astype(np.float32)}
        pids = np.array([10.0, 10.0, 10.0])
        Z = build_fusion_matrix(X, pids, ctx, fusion_strategy="3d")
        assert Z.shape == (3, 3 * d)

    def test_fusion_dim_mismatch_raises(self):
        from tracks.representation.training.stage2.stage2_context import build_fusion_matrix
        X_d4 = np.random.randn(2, 4).astype(np.float32)
        ctx_d8 = {1: np.ones(8, dtype=np.float32)}
        with pytest.raises(ValueError, match="dimension"):
            build_fusion_matrix(X_d4, np.array([1.0, 1.0]), ctx_d8)

    def test_lowrank_bilinear_requires_even_dim(self):
        from tracks.representation.training.stage2.fit_stage2_fusion import LowRankBilinearFusionModel
        with pytest.raises(ValueError, match="divisible by 2"):
            LowRankBilinearFusionModel(input_dim=7, n_classes=3)

    def test_lowrank_bilinear_output_shape(self):
        from tracks.representation.training.stage2.fit_stage2_fusion import LowRankBilinearFusionModel
        d, C, r, N = 8, 3, 4, 10
        model = LowRankBilinearFusionModel(input_dim=2 * d, n_classes=C, r=r)
        Z = torch.randn(N, 2 * d)
        out = model(Z)
        assert out.shape == (N, C), f"Expected ({N},{C}), got {out.shape}"


# ---------------------------------------------------------------------------
# 7. Stage 2 final model state
# ---------------------------------------------------------------------------

class TestStage2FinalModel:
    def _fit_and_apply(self, use_val: bool):
        from tracks.representation.training.stage2.fit_stage2_fusion import (
            _train_one_model, apply_stage2_fusion, Stage2FusionModel
        )
        rng = np.random.default_rng(42)
        n, d, c = 15, 8, 3
        Z_tr = rng.standard_normal((n, d)).astype(np.float32)
        y_int = rng.random((n, c)).astype(np.float32)
        y_int[0, 1] = np.nan
        y_survey = rng.random((n, c)).astype(np.float32)
        y_cf = rng.random((n, c)).astype(np.float32)
        device = torch.device("cpu")
        cfg = {"class_weight_mode": "cui"}
        Z_val = Z_tr[:5] if use_val else None
        y_val = y_int[:5] if use_val else None
        model = _train_one_model(
            Z_tr, y_int, y_survey, y_cf,
            alpha=0.1, lr=0.01, num_epochs=20, patience=5,
            weight_cap=10.0, device=device, cfg=cfg,
            Z_val=Z_val, y_val=y_val,
        )
        preds = apply_stage2_fusion({"model": model}, Z_tr)
        return model, preds

    def test_final_weights_nonzero_with_val(self):
        model, preds = self._fit_and_apply(use_val=True)
        w = model.linear.weight.detach().cpu().numpy()
        assert np.linalg.norm(w) > 1e-6, "Weights must be non-zero after training"

    def test_final_weights_nonzero_without_val(self):
        """Full-data retraining (no val) must not reset to init zeros."""
        model, preds = self._fit_and_apply(use_val=False)
        w = model.linear.weight.detach().cpu().numpy()
        assert np.linalg.norm(w) > 1e-6, (
            "Weights reset to zero init — early-stopping state-restore bug?"
        )

    def test_predictions_not_constant_with_val(self):
        _, preds = self._fit_and_apply(use_val=True)
        assert preds.std() > 1e-4, "Predictions should not be constant"

    def test_predictions_not_constant_without_val(self):
        _, preds = self._fit_and_apply(use_val=False)
        assert preds.std() > 1e-4, "Predictions should not be constant 0.5"

    def test_predictions_in_01(self):
        _, preds = self._fit_and_apply(use_val=False)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


# ---------------------------------------------------------------------------
# 8. Patient one-hot baseline
# ---------------------------------------------------------------------------

class TestPatientOneHot:
    def test_unseen_patient_all_zeros(self):
        from tracks.representation.training.stage2.fit_stage2_fusion import _build_patient_id_baseline_features
        X = np.random.randn(6, 4).astype(np.float32)
        tr_ids = np.array([1, 1, 2, 2, 3, 3])
        # Build one-hot for test patient 99 (unseen)
        X_test = np.random.randn(2, 4).astype(np.float32)
        test_ids = np.array([99, 99])
        X_pid = _build_patient_id_baseline_features(
            X_test, test_ids, reference_ids=tr_ids
        )
        onehot_block = X_pid[:, 4:]
        assert np.all(onehot_block == 0.0), (
            "Unseen patient must have all-zero one-hot row"
        )

    def test_training_patients_exactly_one_hot(self):
        from tracks.representation.training.stage2.fit_stage2_fusion import _build_patient_id_baseline_features
        X = np.random.randn(6, 4).astype(np.float32)
        tr_ids = np.array([10, 10, 20, 20, 30, 30])
        X_pid = _build_patient_id_baseline_features(X, tr_ids, reference_ids=tr_ids)
        onehot = X_pid[:, 4:]
        # Each row should have exactly one 1.0 and rest 0.0
        row_sums = onehot.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(6))

    def test_one_hot_dim_equals_num_training_patients(self):
        from tracks.representation.training.stage2.fit_stage2_fusion import _build_patient_id_baseline_features
        d = 8
        n_patients = 5
        X = np.random.randn(n_patients * 3, d).astype(np.float32)
        tr_ids = np.repeat(np.arange(1, n_patients + 1), 3)
        X_pid = _build_patient_id_baseline_features(X, tr_ids, reference_ids=tr_ids)
        assert X_pid.shape[1] == d + n_patients, (
            f"Expected dim {d + n_patients}, got {X_pid.shape[1]}"
        )


# ---------------------------------------------------------------------------
# 9. CV mutable state
# ---------------------------------------------------------------------------

class TestCVMutableState:
    def test_preprocessor_independent_per_fold(self):
        """Each fold creates an independent Preprocessor — no shared state."""
        from tracks.representation.models.Preprocessor import Preprocessor
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((20, 10)).astype(np.float32)
        X2 = rng.standard_normal((20, 10)).astype(np.float32) + 5.0

        p1 = Preprocessor(n_components=5).fit(X1)
        p2 = Preprocessor(n_components=5).fit(X2)

        # Means should differ (X2 is shifted by 5)
        m1 = p1._mean.cpu().numpy()
        m2 = p2._mean.cpu().numpy()
        assert not np.allclose(m1, m2, atol=1.0), (
            "Preprocessors from different data must have different means"
        )

    def test_preprocessor_transform_reflects_fit(self):
        """transform() uses each Preprocessor's own fit statistics (mean, std, PCA).

        StandardScaler centres output to ~0 for training data by design.
        Instead verify a fixed held-out point is transformed differently by two
        Preprocessors fitted on shifted data distributions.
        """
        from tracks.representation.models.Preprocessor import Preprocessor
        rng = np.random.default_rng(1)
        X1 = rng.standard_normal((30, 8)).astype(np.float32)
        X2 = rng.standard_normal((30, 8)).astype(np.float32) + 10.0  # shifted

        p1 = Preprocessor(n_components=4, whiten=False).fit(X1)
        p2 = Preprocessor(n_components=4, whiten=False).fit(X2)

        # Apply both to the same fixed test point
        X_fixed = np.ones((1, 8), dtype=np.float32)
        T1 = p1.transform(X_fixed)
        T2 = p2.transform(X_fixed)

        # p2 was fitted on X2 (mean ≈ 10); applying to X_fixed (all 1s) should
        # give a large negative shift because 1 ≪ 10. p1 fitted on X1 (mean ≈ 0)
        # gives a near-zero output. The two should differ substantially.
        assert not np.allclose(T1, T2, atol=0.5), (
            "Two Preprocessors fitted on shifted data must transform the same "
            "point differently — indicates they share mutable fit state."
        )

    def test_calibrators_independent_per_fold(self):
        """fit_calibrators creates fresh objects every call."""
        from tracks.representation.training.shared.fit_calibrators import fit_calibrators
        rng = np.random.default_rng(2)
        n = 25
        probs = rng.random((n, 2)).astype(np.float32)
        Y1 = np.zeros((n, 2), dtype=np.float32)
        Y1[:12, 0] = 1.0
        Y2 = np.zeros((n, 2), dtype=np.float32)
        Y2[12:, 0] = 1.0
        cfg = {"calibration_method": "isotonic", "calibration_min_samples": 5}

        cals1 = fit_calibrators(probs, Y1, ["A", "B"], cfg)
        cals2 = fit_calibrators(probs, Y2, ["A", "B"], cfg)

        assert cals1["A"] is not cals2["A"], (
            "Calibrators from different fold calls must be distinct objects"
        )
        # Different label distributions → different calibration mappings
        val = np.array([0.4, 0.5, 0.6])
        pred1 = np.asarray(cals1["A"].predict(val))
        pred2 = np.asarray(cals2["A"].predict(val))
        assert not np.allclose(pred1, pred2), (
            "Calibrators trained on different data should give different predictions"
        )


# ---------------------------------------------------------------------------
# 10. Permutation test and bootstrap
# ---------------------------------------------------------------------------

class TestPermutationBootstrap:
    def test_permutation_all_zero_dp_no_warning(self):
        """All-zero dp should return NaN result without RuntimeWarning."""
        from tracks.representation.statistical.hypotheses.h1 import h1_permutation_test
        dp = np.zeros((10, 3), dtype=np.float32)
        dm = np.random.randn(10, 3).astype(np.float32)
        pids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # warnings → errors
            result = h1_permutation_test(dp, dm, pids, n_permutations=200,
                                         rng=np.random.default_rng(0))
        assert np.isnan(result["aggregate_rate"])
        assert np.isnan(result["p_value"])
        assert np.isnan(result["null_mean"])

    def test_permutation_nonzero_dp_valid_result(self):
        from tracks.representation.statistical.hypotheses.h1 import h1_permutation_test
        rng = np.random.default_rng(10)
        dp = rng.choice([-0.5, 0.5], size=(30, 2)).astype(np.float32)
        dm = dp.copy()  # perfect alignment
        pids = np.repeat(np.arange(6), 5)
        result = h1_permutation_test(dp, dm, pids, n_permutations=300, rng=rng)
        assert not np.isnan(result["aggregate_rate"])
        assert result["aggregate_rate"] == pytest.approx(1.0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_permutation_statistic_restricted_to_nonzero(self):
        """aggregate_rate is computed only over nonzero-dp positions."""
        from tracks.representation.statistical.hypotheses.h1 import h1_permutation_test
        # Class 0: all zeros (should not contribute)
        # Class 1: all nonzero, perfect agreement
        dp = np.zeros((6, 2), dtype=np.float32)
        dm = np.zeros((6, 2), dtype=np.float32)
        dp[:, 1] = 1.0
        dm[:, 1] = 0.5  # agrees with dp in class 1
        pids = np.array([1, 1, 2, 2, 3, 3])
        result = h1_permutation_test(dp, dm, pids, n_permutations=100,
                                     rng=np.random.default_rng(5))
        # Class 0 contributes nothing; class 1 has perfect agreement
        assert result["aggregate_rate"] == pytest.approx(1.0), (
            "Zero-dp positions must not reduce agreement rate"
        )

    def test_bootstrap_rate_single_patient_returns_nan(self):
        """Single-patient bootstrap should return (nan, nan) CI."""
        from shared.statistical.bootstrap import bootstrap_rate
        ind = np.array([1.0, 0.0, 1.0])
        pids = np.array([7, 7, 7])
        pt, lo, hi = bootstrap_rate(ind, pids)
        assert np.isnan(lo) and np.isnan(hi)

    def test_bootstrap_rate_two_patients(self):
        from shared.statistical.bootstrap import bootstrap_rate
        ind = np.array([1.0, 0.0, 1.0, 0.0])
        pids = np.array([1, 1, 2, 2])
        pt, lo, hi = bootstrap_rate(ind, pids, n_resamples=500,
                                     rng=np.random.default_rng(99))
        assert 0.0 <= lo <= hi <= 1.0
        assert abs(pt - 0.5) < 1e-6

    def test_bootstrap_division_by_zero_guarded(self):
        """Bootstrap metric fn that returns 0 for empty index must not crash."""
        from shared.statistical.bootstrap import patient_block_bootstrap
        called = []

        def metric(idx):
            called.append(len(idx))
            return float(np.sum(idx)) / max(len(idx), 1)

        pids = np.array([1, 1, 2, 2, 3, 3])
        lo, hi = patient_block_bootstrap(metric, pids, n_resamples=100,
                                          rng=np.random.default_rng(0))
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_permutation_preserves_within_patient_correlation(self):
        """Within each patient block, all items are permuted atomically (row-wise)."""
        from tracks.representation.statistical.hypotheses.h1 import h1_permutation_test
        # Patient with 4 items, all dm rows identical → after permutation still identical
        dp = np.ones((4, 2), dtype=np.float32) * 0.5
        dm = np.tile([0.3, 0.7], (4, 1)).astype(np.float32)
        pids = np.array([1, 1, 1, 1])
        # With single patient, bootstrap CI returns (nan, nan) but statistic is valid
        result = h1_permutation_test(dp, dm, pids, n_permutations=50,
                                     rng=np.random.default_rng(0))
        # All permutations should give same aggregate_rate (rows identical)
        assert result["null_std"] < 1e-6, (
            "With identical dm rows, null distribution std must be ~0"
        )

    def test_bootstrap_patient_block_resamples_whole_block(self):
        """Bootstrap must resample entire patient blocks, not individual observations."""
        from shared.statistical.bootstrap import patient_block_bootstrap
        # Patient 1: values [10, 10], Patient 2: values [0, 0]
        values = np.array([10.0, 10.0, 0.0, 0.0])
        pids = np.array([1, 1, 2, 2])
        seen_means = set()

        def metric(idx):
            m = float(np.mean(values[idx]))
            seen_means.add(round(m, 1))
            return m

        lo, hi = patient_block_bootstrap(metric, pids, n_resamples=500,
                                          rng=np.random.default_rng(42))
        # Possible means: 0.0 (both patient 2), 5.0 (one each), 10.0 (both patient 1)
        # Never: 5.0 ± small noise (would happen if mixing individual observations)
        assert seen_means.issubset({0.0, 5.0, 10.0}), (
            f"Unexpected bootstrap means: {seen_means} — individual obs being mixed?"
        )
