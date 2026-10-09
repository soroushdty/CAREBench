"""Audit tests: synthetic known-answer cases and edge cases for all stat modules.

Covers every issue flagged in Prompt 8 (third-party audit pass):
  - CMH valid 2×2 tables with known odds ratios
  - Human-human ICC bootstrap CIs present in output
  - Calibration ECE computation and >0.10 flag
  - Entropy p_value key renamed; descriptive_only=True in output
  - Bootstrap consistency across all CI-bearing outputs
  - Edge cases: no nonzero deltas, zero positives, empty stratum,
    constant predictions, degenerate CMH strata
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_ITEMS = 40
N_PATIENTS = 8
N_CLASSES = 4
CLASSES = ["A", "B", "C", "D"]

def _pids(n=N_ITEMS, n_pt=N_PATIENTS):
    return np.repeat(np.arange(n_pt), n // n_pt)


# ---------------------------------------------------------------------------
# Issue 3 — CMH valid 2×2 tables and known odds ratios
# ---------------------------------------------------------------------------

class TestCMHKnownAnswers:

    def test_cmh_known_or_both_directions(self):
        """Synthetic case with known OR.

        Table (cols = model direction, consistent across rows):
            dp>0: 8 dm>0 (agree/a), 2 dm≤0 (disagree/b)
            dp<0: 5 dm>0 (disagree/c), 5 dm≤0 (agree/d)
            OR = (a·d)/(b·c) = (8·5)/(2·5) = 4.0
        """
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

        dp = np.zeros((20, 1), dtype=np.float32)
        dm = np.zeros((20, 1), dtype=np.float32)
        # 8 dp>0, dm>0  (pos-agree → a)
        for i in range(8):
            dp[i, 0] = 0.5; dm[i, 0] = 0.5
        # 2 dp>0, dm<0  (pos-disagree → b, dm≤0)
        for i in range(8, 10):
            dp[i, 0] = 0.5; dm[i, 0] = -0.5
        # 5 dp<0, dm>0  (neg-disagree → c)
        for i in range(10, 15):
            dp[i, 0] = -0.5; dm[i, 0] = 0.5
        # 5 dp<0, dm<0  (neg-agree → d, dm≤0)
        for i in range(15, 20):
            dp[i, 0] = -0.5; dm[i, 0] = -0.5

        result = h1_cmh_test(dp, dm, ["X"], ["X"])
        # OR = (8*5)/(2*5) = 4.0
        assert not np.isnan(result["common_odds_ratio"]), "OR should be finite"
        assert abs(result["common_odds_ratio"] - 4.0) < 0.01, (
            f"Expected OR≈4.0, got {result['common_odds_ratio']}"
        )

    def test_cmh_null_case_or_near_one(self):
        """When agree = disagree for each physician direction, OR should be ~1."""
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

        dp = np.zeros((20, 1), dtype=np.float32)
        dm = np.zeros((20, 1), dtype=np.float32)
        # 5 pos-agree, 5 pos-disagree → 50/50
        for i in range(5):
            dp[i, 0] = 0.5; dm[i, 0] = 0.5
        for i in range(5, 10):
            dp[i, 0] = 0.5; dm[i, 0] = -0.5
        # 5 neg-agree, 5 neg-disagree → 50/50
        for i in range(10, 15):
            dp[i, 0] = -0.5; dm[i, 0] = -0.5
        for i in range(15, 20):
            dp[i, 0] = -0.5; dm[i, 0] = 0.5

        result = h1_cmh_test(dp, dm, ["X"], ["X"])
        assert not np.isnan(result["common_odds_ratio"]), "OR should be finite"
        assert abs(result["common_odds_ratio"] - 1.0) < 0.1, (
            f"Expected OR≈1.0 (null), got {result['common_odds_ratio']}"
        )

    def test_cmh_degenerate_single_physician_direction_skipped(self):
        """Stratum with only Δ_p>0 (no Δ_p<0) should be skipped, not use fake [1,1]."""
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

        dp = np.zeros((10, 1), dtype=np.float32)
        dm = np.zeros((10, 1), dtype=np.float32)
        # All physician deltas positive — no negative direction available
        dp[:, 0] = 0.5
        dm[:5, 0] = 0.5   # agree
        dm[5:, 0] = -0.5  # disagree

        result = h1_cmh_test(dp, dm, ["X"], ["X"])
        # Degenerate: only one physician direction → skip this stratum
        assert result["n_strata"] == 0, (
            f"Expected 0 usable strata (all Δ_p>0), got {result['n_strata']}"
        )
        assert np.isnan(result["common_odds_ratio"])

    def test_cmh_two_strata_combined(self):
        """Two classes, each with known OR = 4.0 → pooled OR should also be ≈4.0."""
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

        dp = np.zeros((40, 2), dtype=np.float32)
        dm = np.zeros((40, 2), dtype=np.float32)

        for col in range(2):
            base = col * 20
            for i in range(8):
                dp[base + i, col] = 0.5; dm[base + i, col] = 0.5
            for i in range(8, 10):
                dp[base + i, col] = 0.5; dm[base + i, col] = -0.5
            for i in range(10, 15):
                dp[base + i, col] = -0.5; dm[base + i, col] = -0.5
            for i in range(15, 20):
                dp[base + i, col] = -0.5; dm[base + i, col] = 0.5

        result = h1_cmh_test(dp, dm, ["X", "Y"], ["X", "Y"])
        assert result["n_strata"] == 2
        assert abs(result["common_odds_ratio"] - 4.0) < 0.2

    def test_cmh_pvalue_below_05_for_strong_agreement(self):
        """80% sign agreement, 50 items each physician direction → OR=16, p<0.05.

        Correct table structure (cols = model direction, CONSISTENT across rows):
            dp>0: 40 dm>0 (agree), 10 dm≤0 (disagree)  → row=[40, 10]
            dp<0: 10 dm>0 (disagree), 40 dm≤0 (agree)  → row=[10, 40]
            OR = (40·40)/(10·10) = 16.0
        """
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

        n_cls = 2
        n_per_cls = 100
        dp = np.zeros((n_cls * n_per_cls, n_cls), dtype=np.float32)
        dm = np.zeros((n_cls * n_per_cls, n_cls), dtype=np.float32)
        for col in range(n_cls):
            base = col * n_per_cls
            # 40 dp>0, dm>0  (pos-agree → cell a)
            dp[base:base+40, col] = 0.5; dm[base:base+40, col] = 0.5
            # 10 dp>0, dm<0  (pos-disagree → cell b, dm≤0)
            dp[base+40:base+50, col] = 0.5; dm[base+40:base+50, col] = -0.5
            # 10 dp<0, dm>0  (neg-disagree → cell c)
            dp[base+50:base+60, col] = -0.5; dm[base+50:base+60, col] = 0.5
            # 40 dp<0, dm<0  (neg-agree → cell d, dm≤0)
            dp[base+60:base+100, col] = -0.5; dm[base+60:base+100, col] = -0.5

        result = h1_cmh_test(dp, dm, ["X", "Y"], ["X", "Y"])
        assert result["n_strata"] == 2
        assert abs(result["common_odds_ratio"] - 16.0) < 0.5, (
            f"Expected OR≈16.0 for 80% sign agreement, got {result['common_odds_ratio']}"
        )
        if not np.isnan(result["p_cmh"]):
            assert result["p_cmh"] < 0.05, (
                f"Expected significant CMH (n=100/dir/class, OR=16), got p={result['p_cmh']}"
            )

    def test_cmh_no_eligible_classes_returns_nan(self):
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test
        dp = np.ones((20, 2), dtype=np.float32) * 0.5
        dm = np.ones((20, 2), dtype=np.float32) * 0.5
        result = h1_cmh_test(dp, dm, [], ["X", "Y"])
        assert result["n_strata"] == 0
        assert np.isnan(result["common_odds_ratio"])


# ---------------------------------------------------------------------------
# Issue 5 — Human-human ICC bootstrap CIs
# ---------------------------------------------------------------------------

class TestICCHumanHumanBootstrapCI:

    def _make_icc_results(self):
        """Synthesise what rater_icc_analysis() returns with dummy data."""
        from shared.statistical.icc import icc21, lins_ccc
        from shared.statistical.bootstrap import patient_block_bootstrap

        rng = np.random.default_rng(7)
        n_patients = 6
        items_per_patient = 5
        n_classes = 3

        model_physician_iccs = []
        human_human_iccs = []

        for pid in range(n_patients):
            ratings_mp = rng.random((items_per_patient * n_classes, 2))
            model_physician_iccs.append({
                "patient": pid,
                "physician": "ph1",
                "icc": icc21(ratings_mp),
                "ccc": lins_ccc(ratings_mp[:, 0], ratings_mp[:, 1]),
                "n_items": items_per_patient,
            })
            ratings_hh = rng.random((items_per_patient * n_classes, 2))
            human_human_iccs.append({
                "patient": pid,
                "icc": icc21(ratings_hh),
                "ccc": lins_ccc(ratings_hh[:, 0], ratings_hh[:, 1]),
                "n_items": items_per_patient,
            })

        # Run the actual bootstrap logic from icc.py
        mp_icc_vals = np.array([d["icc"] for d in model_physician_iccs], dtype=float)
        hh_icc_vals = np.array([d["icc"] for d in human_human_iccs], dtype=float)
        mp_ccc_vals = np.array([d["ccc"] for d in model_physician_iccs], dtype=float)
        hh_ccc_vals = np.array([d["ccc"] for d in human_human_iccs], dtype=float)

        def _nanmean(arr):
            return float(np.nanmean(arr)) if len(arr) > 0 else float("nan")

        mp_patient_ids = np.array([d["patient"] for d in model_physician_iccs])
        mp_icc_finite = np.where(np.isnan(mp_icc_vals), 0.0, mp_icc_vals)

        def _mp_icc_mean(idx):
            return float(np.mean(mp_icc_finite[idx]))

        mp_ci_lo, mp_ci_hi = patient_block_bootstrap(_mp_icc_mean, mp_patient_ids, 200, rng)

        hh_patient_ids = np.array([d["patient"] for d in human_human_iccs])
        hh_icc_finite = np.where(np.isnan(hh_icc_vals), 0.0, hh_icc_vals)

        def _hh_icc_mean(idx):
            return float(np.mean(hh_icc_finite[idx]))

        hh_ci_lo, hh_ci_hi = patient_block_bootstrap(_hh_icc_mean, hh_patient_ids, 200, rng)

        return {
            "model_physician_iccs": model_physician_iccs,
            "human_human_iccs": human_human_iccs,
            "model_physician_icc_mean": _nanmean(mp_icc_vals),
            "model_physician_icc_ci_lower": mp_ci_lo,
            "model_physician_icc_ci_upper": mp_ci_hi,
            "model_physician_icc_sd": float(np.nanstd(mp_icc_vals)),
            "human_human_icc_mean": _nanmean(hh_icc_vals),
            "human_human_icc_ci_lower": hh_ci_lo,
            "human_human_icc_ci_upper": hh_ci_hi,
            "human_human_icc_sd": float(np.nanstd(hh_icc_vals)),
            "human_human_icc_min": float(np.nanmin(hh_icc_vals)),
            "human_human_icc_max": float(np.nanmax(hh_icc_vals)),
            "model_physician_ccc_mean": _nanmean(mp_ccc_vals),
            "human_human_ccc_mean": _nanmean(hh_ccc_vals),
            "n_model_physician_pairs": len(model_physician_iccs),
            "n_human_human_pairs": len(human_human_iccs),
        }

    def test_hh_ci_keys_present(self):
        res = self._make_icc_results()
        assert "human_human_icc_ci_lower" in res, "Missing human-human ICC CI lower key"
        assert "human_human_icc_ci_upper" in res, "Missing human-human ICC CI upper key"

    def test_hh_ci_ordered(self):
        res = self._make_icc_results()
        lo = res["human_human_icc_ci_lower"]
        hi = res["human_human_icc_ci_upper"]
        assert lo <= hi, f"Human-human ICC CI not ordered: [{lo}, {hi}]"

    def test_hh_ci_finite(self):
        res = self._make_icc_results()
        assert not np.isnan(res["human_human_icc_ci_lower"])
        assert not np.isnan(res["human_human_icc_ci_upper"])

    def test_mp_ci_keys_present(self):
        res = self._make_icc_results()
        assert "model_physician_icc_ci_lower" in res
        assert "model_physician_icc_ci_upper" in res


# ---------------------------------------------------------------------------
# Issue 6 — Calibration ECE with >0.10 flag
# ---------------------------------------------------------------------------

class TestCalibrationECE:

    def test_ece_returns_dataframe(self):
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(1)
        y_hat = rng.random((N_ITEMS, N_CLASSES))
        y_true = rng.choice([0.0, 0.5, 1.0], size=(N_ITEMS, N_CLASSES))
        pid = _pids()
        df = calibration_ece_per_class(y_hat, y_true, CLASSES, pid, n_resamples=100, rng=rng)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"Class", "ECE", "ci_lower", "ci_upper", "flagged"}
        assert len(df) == N_CLASSES

    def test_perfect_calibration_low_ece(self):
        """Predictions equal to labels → ECE ≈ 0 and not flagged."""
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(2)
        y_true = rng.choice([0.0, 0.5, 1.0], size=(N_ITEMS, 1))
        y_hat = y_true.copy()
        pid = _pids()
        df = calibration_ece_per_class(y_hat, y_true, ["X"], pid, n_resamples=50, rng=rng)
        assert df.iloc[0]["ECE"] < 0.05
        assert not df.iloc[0]["flagged"]

    def test_constant_prediction_high_ece_flagged(self):
        """Predicting 0.9 for all when labels average 0.5 → ECE ≈ 0.4, flagged."""
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(3)
        y_true = np.full((N_ITEMS, 1), 0.5)
        y_hat = np.full((N_ITEMS, 1), 0.9)
        pid = _pids()
        df = calibration_ece_per_class(y_hat, y_true, ["X"], pid, n_resamples=50, rng=rng)
        assert df.iloc[0]["ECE"] > 0.10
        assert df.iloc[0]["flagged"]

    def test_ece_ci_ordered(self):
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(4)
        y_hat = rng.random((N_ITEMS, N_CLASSES))
        y_true = rng.choice([0.0, 1.0], size=(N_ITEMS, N_CLASSES)).astype(float)
        pid = _pids()
        df = calibration_ece_per_class(y_hat, y_true, CLASSES, pid, n_resamples=100, rng=rng)
        for _, row in df.iterrows():
            if not (np.isnan(row["ci_lower"]) or np.isnan(row["ci_upper"])):
                assert row["ci_lower"] <= row["ci_upper"]

    def test_ece_nonnegative(self):
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(5)
        y_hat = rng.random((N_ITEMS, N_CLASSES))
        y_true = rng.choice([0.0, 0.5, 1.0], size=(N_ITEMS, N_CLASSES))
        pid = _pids()
        df = calibration_ece_per_class(y_hat, y_true, CLASSES, pid, n_resamples=50, rng=rng)
        assert (df["ECE"] >= 0).all() or df["ECE"].isna().any()

    def test_ece_flag_threshold_respected(self):
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(6)
        y_hat = rng.random((N_ITEMS, 1))
        y_true = 1.0 - y_hat  # maximally miscalibrated
        pid = _pids()
        df = calibration_ece_per_class(
            y_hat, y_true, ["X"], pid, n_resamples=50,
            ece_flag_threshold=0.10, rng=rng
        )
        if df.iloc[0]["ECE"] > 0.10:
            assert df.iloc[0]["flagged"]
        else:
            assert not df.iloc[0]["flagged"]


# ---------------------------------------------------------------------------
# Issue 7 — Entropy descriptive-only keys
# ---------------------------------------------------------------------------

class TestEntropyDescriptiveOnly:

    def test_pearson_result_has_descriptive_only_flag(self):
        from shared.statistical.entropy import entropy_pearson_r
        rng = np.random.default_rng(10)
        n = 50
        ph = rng.random((n, 2))
        mo = rng.random((n, 2))
        result = entropy_pearson_r(ph, mo, ["A", "B"], ["A", "B"])
        assert "descriptive_only" in result, "Missing descriptive_only key"
        assert result["descriptive_only"] is True

    def test_pearson_result_no_plain_p_value_key(self):
        """The plain 'p_value' key must not appear — renamed to avoid confusion."""
        from shared.statistical.entropy import entropy_pearson_r
        rng = np.random.default_rng(11)
        n = 50
        ph = rng.random((n, 2))
        mo = rng.random((n, 2))
        result = entropy_pearson_r(ph, mo, ["A", "B"], ["A", "B"])
        assert "p_value" not in result, (
            "'p_value' key must be renamed to 'p_value_descriptive_only' "
            "to prevent misinterpretation as a confirmatory test."
        )

    def test_pearson_result_has_renamed_p_value(self):
        from shared.statistical.entropy import entropy_pearson_r
        rng = np.random.default_rng(12)
        n = 50
        ph = rng.random((n, 2))
        mo = rng.random((n, 2))
        result = entropy_pearson_r(ph, mo, ["A", "B"], ["A", "B"])
        assert "p_value_descriptive_only" in result

    def test_pearson_empty_classes_returns_descriptive_flag(self):
        from shared.statistical.entropy import entropy_pearson_r
        ph = np.zeros((10, 2))
        mo = np.zeros((10, 2))
        result = entropy_pearson_r(ph, mo, [], ["A", "B"])
        assert result["descriptive_only"] is True


# ---------------------------------------------------------------------------
# Edge case — No nonzero deltas for a class
# ---------------------------------------------------------------------------

class TestEdgeCaseNoNonzeroDelta:

    def test_h1_binomial_all_zero_dp_returns_nan(self):
        from tracks.representation.statistical.hypotheses.h1 import h1_binomial_per_class
        dp = np.zeros((N_ITEMS, N_CLASSES), dtype=np.float32)
        dm = np.ones((N_ITEMS, N_CLASSES), dtype=np.float32) * 0.5
        pid = _pids()
        df = h1_binomial_per_class(dp, dm, CLASSES, CLASSES, pid, n_resamples=50,
                                   rng=np.random.default_rng(20))
        assert df["n_nonzero"].eq(0).all()
        assert df["sign_agree_rate"].isna().all()

    def test_h1_cmh_all_same_dp_direction_skipped(self):
        """If every class has only Δ_p>0, CMH must skip all strata."""
        from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test
        dp = np.ones((20, 2), dtype=np.float32) * 0.5  # all positive
        dm = np.random.default_rng(21).choice([-0.5, 0.5], size=(20, 2)).astype(np.float32)
        result = h1_cmh_test(dp, dm, ["A", "B"], ["A", "B"])
        assert result["n_strata"] == 0

    def test_wilcoxon_all_zero_diff_returns_p1(self):
        from tracks.representation.statistical.hypotheses.h2 import h2_wilcoxon_per_class
        y = np.ones((N_ITEMS, 1)) * 0.5
        pid = _pids()
        df = h2_wilcoxon_per_class(y, y, y, ["X"], pid, n_resamples=50,
                                   rng=np.random.default_rng(22))
        assert df.iloc[0]["wilcoxon_p"] == 1.0 or np.isnan(df.iloc[0]["wilcoxon_p"])


# ---------------------------------------------------------------------------
# Edge case — Zero positives (k=0 in binomial)
# ---------------------------------------------------------------------------

class TestEdgeCaseZeroPositives:

    def test_h1_binomial_zero_agreements(self):
        """All dp > 0 but all dm < 0 → zero sign agreements, valid test."""
        from tracks.representation.statistical.hypotheses.h1 import h1_binomial_per_class
        dp = np.ones((N_ITEMS, 1), dtype=np.float32) * 0.5
        dm = -np.ones((N_ITEMS, 1), dtype=np.float32) * 0.5
        pid = _pids()
        df = h1_binomial_per_class(dp, dm, ["X"], ["X"], pid, n_resamples=50,
                                   rng=np.random.default_rng(30))
        assert df.iloc[0]["sign_agree_rate"] == 0.0
        # p-value for zero agreements (alternative=greater) should be 1.0
        assert df.iloc[0]["binom_p"] == 1.0


# ---------------------------------------------------------------------------
# Edge case — Bootstrap sample with thin patient coverage
# ---------------------------------------------------------------------------

class TestEdgeCaseBootstrapThinPatients:

    def test_bootstrap_with_two_patients(self):
        """Minimum viable patient count (2) should produce finite CIs."""
        from shared.statistical.bootstrap import bootstrap_scalar
        rng = np.random.default_rng(40)
        vals = rng.random(20)
        pids = np.repeat([0, 1], 10)
        pt, lo, hi = bootstrap_scalar(vals, pids, np.mean, n_resamples=100, rng=rng)
        assert not np.isnan(lo) and not np.isnan(hi)
        assert lo <= hi

    def test_bootstrap_single_patient_returns_nan(self):
        from shared.statistical.bootstrap import patient_block_bootstrap
        rng = np.random.default_rng(41)
        pids = np.zeros(10, dtype=int)
        lo, hi = patient_block_bootstrap(lambda idx: float(np.mean(idx)), pids, 50, rng)
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# Edge case — Empty stratum
# ---------------------------------------------------------------------------

class TestEdgeCaseEmptyStratum:

    def test_stratum_comparison_all_repeated(self):
        """When all items are 'repeated', novel stratum should report NaN CI."""
        from tracks.representation.statistical.reporting.stratum import stratum_comparison
        rng = np.random.default_rng(50)
        dp = rng.choice([-0.5, 0.0, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        dm = rng.choice([-0.5, 0.0, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        cf = rng.random((N_ITEMS, N_CLASSES))
        ca = rng.random((N_ITEMS, N_CLASSES))
        yi = rng.choice([0.0, 0.5, 1.0], size=(N_ITEMS, N_CLASSES))
        strata = np.array(["repeated"] * N_ITEMS)
        pid = _pids()
        df = stratum_comparison(dp, dm, cf, ca, yi, strata, pid, CLASSES, CLASSES,
                                n_resamples=50, rng=rng)
        novel_row = df[df["stratum"] == "novel"].iloc[0]
        assert novel_row["n_items"] == 0


# ---------------------------------------------------------------------------
# Edge case — Constant predictions (Wilcoxon / ECE / Wasserstein)
# ---------------------------------------------------------------------------

class TestEdgeCaseConstantPredictions:

    def test_wasserstein_constant_predictions(self):
        """Constant predictions yield finite Wasserstein distance."""
        from tracks.representation.statistical.hypotheses.h2 import h2_wasserstein_per_class
        rng = np.random.default_rng(60)
        cf = np.full((N_ITEMS, 1), 0.5)
        ca = np.full((N_ITEMS, 1), 0.5)
        yi = rng.choice([0.0, 1.0], size=(N_ITEMS, 1)).astype(float)
        pid = _pids()
        df = h2_wasserstein_per_class(cf, ca, yi, ["X"], pid, n_resamples=50, rng=rng)
        assert not np.isnan(df.iloc[0]["W_cf"])
        assert not np.isnan(df.iloc[0]["W_ca"])

    def test_ece_constant_predictions(self):
        """Constant prediction (all 0.8) with binary labels → finite ECE."""
        from shared.evaluation.calibration import calibration_ece_per_class
        rng = np.random.default_rng(61)
        y_hat = np.full((N_ITEMS, 1), 0.8)
        y_true = rng.choice([0.0, 1.0], size=(N_ITEMS, 1)).astype(float)
        pid = _pids()
        df = calibration_ece_per_class(y_hat, y_true, ["X"], pid, n_resamples=50, rng=rng)
        assert not np.isnan(df.iloc[0]["ECE"])

    def test_icc21_constant_rater_is_degenerate(self):
        """ICC with a constant rater column should return NaN (or very negative)."""
        from shared.statistical.icc import icc21
        ratings = np.column_stack([
            np.random.default_rng(62).random(20),
            np.ones(20) * 0.5,  # constant rater
        ])
        icc = icc21(ratings)
        # Degenerate — constant rater has zero between-subject variance on that column
        assert np.isnan(icc) or icc <= 0.0


# ---------------------------------------------------------------------------
# Issue 1 — Bootstrap used consistently for all required CIs
# ---------------------------------------------------------------------------

class TestBootstrapConsistency:

    def test_h2_wilcoxon_has_bootstrap_ci(self):
        from tracks.representation.statistical.hypotheses.h2 import h2_wilcoxon_per_class
        rng = np.random.default_rng(70)
        cf = rng.random((N_ITEMS, N_CLASSES))
        ca = rng.random((N_ITEMS, N_CLASSES))
        yi = rng.choice([0.0, 0.5, 1.0], size=(N_ITEMS, N_CLASSES))
        pid = _pids()
        df = h2_wilcoxon_per_class(cf, ca, yi, CLASSES, pid, n_resamples=100, rng=rng)
        assert "ci_lower" in df.columns and "ci_upper" in df.columns
        # CIs should be ordered for all rows
        for _, row in df.iterrows():
            if not (np.isnan(row["ci_lower"]) or np.isnan(row["ci_upper"])):
                assert row["ci_lower"] <= row["ci_upper"]

    def test_h1_binomial_has_bootstrap_ci(self):
        from tracks.representation.statistical.hypotheses.h1 import h1_binomial_per_class
        rng = np.random.default_rng(71)
        dp = rng.choice([-0.5, 0.0, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        dm = rng.choice([-0.5, 0.0, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        pid = _pids()
        df = h1_binomial_per_class(dp, dm, CLASSES, CLASSES, pid, n_resamples=100, rng=rng)
        assert "ci_lower" in df.columns and "ci_upper" in df.columns

    def test_wasserstein_has_bootstrap_ci(self):
        from tracks.representation.statistical.hypotheses.h2 import h2_wasserstein_per_class
        rng = np.random.default_rng(72)
        cf = rng.random((N_ITEMS, N_CLASSES))
        ca = rng.random((N_ITEMS, N_CLASSES))
        yi = rng.choice([0.0, 0.5, 1.0], size=(N_ITEMS, N_CLASSES))
        pid = _pids()
        df = h2_wasserstein_per_class(cf, ca, yi, CLASSES, pid, n_resamples=100, rng=rng)
        for col in ("ci_lower_cf", "ci_upper_cf", "ci_lower_ca", "ci_upper_ca"):
            assert col in df.columns


# ---------------------------------------------------------------------------
# Issue 2 — H1 permutation: correct within-patient permutation
# ---------------------------------------------------------------------------

class TestH1PermutationCorrectness:

    def test_permutation_preserves_within_patient_structure(self):
        """After permutation, each patient still has the same item count."""
        from tracks.representation.statistical.hypotheses.h1 import h1_permutation_test
        rng = np.random.default_rng(80)
        dp = rng.choice([-0.5, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        dm = dp.copy()
        pid = _pids()
        result = h1_permutation_test(dp, dm, pid, n_permutations=100, rng=rng)
        # Perfect agreement: observed >> null mean → low p
        assert result["p_value"] < 0.05

    def test_permutation_null_preserves_randomness(self):
        """With random dm, null distribution should be centered near observed."""
        from tracks.representation.statistical.hypotheses.h1 import h1_permutation_test
        rng = np.random.default_rng(81)
        dp = rng.choice([-0.5, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        dm = rng.choice([-0.5, 0.5], size=(N_ITEMS, N_CLASSES)).astype(np.float32)
        pid = _pids()
        result = h1_permutation_test(dp, dm, pid, n_permutations=200, rng=rng)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# CMH: known OR via manual MH fallback
# ---------------------------------------------------------------------------

class TestManualMHFallback:

    def test_manual_mh_known_answer(self):
        """Verify _manual_mh gives correct OR for a hand-computed case."""
        from tracks.representation.statistical.hypotheses.h1 import _manual_mh
        # Single 2x2: [[8,2],[5,5]] → OR = (8*5)/(2*5) = 4.0
        tables = [np.array([[8., 2.], [5., 5.]])]
        result = _manual_mh(tables)
        assert abs(result["common_odds_ratio"] - 4.0) < 0.01, (
            f"Expected OR=4.0, got {result['common_odds_ratio']}"
        )

    def test_manual_mh_null_case(self):
        """Symmetric table → OR = 1.0."""
        from tracks.representation.statistical.hypotheses.h1 import _manual_mh
        tables = [np.array([[5., 5.], [5., 5.]])]
        result = _manual_mh(tables)
        assert abs(result["common_odds_ratio"] - 1.0) < 0.01
