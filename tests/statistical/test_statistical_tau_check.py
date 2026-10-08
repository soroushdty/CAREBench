"""Tests for shared/statistical/tau_check.py."""
import numpy as np
import pandas as pd
from shared.statistical.tau_check import tau_robustness_check, tau_robustness_summary

CLASSES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# tau_robustness_check
# ---------------------------------------------------------------------------

def test_returns_dataframe_with_correct_columns():
    thresholds = np.array([0.48, 0.52, 0.40, 0.55])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(CLASSES)
    assert set(df.columns) >= {"Class", "tau_fixed", "tau_f1opt", "abs_diff", "within_tolerance"}


def test_within_tolerance_all_close():
    # All F1-optimal thresholds within 0.05 of 0.5
    thresholds = np.array([0.49, 0.51, 0.50, 0.48])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    assert df["within_tolerance"].all()


def test_flags_class_exceeding_tolerance():
    # Class C has threshold 0.40 → |0.5 - 0.40| = 0.10 > 0.05
    thresholds = np.array([0.48, 0.52, 0.40, 0.55])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    flagged = df[~df["within_tolerance"]]
    assert "C" in flagged["Class"].values
    assert "D" in flagged["Class"].values


def test_sorted_by_abs_diff_descending():
    thresholds = np.array([0.48, 0.45, 0.60, 0.50])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    diffs = df["abs_diff"].values
    assert np.all(diffs[:-1] >= diffs[1:] - 1e-12)


def test_abs_diff_values():
    thresholds = np.array([0.30, 0.50, 0.70, 0.55])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    # Sort by class to check abs_diff values
    df_by_class = df.set_index("Class")
    assert abs(df_by_class.loc["A", "abs_diff"] - 0.20) < 1e-10
    assert abs(df_by_class.loc["B", "abs_diff"] - 0.00) < 1e-10
    assert abs(df_by_class.loc["C", "abs_diff"] - 0.20) < 1e-10
    assert abs(df_by_class.loc["D", "abs_diff"] - 0.05) < 1e-10


def test_tau_fixed_column_constant():
    thresholds = np.array([0.48, 0.52, 0.40, 0.55])
    tau = 0.35
    df = tau_robustness_check(tau, thresholds, CLASSES, tolerance=0.05)
    assert (df["tau_fixed"] == tau).all()


def test_custom_tolerance():
    thresholds = np.array([0.42, 0.50, 0.58, 0.50])
    # Tolerance 0.10: |0.5 - 0.42| = 0.08 < 0.10 → within
    df = tau_robustness_check(0.5, thresholds, CLASSES, tolerance=0.10)
    assert df["within_tolerance"].all()


# ---------------------------------------------------------------------------
# tau_robustness_summary
# ---------------------------------------------------------------------------

def test_summary_no_flagged():
    thresholds = np.array([0.49, 0.51, 0.50, 0.48])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    summary = tau_robustness_summary(df)
    assert "All" in summary
    assert "within" in summary.lower()


def test_summary_with_flagged():
    thresholds = np.array([0.30, 0.50, 0.70, 0.50])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    summary = tau_robustness_summary(df)
    assert "2 of 4" in summary
    assert "A" in summary or "C" in summary


def test_summary_is_string():
    thresholds = np.array([0.48, 0.52, 0.40, 0.55])
    df = tau_robustness_check(0.5, thresholds, CLASSES)
    summary = tau_robustness_summary(df)
    assert isinstance(summary, str)
    assert len(summary) > 0
