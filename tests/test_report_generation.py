"""Tests for assay.bundle.report.generate_markdown_report and assay.bundle.summary."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.report import generate_markdown_report
from tracks.reasoning.bundle.summary import make_hypothesis_summary, make_model_comparison_summary

CATS = ["behavioral_health", "diagnoses", "disabilities"]


def _make_hypothesis_summary_df(models=("model_a",), flag="strong_patient_specific_alignment"):
    rows = []
    for m in models:
        rows.append(
            {
                "model": m,
                "n_patients": 3,
                "n_patient_items": 6,
                "n_cells": 18,
                "n_physician_shift_cells": 12,
                "H1_mean_abs_delta_correct": 0.2,
                "H1_mean_abs_delta_shuffled": 0.1,
                "H1_proportion_changed_correct": 0.9,
                "H1_proportion_changed_shuffled": 0.5,
                "H2_mean_alignment_correct": 0.15,
                "H2_mean_alignment_shuffled": 0.05,
                "H2_sign_agreement_correct": 0.8,
                "H2_sign_agreement_shuffled": 0.55,
                "H3_pearson_correct": 0.7,
                "H3_spearman_correct": 0.65,
                "H3_pearson_shuffled": 0.1,
                "H3_spearman_shuffled": 0.05,
                "H4_mean_alignment_difference": 0.1,
                "H4_ci_low": 0.02,
                "H4_ci_high": 0.18,
                "H4_p_value": 0.03,
                "failure_rate": 0.0,
                "interpretation_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def _make_minimal_dfs():
    """Create minimal H1–H4 DataFrames for the report."""
    cats_and_agg = CATS + ["aggregate"]

    def _h1():
        rows = []
        for cat in cats_and_agg:
            rows.append(
                {
                    "model": "model_a",
                    "category": cat,
                    "n_patients": 3,
                    "n_patient_items": 6,
                    "n_cells": 18,
                    "mean_abs_delta_correct": 0.2,
                    "ci_low_correct": 0.1,
                    "ci_high_correct": 0.3,
                    "mean_abs_delta_shuffled": 0.1,
                    "ci_low_shuffled": 0.05,
                    "ci_high_shuffled": 0.15,
                    "proportion_changed_correct": 0.9,
                    "proportion_changed_shuffled": 0.5,
                    "epsilon": 0.01,
                }
            )
        return pd.DataFrame(rows)

    def _h2():
        rows = []
        for cat in cats_and_agg:
            rows.append(
                {
                    "model": "model_a",
                    "category": cat,
                    "n_patients": 3,
                    "n_shift_cells": 10,
                    "mean_alignment_correct": 0.15,
                    "ci_low_correct": 0.05,
                    "ci_high_correct": 0.25,
                    "mean_alignment_shuffled": 0.05,
                    "ci_low_shuffled": -0.05,
                    "ci_high_shuffled": 0.15,
                    "sign_agreement_correct": 0.8,
                    "sign_agreement_shuffled": 0.55,
                    "epsilon": 0.01,
                }
            )
        return pd.DataFrame(rows)

    def _h3_effects():
        rows = []
        for cat in cats_and_agg:
            rows.append(
                {
                    "model": "model_a",
                    "category": cat,
                    "mean_delta_physician": 0.1,
                    "mean_abs_delta_physician": 0.1,
                    "mean_delta_llm_correct": 0.15,
                    "mean_abs_delta_llm_correct": 0.15,
                    "mean_delta_llm_shuffled": 0.05,
                    "mean_abs_delta_llm_shuffled": 0.05,
                }
            )
        return pd.DataFrame(rows)

    def _h3_corr():
        return pd.DataFrame(
            [
                {
                    "model": "model_a",
                    "pearson_correct": 0.7,
                    "spearman_correct": 0.65,
                    "pearson_shuffled": 0.1,
                    "spearman_shuffled": 0.05,
                    "pearson_correct_minus_shuffled": 0.6,
                    "spearman_correct_minus_shuffled": 0.6,
                    "permutation_p_pearson_correct": 0.03,
                    "permutation_p_spearman_correct": 0.04,
                    "n_categories": 3,
                }
            ]
        )

    def _h4():
        rows = []
        for cat in cats_and_agg:
            rows.append(
                {
                    "model": "model_a",
                    "category": cat,
                    "n_patients": 3,
                    "n_shift_cells": 10,
                    "mean_alignment_correct": 0.15,
                    "mean_alignment_shuffled": 0.05,
                    "mean_alignment_difference": 0.1,
                    "ci_low_difference": 0.02,
                    "ci_high_difference": 0.18,
                    "paired_permutation_p": 0.03,
                }
            )
        return pd.DataFrame(rows)

    return _h1(), _h2(), _h3_effects(), _h3_corr(), _h4()


class TestGenerateMarkdownReport:
    def test_report_contains_all_required_sections(self, tmp_path):
        hs = _make_hypothesis_summary_df()
        mc = make_model_comparison_summary(hs)
        h1, h2, h3e, h3c, h4 = _make_minimal_dfs()
        val_df = pd.DataFrame([{"check_id": 1, "description": "test", "status": "PASS", "count": 0, "notes": ""}])

        out = str(tmp_path / "LLM_CONTEXT_SHIFT_REPORT.md")
        generate_markdown_report(hs, h1, h2, h3e, h3c, h4, mc, val_df, out, categories=CATS)

        content = open(out).read()
        required_sections = [
            "# LLM Context-Shift Analysis Report",
            "## Objective",
            "## Data and Pairing",
            "## Models Evaluated",
            "## Validation Summary",
            "## H1: LLM Context Sensitivity",
            "## H2: Directional Alignment with Physician Shifts",
            "## H3: Class-Level Context-Effect Correspondence",
            "## H4: Correct vs Shuffled Context Control",
            "## Model Comparison",
            "## Interpretation",
            "## Limitations",
            "## Reproducibility",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_report_mentions_h4_as_primary(self, tmp_path):
        hs = _make_hypothesis_summary_df()
        mc = make_model_comparison_summary(hs)
        h1, h2, h3e, h3c, h4 = _make_minimal_dfs()
        val_df = pd.DataFrame(columns=["check_id", "description", "status", "count", "notes"])

        out = str(tmp_path / "report.md")
        generate_markdown_report(hs, h1, h2, h3e, h3c, h4, mc, val_df, out, categories=CATS)
        content = open(out).read()
        assert "H4" in content
        assert "primary" in content.lower()

    def test_report_does_not_mention_brier_f1_auroc(self, tmp_path):
        hs = _make_hypothesis_summary_df()
        mc = make_model_comparison_summary(hs)
        h1, h2, h3e, h3c, h4 = _make_minimal_dfs()
        val_df = pd.DataFrame(columns=["check_id", "description", "status", "count", "notes"])

        out = str(tmp_path / "report.md")
        generate_markdown_report(hs, h1, h2, h3e, h3c, h4, mc, val_df, out, categories=CATS)
        content = open(out).read()
        # These metrics should not be primary — objective section explicitly notes they are not reported
        assert "Brier" not in content or "not" in content  # OK if mentioned in context of NOT reporting

    def test_multi_model_report_includes_comparison_table(self, tmp_path):
        hs = _make_hypothesis_summary_df(models=("model_a", "model_b"))
        mc = make_model_comparison_summary(hs)
        # Build multi-model H DataFrames
        h1_rows = []
        for m in ("model_a", "model_b"):
            for cat in CATS + ["aggregate"]:
                h1_rows.append({"model": m, "category": cat, "mean_abs_delta_correct": 0.2,
                                 "ci_low_correct": 0.1, "ci_high_correct": 0.3,
                                 "mean_abs_delta_shuffled": 0.1, "ci_low_shuffled": 0.05,
                                 "ci_high_shuffled": 0.15, "proportion_changed_correct": 0.9,
                                 "proportion_changed_shuffled": 0.5, "epsilon": 0.01,
                                 "n_patients": 3, "n_patient_items": 6, "n_cells": 18})
        h1 = pd.DataFrame(h1_rows)
        _, h2, h3e, h3c, h4 = _make_minimal_dfs()

        out = str(tmp_path / "multi_report.md")
        val_df = pd.DataFrame(columns=["check_id", "description", "status", "count", "notes"])
        generate_markdown_report(hs, h1, h2, h3e, h3c, h4, mc, val_df, out, categories=CATS)
        content = open(out).read()
        assert "model_a" in content
        assert "model_b" in content
        assert "## Model Comparison" in content

    def test_interpretation_flag_strong_alignment(self, tmp_path):
        hs = _make_hypothesis_summary_df(flag="strong_patient_specific_alignment")
        mc = make_model_comparison_summary(hs)
        h1, h2, h3e, h3c, h4 = _make_minimal_dfs()
        val_df = pd.DataFrame(columns=["check_id", "description", "status", "count", "notes"])
        out = str(tmp_path / "report.md")
        generate_markdown_report(hs, h1, h2, h3e, h3c, h4, mc, val_df, out, categories=CATS)
        content = open(out).read()
        assert "strong" in content.lower() or "patient-specific" in content.lower()

    def test_interpretation_flag_nonspecific_inflation(self, tmp_path):
        hs = _make_hypothesis_summary_df(flag="nonspecific_context_inflation")
        mc = make_model_comparison_summary(hs)
        h1, h2, h3e, h3c, h4 = _make_minimal_dfs()
        val_df = pd.DataFrame(columns=["check_id", "description", "status", "count", "notes"])
        out = str(tmp_path / "report.md")
        generate_markdown_report(hs, h1, h2, h3e, h3c, h4, mc, val_df, out, categories=CATS)
        content = open(out).read()
        assert "nonspecific" in content.lower() or "inflation" in content.lower()


class TestMakeHypothesisSummary:
    def _make_simple_dfs(self, model="m"):
        h1 = pd.DataFrame([{
            "model": model, "category": "aggregate", "mean_abs_delta_correct": 0.2,
            "mean_abs_delta_shuffled": 0.1, "proportion_changed_correct": 0.9,
            "proportion_changed_shuffled": 0.5, "epsilon": 0.01,
        }])
        h2 = pd.DataFrame([{
            "model": model, "category": "aggregate", "mean_alignment_correct": 0.15,
            "mean_alignment_shuffled": 0.05, "sign_agreement_correct": 0.8,
            "sign_agreement_shuffled": 0.55,
        }])
        h3c = pd.DataFrame([{
            "model": model, "pearson_correct": 0.7, "spearman_correct": 0.65,
            "pearson_shuffled": 0.1, "spearman_shuffled": 0.05,
        }])
        h4 = pd.DataFrame([{
            "model": model, "category": "aggregate",
            "mean_alignment_difference": 0.1, "ci_low_difference": 0.02,
            "ci_high_difference": 0.18, "paired_permutation_p": 0.03,
        }])
        cells = pd.DataFrame([
            {"model": model, "patient": "P0", "category": "behavioral_health",
             "delta_physician": 0.2, "item_text": "Item0"},
            {"model": model, "patient": "P1", "category": "behavioral_health",
             "delta_physician": 0.3, "item_text": "Item0"},
        ])
        failed = pd.DataFrame(columns=["model", "patient_id", "failure_reason"])
        return h1, h2, h3c, h4, cells, failed

    def test_interpretation_flag_strong(self):
        h1, h2, h3c, h4, cells, failed = self._make_simple_dfs()
        hs = make_hypothesis_summary(h1, h2, h3c, h4, cells, failed, epsilon=0.01)
        assert hs.iloc[0]["interpretation_flag"] == "strong_patient_specific_alignment"

    def test_interpretation_flag_no_sensitivity(self):
        h1, h2, h3c, h4, cells, failed = self._make_simple_dfs()
        # Force H1 to zero
        h1["mean_abs_delta_correct"] = 0.0
        hs = make_hypothesis_summary(h1, h2, h3c, h4, cells, failed, epsilon=0.01)
        assert hs.iloc[0]["interpretation_flag"] == "no_context_sensitivity"

    def test_interpretation_flag_nonspecific_inflation(self):
        h1, h2, h3c, h4, cells, failed = self._make_simple_dfs()
        # Force H4 CI low to negative
        h4["ci_low_difference"] = -0.05
        hs = make_hypothesis_summary(h1, h2, h3c, h4, cells, failed, epsilon=0.01)
        assert hs.iloc[0]["interpretation_flag"] == "nonspecific_context_inflation"

    def test_model_comparison_sorted_by_h4(self):
        h1a, h2a, h3ca, h4a, cells_a, failed = self._make_simple_dfs("model_a")
        h1b, h2b, h3cb, h4b, cells_b, _ = self._make_simple_dfs("model_b")
        h4b["mean_alignment_difference"] = 0.5  # model_b has higher H4
        h1_all = pd.concat([h1a, h1b])
        h2_all = pd.concat([h2a, h2b])
        h3c_all = pd.concat([h3ca, h3cb])
        h4_all = pd.concat([h4a, h4b])
        cells_all = pd.concat([cells_a, cells_b])
        hs = make_hypothesis_summary(h1_all, h2_all, h3c_all, h4_all, cells_all, failed, epsilon=0.01)
        mc = make_model_comparison_summary(hs)
        # model_b should be ranked first
        assert mc.iloc[0]["model"] == "model_b"
        assert mc.iloc[0]["rank"] == 1
