"""
Analysis bundle generation for the LLM context-shift assay.

Post-hoc pipeline that consumes pre-generated score CSVs and produces a full
reproducible analysis bundle (paired cell deltas, endpoint results, report, zip).
"""

from tracks.reasoning.bundle.loader import load_physician_consensus, load_llm_scores
from tracks.reasoning.bundle.validator import validate_inputs
from tracks.reasoning.bundle.delta_builder import build_paired_cell_deltas
from tracks.reasoning.bundle.hypotheses import compute_h1, compute_h2, compute_h3, compute_h4
from tracks.reasoning.bundle.summary import make_hypothesis_summary, make_model_comparison_summary
from tracks.reasoning.bundle.report import generate_markdown_report
from tracks.reasoning.bundle.bundle import write_analysis_bundle

__all__ = [
    "load_physician_consensus",
    "load_llm_scores",
    "validate_inputs",
    "build_paired_cell_deltas",
    "compute_h1",
    "compute_h2",
    "compute_h3",
    "compute_h4",
    "make_hypothesis_summary",
    "make_model_comparison_summary",
    "generate_markdown_report",
    "write_analysis_bundle",
]
