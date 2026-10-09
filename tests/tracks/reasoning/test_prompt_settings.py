"""Configurable prompt wording and stale-cache detection in Track 3 (#5)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shared.evaluation.report_generator import ReportGenerator
from tracks.reasoning.bundle.report import generate_markdown_report
from tracks.reasoning.config_loader import ConfigError, validate_prompt_config
from tracks.reasoning.prompt_template import DEFAULT_INTRO, PromptTemplate
from tracks.reasoning.run_assay import main

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTRO = "You are a clinical documentation expert. Classify the following EHR item into note sections."


class TestPromptTemplate:
    def test_defaults(self):
        prompt = PromptTemplate().build_context_free_prompt("x")
        assert prompt.startswith(DEFAULT_INTRO + "\n")
        assert "following ten privacy categories," in prompt

    def test_intro_and_category_type(self):
        template = PromptTemplate(intro=_INTRO, category_type="note-section")
        for prompt in (
            template.build_context_free_prompt("x"),
            template.build_correct_context_prompt("x", "ctx"),
            template.build_shuffled_context_prompt("x", "ctx"),
        ):
            assert prompt.startswith(_INTRO + "\n")
            assert "following ten note-section categories," in prompt
            assert "privacy" not in prompt

    def test_empty_category_type(self):
        prompt = PromptTemplate(category_type="").build_context_free_prompt("x")
        assert "following ten categories," in prompt

    def test_from_config(self):
        template = PromptTemplate.from_config({"intro": _INTRO, "category_type": ""})
        assert template.build_context_free_prompt("x").startswith(_INTRO)
        assert template.category_type == ""
        assert PromptTemplate.from_config(None).build_context_free_prompt("x") == (
            PromptTemplate().build_context_free_prompt("x")
        )

    def test_empty_intro_rejected(self):
        with pytest.raises(ValueError, match="intro"):
            PromptTemplate(intro="  ")


class TestPromptConfigValidation:
    @pytest.mark.parametrize(
        "prompt_cfg",
        [None, {}, {"intro": _INTRO}, {"category_type": ""}, {"category_type": "safety"}],
    )
    def test_valid(self, prompt_cfg):
        validate_prompt_config(prompt_cfg)

    @pytest.mark.parametrize(
        "prompt_cfg, match",
        [
            ("text", "mapping"),
            ({"intro": ""}, "non-empty"),
            ({"intro": 3}, "non-empty"),
            ({"category_type": None, "tone": "x"}, "Unknown"),
            ({"category_type": 1}, "string"),
        ],
    )
    def test_invalid(self, prompt_cfg, match):
        with pytest.raises(ConfigError, match=match):
            validate_prompt_config(prompt_cfg)


def test_run_report_uses_category_type(tmp_path):
    report = ReportGenerator(
        h1_result={}, h2_result={}, h3_result={}, h4_result={},
        model_id="m", run_timestamp="2026-01-01T00:00:00Z",
        category_names=["a", "b"], category_type="",
    )
    path = tmp_path / "report.md"
    report.generate(str(path))
    text = path.read_text()
    assert "privacy" not in text.lower()
    assert "2 categories (output_dimension)" in text


def test_bundle_report_uses_category_type(tmp_path):
    empty = pd.DataFrame()
    path = tmp_path / "report.md"
    generate_markdown_report(
        hypothesis_summary=pd.DataFrame(), h1_df=empty, h2_df=empty,
        h3_effects_df=empty, h3_corr_df=empty, h4_df=empty,
        model_comparison=empty, validation_summary=empty,
        output_path=str(path), categories=["a"], category_type="safety",
    )
    text = path.read_text()
    assert "privacy" not in text.lower()
    assert "safety-category judgments" in text


# ---------------------------------------------------------------------------
# Stale cache entries are run again when the prompt changes
# ---------------------------------------------------------------------------


def _run_dry(tmp_path: Path, monkeypatch, prompt_cfg: dict | None) -> int:
    monkeypatch.chdir(_REPO_ROOT)
    with open(_REPO_ROOT / "configs" / "assay_config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["output_dir"] = str(tmp_path / "output")
    cfg["cache_dir"] = str(tmp_path / "cache")
    if prompt_cfg is not None:
        cfg["prompt"] = prompt_cfg
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return main(["--config", str(config_path), "--dry_run", "--model", "dry_run_example"])


def _latest_log(tmp_path: Path) -> str:
    run_dirs = sorted((tmp_path / "output").glob("run_*"))
    return (run_dirs[-1] / "log.txt").read_text()


def _cached_hashes(tmp_path: Path) -> dict[str, str]:
    return {
        str(p): json.loads(p.read_text())["prompt_hash"]
        for p in sorted((tmp_path / "cache").rglob("*.json"))
    }


def test_changed_prompt_reruns_cached_responses(tmp_path, monkeypatch):
    assert _run_dry(tmp_path, monkeypatch, None) == 0
    first = _cached_hashes(tmp_path)
    assert first

    assert _run_dry(tmp_path, monkeypatch, None) == 0
    log = _latest_log(tmp_path)
    assert "different prompt" not in log
    assert f"0 items to infer ({len(first)} cached)" in log
    assert _cached_hashes(tmp_path) == first

    assert _run_dry(tmp_path, monkeypatch, {"intro": _INTRO}) == 0
    log = _latest_log(tmp_path)
    assert f"{len(first)} cached response(s) were made with a different prompt" in log
    third = _cached_hashes(tmp_path)
    assert third.keys() == first.keys()
    assert all(third[k] != first[k] for k in first)
