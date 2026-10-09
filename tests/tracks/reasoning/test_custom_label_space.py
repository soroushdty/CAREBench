"""Track 3 with a label space other than the default ten categories (#5)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from adapters.paired_context.reasoning_adapter import PairedDataset
from shared.evaluation.report_generator import ReportGenerator
from shared.label_space import LabelSpace
from tracks.reasoning.config_loader import ConfigError, _validate_config_dict
from tracks.reasoning.llm_client import LLMClient
from tracks.reasoning.prompt_template import PromptTemplate
from tracks.reasoning.response_cache import ResponseCache
from tracks.reasoning.schema_validator import MissingKeyError, validate_response
from tracks.reasoning.score_parser import ScoreParser

CLASSES = ["Mood & anxiety", "Drug therapy", "Misc. sensitive"]
DEFINITIONS = {
    "Mood & anxiety": "Depression, anxiety, and related conditions",
    "Drug therapy": {"key": "medications", "definition": "Any prescribed drug"},
}
SPACE = LabelSpace.from_config(CLASSES, DEFINITIONS)
KEYS = ["mood_anxiety", "medications", "misc_sensitive"]


def _dry_run_cfg() -> dict:
    return {"backend": "dry_run", "retry_limit": 1}


class TestPrompt:
    def test_lists_configured_categories_and_definitions(self):
        prompt = PromptTemplate(SPACE).build_context_free_prompt("sertraline 50mg")
        assert "following three privacy categories" in prompt
        assert "- mood_anxiety: Depression, anxiety, and related conditions" in prompt
        assert "- medications: Any prescribed drug" in prompt
        assert "- misc_sensitive\n" in prompt
        assert "behavioral_health" not in prompt
        json_block = prompt[prompt.index("{"):]
        assert [line.split('"')[1] for line in json_block.splitlines() if '"' in line] == KEYS

    def test_context_prompts_use_the_same_categories(self):
        template = PromptTemplate(SPACE)
        for prompt in (
            template.build_correct_context_prompt("x", "ctx"),
            template.build_shuffled_context_prompt("x", "ctx"),
        ):
            assert "PATIENT CONTEXT:\nctx" in prompt
            assert '"misc_sensitive": <score>' in prompt

    def test_single_category_header(self):
        space = LabelSpace.from_config(["Housing"])
        prompt = PromptTemplate(space).build_context_free_prompt("x")
        assert "following one privacy category," in prompt


class TestValidation:
    def test_accepts_configured_keys(self):
        raw = json.dumps({"mood_anxiety": 0.2, "medications": 1, "misc_sensitive": 0})
        assert validate_response(raw, SPACE) == {
            "mood_anxiety": 0.2, "medications": 1.0, "misc_sensitive": 0.0,
        }

    def test_reports_missing_configured_key(self):
        raw = json.dumps({"mood_anxiety": 0.2, "medications": 1})
        with pytest.raises(MissingKeyError) as exc_info:
            validate_response(raw, SPACE)
        assert exc_info.value.missing_keys == ["misc_sensitive"]

    def test_dry_run_client_returns_configured_keys(self):
        client = LLMClient(_dry_run_cfg(), label_space=SPACE)
        response = client.call("p", "model", "context_free", "1", "item")
        assert response.error is None
        assert sorted(json.loads(response.raw_text)) == sorted(KEYS)


class TestScoreParser:
    def test_writes_configured_columns(self, tmp_path):
        client = LLMClient(_dry_run_cfg(), label_space=SPACE)
        cache = ResponseCache(tmp_path / "cache")
        for condition in ("context_free", "correct_context", "shuffled_context"):
            cache.put(client.call("p", "org/model", condition, "1", "item"))

        dataset = PairedDataset(
            patient_ids=np.array(["1"]),
            item_texts=np.array(["item"]),
            survey_consensus=np.zeros((1, 3)),
            interview_consensus=np.zeros((1, 3)),
            delta_physician=np.zeros((1, 3)),
            category_names=CLASSES,
            label_space=SPACE,
        )
        parser = ScoreParser(tmp_path / "cache", tmp_path / "scores", dataset)
        result = parser.parse_all("org/model")

        assert result["errors_written"] == 0
        df = pd.read_csv(result["output_dir"] / "context_free_scores.csv")
        assert list(df.columns) == ["patient_id", "item_text", *KEYS, "model_id", "timestamp"]


class TestPairedDataset:
    def _arrays(self):
        z = np.zeros((1, 3))
        return {
            "patient_ids": np.array(["1"]), "item_texts": np.array(["x"]),
            "survey_consensus": z, "interview_consensus": z, "delta_physician": z,
        }

    def test_canonical_names_come_from_label_space(self):
        ds = PairedDataset(**self._arrays(), category_names=CLASSES, label_space=SPACE)
        assert ds.canonical_category_names == KEYS

    def test_label_space_defaults_from_category_names(self):
        ds = PairedDataset(**self._arrays(), category_names=CLASSES)
        assert ds.canonical_category_names == ["mood_anxiety", "drug_therapy", "misc_sensitive"]

    def test_mismatched_label_space_rejected(self):
        with pytest.raises(ValueError, match="do not match"):
            PairedDataset(
                **self._arrays(), category_names=["A", "B", "C"], label_space=SPACE
            )


def test_report_looks_up_keys_and_shows_display_names(tmp_path):
    h1 = {
        "mean_abs_delta": 0.1, "ci_lower": 0.0, "ci_upper": 0.2,
        "per_category": {
            k: {"mean_abs_delta": 0.25, "ci_lower": 0.1, "ci_upper": 0.3} for k in KEYS
        },
    }
    report = ReportGenerator(
        h1_result=h1, h2_result={}, h3_result={}, h4_result={},
        model_id="org/model", run_timestamp="2026-01-01T00:00:00Z",
        label_space=SPACE,
    )
    path = tmp_path / "report.md"
    report.generate(str(path))
    text = path.read_text()
    for name in CLASSES:
        assert f"| {name} | 0.2500 | [0.1000, 0.3000] |" in text


class TestConfigValidation:
    def _cfg(self, **data_overrides) -> dict:
        data = {
            "dataset_path": "d.xlsx", "patient_summaries_path": "s.json",
            "train_sheet": "train", "test_sheet": "test", "interview_sheet": "interview",
            "patient_col": "Patient", "physician_col": "Physician", "item_col": "Item",
            "classes": CLASSES,
        }
        data.update(data_overrides)
        return {
            "backend": "dry_run", "model_ids": ["m"], "temperature": 0.0,
            "max_tokens": 16, "top_p": 1.0, "seed": 1, "retry_limit": 1,
            "request_timeout": 1, "shuffled_context_seed": 1, "bootstrap_seed": 1,
            "permutation_seed": 1, "n_bootstrap_resamples": 10, "n_permutations": 10,
            "output_dir": "o", "cache_dir": "c", "scores_dir": "s", "reports_dir": "r",
            "data": data,
        }

    def test_custom_classes_and_definitions_accepted(self):
        _validate_config_dict(self._cfg(class_definitions=DEFINITIONS))

    def test_definition_for_unknown_class_is_config_error(self):
        with pytest.raises(ConfigError, match="Invalid label space"):
            _validate_config_dict(self._cfg(class_definitions={"Nope": "x"}))
