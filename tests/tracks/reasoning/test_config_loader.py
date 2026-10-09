"""Tests for tracks/reasoning/config_loader.py."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from tracks.reasoning.config_loader import ConfigError, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONFIG: dict = {
    "backend": "dry_run",
    "model_ids": ["meta-llama/Llama-3.1-8B-Instruct"],
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
    "seed": 42,
    "retry_limit": 3,
    "request_timeout": 60,
    "shuffled_context_seed": 123,
    "bootstrap_seed": 42,
    "permutation_seed": 42,
    "n_bootstrap_resamples": 1000,
    "n_permutations": 10000,
    "output_dir": "output/reasoning",
    "cache_dir": "output/reasoning/cache",
    "scores_dir": "output/reasoning/scores",
    "reports_dir": "output/reasoning/reports",
    "data": {
        "dataset_path": "examples/synthetic/dataset.xlsx",
        "patient_summaries_path": "examples/synthetic/patient_summaries.json",
        "train_sheet": "train",
        "test_sheet": "test",
        "interview_sheet": "interview",
        "patient_col": "Patient",
        "physician_col": "Physician",
        "item_col": "Item",
        "classes": [
            "behavioral_health",
            "diagnoses",
            "disabilities",
            "infectious_diseases",
            "genetics",
            "medications",
            "sexual_reproductive_health",
            "social_determinants_of_health",
            "violence",
            "other",
        ],
    },
}


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    """Write a config dict to a temporary YAML file and return its path."""
    p = tmp_path / "reasoning_config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


def _config_without(key: str) -> dict:
    """Return a copy of VALID_CONFIG with one top-level key removed."""
    cfg = dict(VALID_CONFIG)
    cfg.pop(key)
    return cfg


def _config_without_data_key(data_key: str) -> dict:
    """Return a copy of VALID_CONFIG with one data sub-key removed."""
    import copy
    cfg = copy.deepcopy(VALID_CONFIG)
    cfg["data"].pop(data_key)
    return cfg


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestLoadConfigValid:
    def test_dry_run_loads_successfully(self, tmp_path):
        """A complete dry_run config loads without error."""
        p = _write_config(tmp_path, VALID_CONFIG)
        cfg = load_config(p)
        assert cfg["backend"] == "dry_run"
        assert cfg["model_ids"] == ["meta-llama/Llama-3.1-8B-Instruct"]

    def test_huggingface_with_env_var(self, tmp_path, monkeypatch):
        """huggingface backend succeeds when the named env var is set."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        import copy
        cfg_dict = copy.deepcopy(VALID_CONFIG)
        cfg_dict["backend"] = "huggingface"
        cfg_dict["hf_token_env"] = "HF_TOKEN"
        p = _write_config(tmp_path, cfg_dict)
        cfg = load_config(p)
        assert cfg["backend"] == "huggingface"

    def test_huggingface_with_direct_token(self, tmp_path):
        """huggingface backend succeeds when hf_token is set directly."""
        import copy
        cfg_dict = copy.deepcopy(VALID_CONFIG)
        cfg_dict["backend"] = "huggingface"
        cfg_dict["hf_token"] = "hf_direct_token"
        p = _write_config(tmp_path, cfg_dict)
        cfg = load_config(p)
        assert cfg["backend"] == "huggingface"

    def test_multiple_model_ids(self, tmp_path):
        """Multiple unique model IDs are accepted."""
        import copy
        cfg_dict = copy.deepcopy(VALID_CONFIG)
        cfg_dict["model_ids"] = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
        p = _write_config(tmp_path, cfg_dict)
        cfg = load_config(p)
        assert len(cfg["model_ids"]) == 2

    def test_returns_dict(self, tmp_path):
        """load_config returns a dict."""
        p = _write_config(tmp_path, VALID_CONFIG)
        cfg = load_config(p)
        assert isinstance(cfg, dict)

    def test_actual_config_file_loads(self):
        """The shipped reasoning_config.yaml loads in dry_run mode (no HF token needed)."""
        config_path = Path(__file__).resolve().parents[3] / "configs" / "reasoning_config.yaml"
        # The shipped config uses huggingface backend; load it after patching to dry_run
        import copy
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["backend"] = "dry_run"
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.dump(raw, fh)
            tmp = fh.name
        try:
            cfg = load_config(tmp)
            assert "data" in cfg
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Missing required top-level keys
# ---------------------------------------------------------------------------

class TestMissingTopLevelKeys:
    @pytest.mark.parametrize("key", [
        "backend", "model_ids", "temperature", "max_tokens", "top_p", "seed",
        "retry_limit", "request_timeout", "shuffled_context_seed",
        "bootstrap_seed", "permutation_seed", "n_bootstrap_resamples",
        "n_permutations", "output_dir", "cache_dir", "scores_dir",
        "reports_dir", "data",
    ])
    def test_missing_key_raises_config_error(self, tmp_path, key):
        """Each missing required top-level key raises ConfigError."""
        p = _write_config(tmp_path, _config_without(key))
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        assert key in str(exc_info.value)

    def test_error_message_contains_missing_key_name(self, tmp_path):
        """ConfigError message names the missing parameter."""
        p = _write_config(tmp_path, _config_without("n_permutations"))
        with pytest.raises(ConfigError, match="n_permutations"):
            load_config(p)


# ---------------------------------------------------------------------------
# Missing required data sub-keys
# ---------------------------------------------------------------------------

class TestMissingDataKeys:
    @pytest.mark.parametrize("data_key", [
        "dataset_path", "patient_summaries_path", "train_sheet", "test_sheet",
        "interview_sheet", "patient_col", "physician_col", "item_col", "classes",
    ])
    def test_missing_data_key_raises_config_error(self, tmp_path, data_key):
        """Each missing required data sub-key raises ConfigError."""
        p = _write_config(tmp_path, _config_without_data_key(data_key))
        with pytest.raises(ConfigError) as exc_info:
            load_config(p)
        assert data_key in str(exc_info.value)


# ---------------------------------------------------------------------------
# model_ids validation (sub-task 2.3)
# ---------------------------------------------------------------------------

class TestModelIdsValidation:
    def test_empty_list_raises(self, tmp_path):
        """An empty model_ids list raises ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["model_ids"] = []
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="model_ids"):
            load_config(p)

    def test_not_a_list_raises(self, tmp_path):
        """model_ids that is not a list raises ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["model_ids"] = "meta-llama/Llama-3.1-8B-Instruct"
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="model_ids"):
            load_config(p)

    def test_empty_string_model_id_raises(self, tmp_path):
        """An empty string in model_ids raises ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["model_ids"] = [""]
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="model_ids"):
            load_config(p)

    def test_whitespace_only_model_id_raises(self, tmp_path):
        """A whitespace-only string in model_ids raises ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["model_ids"] = ["   "]
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="model_ids"):
            load_config(p)

    def test_duplicate_model_ids_raises(self, tmp_path):
        """Duplicate model IDs raise ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["model_ids"] = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="duplicate"):
            load_config(p)

    def test_non_string_model_id_raises(self, tmp_path):
        """A non-string entry in model_ids raises ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["model_ids"] = [42]
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="model_ids"):
            load_config(p)


# ---------------------------------------------------------------------------
# HF credential validation (sub-task 2.4)
# ---------------------------------------------------------------------------

class TestHFCredentialValidation:
    def test_huggingface_no_token_raises(self, tmp_path, monkeypatch):
        """huggingface backend without any token raises ConfigError."""
        # Ensure the env var is not set
        monkeypatch.delenv("HF_TOKEN", raising=False)
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["backend"] = "huggingface"
        cfg.pop("hf_token", None)
        cfg["hf_token_env"] = "HF_TOKEN"
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="huggingface"):
            load_config(p)

    def test_huggingface_empty_env_var_raises(self, tmp_path, monkeypatch):
        """huggingface backend with an empty env var raises ConfigError."""
        monkeypatch.setenv("HF_TOKEN", "")
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["backend"] = "huggingface"
        cfg.pop("hf_token", None)
        cfg["hf_token_env"] = "HF_TOKEN"
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError):
            load_config(p)

    def test_dry_run_no_token_succeeds(self, tmp_path):
        """dry_run backend does not require any HF credentials."""
        p = _write_config(tmp_path, VALID_CONFIG)  # backend=dry_run, no token
        cfg = load_config(p)
        assert cfg["backend"] == "dry_run"

    def test_huggingface_no_hf_token_env_key_raises(self, tmp_path, monkeypatch):
        """huggingface backend with no hf_token and no hf_token_env key raises ConfigError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["backend"] = "huggingface"
        cfg.pop("hf_token", None)
        cfg.pop("hf_token_env", None)
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError):
            load_config(p)


# ---------------------------------------------------------------------------
# Invalid backend
# ---------------------------------------------------------------------------

class TestInvalidBackend:
    def test_unknown_backend_raises(self, tmp_path):
        """An unrecognised backend value raises ConfigError."""
        import copy
        cfg = copy.deepcopy(VALID_CONFIG)
        cfg["backend"] = "openai"
        p = _write_config(tmp_path, cfg)
        with pytest.raises(ConfigError, match="backend"):
            load_config(p)


# ---------------------------------------------------------------------------
# File-level errors
# ---------------------------------------------------------------------------

class TestFileErrors:
    def test_missing_file_raises(self, tmp_path):
        """A non-existent config file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        """A file with invalid YAML raises ConfigError."""
        p = tmp_path / "bad.yaml"
        p.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)

    def test_non_mapping_yaml_raises(self, tmp_path):
        """A YAML file that is not a mapping raises ConfigError."""
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)
