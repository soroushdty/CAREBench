"""
Configuration loader for Track 3 (reasoning).

Loads and validates reasoning_config.yaml, raising ConfigError with details
for any missing or invalid parameter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when the reasoning configuration is invalid or incomplete."""


# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS: list[str] = [
    "backend",
    "model_ids",
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "retry_limit",
    "request_timeout",
    "shuffled_context_seed",
    "bootstrap_seed",
    "permutation_seed",
    "n_bootstrap_resamples",
    "n_permutations",
    "output_dir",
    "cache_dir",
    "scores_dir",
    "reports_dir",
    "data",
]

# Optional top-level keys and their defaults (applied after validation).
OPTIONAL_TOP_LEVEL_DEFAULTS: dict[str, object] = {
    "skip_label_leakage_check": False,
}

REQUIRED_DATA_KEYS: list[str] = [
    "dataset_path",
    "patient_summaries_path",
    "train_sheet",
    "test_sheet",
    "interview_sheet",
    "patient_col",
    "physician_col",
    "item_col",
    "classes",
]

VALID_BACKENDS: set[str] = {"huggingface", "dry_run", "local_transformers"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_label_space_config(data_section: dict[str, Any]) -> None:
    """Check that ``classes`` and ``class_definitions`` form a valid label space."""
    from shared.label_space import LabelSpace

    try:
        LabelSpace.from_config(
            data_section["classes"], data_section.get("class_definitions")
        )
    except ValueError as exc:
        raise ConfigError(f"Invalid label space in 'data': {exc}") from exc


_PROMPT_KEYS = {"intro", "category_type"}


def validate_prompt_config(prompt_cfg: Any) -> None:
    """Check the optional ``prompt`` section (``intro``, ``category_type``)."""
    if prompt_cfg is None:
        return
    if not isinstance(prompt_cfg, dict):
        raise ConfigError("The 'prompt' configuration key must be a mapping.")
    unknown = sorted(set(prompt_cfg) - _PROMPT_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown 'prompt' configuration key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(_PROMPT_KEYS))}."
        )
    intro = prompt_cfg.get("intro")
    if intro is not None and (not isinstance(intro, str) or not intro.strip()):
        raise ConfigError("'prompt.intro' must be a non-empty string.")
    category_type = prompt_cfg.get("category_type")
    if category_type is not None and not isinstance(category_type, str):
        raise ConfigError("'prompt.category_type' must be a string (use \"\" for none).")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the reasoning configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    ConfigError
        If the file cannot be read, any required parameter is missing,
        or any validation rule is violated.
    """
    config_path = Path(config_path)

    # --- Load YAML -----------------------------------------------------------
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML configuration: {exc}")

    if not isinstance(cfg, dict):
        raise ConfigError("Configuration file must contain a YAML mapping at the top level.")

    # --- Validate top-level required keys ------------------------------------
    missing_top = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing_top:
        raise ConfigError(
            f"Missing required configuration parameter(s): {', '.join(missing_top)}"
        )
    for key, default in OPTIONAL_TOP_LEVEL_DEFAULTS.items():
        cfg.setdefault(key, default)

    # --- Validate backend ----------------------------------------------------
    backend: str = cfg["backend"]
    if backend not in VALID_BACKENDS:
        raise ConfigError(
            f"Invalid backend '{backend}'. Must be one of: {', '.join(sorted(VALID_BACKENDS))}"
        )

    # --- Validate model_ids --------------------------------------------------
    _validate_model_ids(cfg["model_ids"])

    # --- Validate HF credentials for huggingface backend --------------------
    if backend == "huggingface":
        _validate_hf_credentials(cfg)
    elif backend == "local_transformers":
        _validate_hf_credentials_optional(cfg)

    # --- Validate data section -----------------------------------------------
    data_section = cfg["data"]
    if not isinstance(data_section, dict):
        raise ConfigError("The 'data' configuration key must be a mapping.")

    missing_data = [k for k in REQUIRED_DATA_KEYS if k not in data_section]
    if missing_data:
        raise ConfigError(
            f"Missing required 'data' configuration parameter(s): {', '.join(missing_data)}"
        )

    validate_label_space_config(data_section)
    validate_prompt_config(cfg.get("prompt"))
    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_model_ids(model_ids: Any) -> None:
    """Validate that model_ids is a non-empty list of unique, non-empty strings.

    Raises
    ------
    ConfigError
        On any violation.
    """
    if not isinstance(model_ids, list):
        raise ConfigError("'model_ids' must be a list.")

    if len(model_ids) == 0:
        raise ConfigError("'model_ids' must contain at least one model ID.")

    for i, mid in enumerate(model_ids):
        if not isinstance(mid, str) or not mid.strip():
            raise ConfigError(
                f"'model_ids[{i}]' must be a non-empty string, got: {mid!r}"
            )

    # Check uniqueness
    seen: set[str] = set()
    duplicates: list[str] = []
    for mid in model_ids:
        if mid in seen:
            duplicates.append(mid)
        seen.add(mid)

    if duplicates:
        raise ConfigError(
            f"'model_ids' contains duplicate entries: {', '.join(duplicates)}"
        )


def _validate_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-loaded config dictionary (useful for unit tests).

    Runs the same validation logic as :func:`load_config` but skips YAML
    parsing.  Raises :class:`ConfigError` on any violation.
    """
    if not isinstance(cfg, dict):
        raise ConfigError("Configuration must be a mapping.")

    missing_top = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing_top:
        raise ConfigError(
            f"Missing required configuration parameter(s): {', '.join(missing_top)}"
        )
    for key, default in OPTIONAL_TOP_LEVEL_DEFAULTS.items():
        cfg.setdefault(key, default)

    backend: str = cfg["backend"]
    if backend not in VALID_BACKENDS:
        raise ConfigError(
            f"Invalid backend '{backend}'. Must be one of: {', '.join(sorted(VALID_BACKENDS))}"
        )

    _validate_model_ids(cfg["model_ids"])

    if backend == "huggingface":
        _validate_hf_credentials(cfg)
    elif backend == "local_transformers":
        _validate_hf_credentials_optional(cfg)

    data_section = cfg["data"]
    if not isinstance(data_section, dict):
        raise ConfigError("The 'data' configuration key must be a mapping.")

    missing_data = [k for k in REQUIRED_DATA_KEYS if k not in data_section]
    if missing_data:
        raise ConfigError(
            f"Missing required 'data' configuration parameter(s): {', '.join(missing_data)}"
        )

    validate_label_space_config(data_section)
    validate_prompt_config(cfg.get("prompt"))
    return cfg


def _validate_hf_credentials(cfg: dict[str, Any]) -> None:
    """Validate that a Hugging Face token or credential source is available.

    Accepts any of:
    - ``hf_token`` key present in the config (direct token value)
    - ``hf_token_env`` key present and the named environment variable is set

    Raises
    ------
    ConfigError
        If neither a direct token nor a resolvable environment variable is found.
    """
    # Direct token in config
    if cfg.get("hf_token"):
        return

    # Environment variable name provided and the variable is set
    hf_token_env: str | None = cfg.get("hf_token_env")
    if hf_token_env and os.environ.get(hf_token_env):
        return

    # Neither source is available
    raise ConfigError(
        "backend=huggingface requires a Hugging Face token. "
        "Set 'hf_token' in the config or provide 'hf_token_env' pointing to a "
        "non-empty environment variable (e.g., HF_TOKEN)."
    )


def _validate_hf_credentials_optional(cfg: dict[str, Any]) -> None:
    """Warn (but do not raise) when no HF token is configured for local_transformers.

    Model weights may already be cached locally, so a token is not a hard
    requirement.  If a token IS needed for the initial download the user sets
    it via the existing ``hf_token`` / ``hf_token_env`` mechanism.
    """
    # Nothing to enforce — token is optional for local_transformers.
    return
