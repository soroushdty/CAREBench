from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import yaml

from shared.utils.path_utils import is_remote_location as _is_remote_location

logger = logging.getLogger(__name__)

CRITICAL_PATH_KEYS = ("DIR_DATASET", "DIR_CONTEXT", "DIR_JSON_MAP", "DIR_OUTPUT")


def resolve_config_path(config_path: str | Path) -> Path:
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return resolved


def resolve_path(value: str | Path, *, base_dir: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) / candidate).resolve()


def resolve_path_or_remote(value: str | Path, *, base_dir: str | Path) -> str | Path:
    text = str(value).strip()
    if _is_remote_location(text):
        return text
    return resolve_path(text, base_dir=base_dir)


def infer_project_root(config_path: str | Path) -> Path:
    resolved = Path(config_path).expanduser().resolve()
    if resolved.parent.name in ("config", "configs"):
        return resolved.parent.parent
    return resolved.parent


def _validate_reproducibility_config(cfg: dict[str, Any]) -> None:
    repro = cfg.get("REPRODUCIBILITY")
    if not isinstance(repro, dict):
        raise ValueError("Config key 'REPRODUCIBILITY' must be an object.")

    python_version = repro.get("python_version")
    if not isinstance(python_version, str) or not python_version.strip():
        raise ValueError("REPRODUCIBILITY.python_version must be a non-empty string.")

    if not isinstance(repro.get("enforce_pinned_requirements"), bool):
        raise ValueError("REPRODUCIBILITY.enforce_pinned_requirements must be a boolean.")

    if "write_manifest" in repro:
        raise ValueError("REPRODUCIBILITY.write_manifest is no longer supported; use REPRODUCIBILITY.artifacts.run_manifest_json.")

    if "folder_enabled" in repro and not isinstance(repro.get("folder_enabled"), bool):
        raise ValueError("REPRODUCIBILITY.folder_enabled must be a boolean when provided.")

    if "folder_path" in repro and not isinstance(repro.get("folder_path"), str):
        raise ValueError("REPRODUCIBILITY.folder_path must be a string when provided.")

    artifacts = repro.get("artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, dict):
            raise ValueError("REPRODUCIBILITY.artifacts must be a mapping when provided.")
        for key, value in artifacts.items():
            if not isinstance(value, bool):
                raise ValueError(f"REPRODUCIBILITY.artifacts.{key} must be a boolean.")


def _validate_inference_config(cfg: dict[str, Any]) -> None:
    inference = cfg.get("inference")
    if inference is None:
        return

    if not isinstance(inference, dict):
        raise ValueError("Config key 'inference' must be an object when provided.")

    if "persistent_models" in inference and not isinstance(inference["persistent_models"], bool):
        raise ValueError("inference.persistent_models must be a boolean.")

    if "model_cache_size" in inference:
        size = inference["model_cache_size"]
        if not isinstance(size, int) or size < 1:
            raise ValueError("inference.model_cache_size must be a positive integer.")

    if "eager_materialize" in inference and not isinstance(inference["eager_materialize"], bool):
        raise ValueError("inference.eager_materialize must be a boolean.")


def validate_critical_config(cfg: dict[str, Any]) -> None:
    """Validate that all required top-level config keys are present and well-formed.

    Validated keys (non-exhaustive):

    - ``DIR_DATASET``, ``DIR_CONTEXT``, ``DIR_JSON_MAP``, ``DIR_OUTPUT`` —
      must be non-empty strings; ``DIR_OUTPUT`` must be a local path.
    - ``REPRODUCIBILITY`` block — must contain ``python_version`` (non-empty
      string) and ``enforce_pinned_requirements`` (bool). The deprecated key
      ``write_manifest`` is rejected; ``artifacts`` values must all be bools.
    - ``llm`` and ``llm_revision`` — must be non-empty strings.
    - ``hf_local_files_only`` — must be a bool.

    Args:
        cfg: Resolved configuration dict (already path-expanded).

    Raises:
        ValueError: On any missing or malformed required key.
    """
    missing_paths = [key for key in CRITICAL_PATH_KEYS if key not in cfg or not str(cfg.get(key)).strip()]
    if missing_paths:
        raise ValueError(f"Missing required config path keys: {', '.join(missing_paths)}")

    _validate_reproducibility_config(cfg)
    _validate_inference_config(cfg)

    if _is_remote_location(str(cfg.get("DIR_OUTPUT", ""))):
        raise ValueError("DIR_OUTPUT must be a local path, not a remote URL.")

    model_id = cfg.get("llm")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Config key 'llm' must be a non-empty string.")

    llm_revision = cfg.get("llm_revision")
    if not isinstance(llm_revision, str) or not llm_revision.strip():
        raise ValueError("Config key 'llm_revision' must be a non-empty string.")

    if not isinstance(cfg.get("hf_local_files_only"), bool):
        raise ValueError("Config key 'hf_local_files_only' must be a boolean.")


def resolve_config_paths(cfg: dict[str, Any], *, config_root: str | Path) -> dict[str, Any]:
    """Expand relative path values in the config to absolute paths.

    Resolves the keys ``DIR_DATASET``, ``DIR_CONTEXT``, ``DIR_JSON_MAP``, and
    ``DIR_OUTPUT`` relative to ``config_root``. Remote URLs (detected by
    :func:`shared.utils.path_utils.is_remote_location`) are passed through
    unchanged.

    Args:
        cfg: Raw config dict as loaded from YAML.
        config_root: Base directory used to resolve relative paths (typically
            the project root inferred from the config file location).

    Returns:
        dict: Shallow copy of ``cfg`` with the four path keys replaced by
            their absolute string representations.
    """
    resolved = dict(cfg)
    for key in ("DIR_DATASET", "DIR_CONTEXT", "DIR_JSON_MAP", "DIR_OUTPUT"):
        value = cfg.get(key)
        if value is not None:
            resolved[key] = str(resolve_path_or_remote(value, base_dir=config_root))
    return resolved


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load, validate, and return the resolved project configuration.

    Discovery and resolution algorithm:

    1. ``config_path`` is resolved to an absolute path and must exist.
    2. The project root is inferred: if the config lives in a ``config/`` or ``configs/``
       directory, the parent of that directory is used; otherwise the config's
       own parent directory is used.
    3. The YAML is parsed and must be a top-level mapping.
    4. Path values (``DIR_DATASET``, ``DIR_CONTEXT``, ``DIR_JSON_MAP``,
       ``DIR_OUTPUT``) are expanded relative to the project root via
       :func:`resolve_config_paths`.
    5. :func:`validate_critical_config` checks all required keys and raises
       ``ValueError`` on failure.
    6. ``PROJECT_ROOT`` and ``CONFIG_PATH`` are injected into the returned dict.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Fully resolved and validated configuration mapping.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the YAML is not a top-level mapping, or if required
            config keys are missing or malformed.
    """
    resolved_config_path = resolve_config_path(config_path)
    project_root = infer_project_root(resolved_config_path)

    logger.info("Loading configuration from %s...", resolved_config_path)
    with resolved_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be an object (dict) at the top level.")

    cfg = resolve_config_paths(cfg, config_root=project_root)
    validate_critical_config(cfg)
    cfg["PROJECT_ROOT"] = str(project_root)
    cfg["CONFIG_PATH"] = str(resolved_config_path)
    logger.info("Configuration loaded successfully.")
    return cfg
