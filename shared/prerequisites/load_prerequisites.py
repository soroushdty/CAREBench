from __future__ import annotations

import platform
import re
from pathlib import Path
import yaml

import logging

from .load_config import load_config, resolve_path_or_remote
from shared.utils.path_utils import is_remote_location as _is_remote_location
from .logging_config import logging_config
from .requirements_utils import requirements_utils
from .reproducibility_manifest import write_reproducibility_manifest
from .run_timestamping import apply_timestamped_run_folder

logger = logging.getLogger(__name__)


def check_path_exists(path: str | Path, *, kind: str = "any") -> Path:
    path = Path(path)

    if kind not in {"any", "file", "dir"}:
        raise ValueError("kind must be one of: 'any', 'file', 'dir'")

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Expected a file, got: {path}")

    if kind == "dir" and not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")

    return path


def _parse_version_tuple(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?\s*", version)
    if not match:
        raise ValueError(
            "Invalid Python version format. Expected 'X.Y' or 'X.Y.Z' in "
            "REPRODUCIBILITY.python_version."
        )
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    return major, minor, patch


def _is_version_satisfied(current: str, required: str) -> bool:
    # Accept either exact pin (e.g. "3.14.3") or comparator specs (e.g. ">=3.10").
    spec_match = re.fullmatch(r"\s*(>=|<=|==|>|<)\s*(\d+\.\d+(?:\.\d+)?)\s*", required)
    current_tuple = _parse_version_tuple(current)

    if spec_match:
        operator = spec_match.group(1)
        required_tuple = _parse_version_tuple(spec_match.group(2))

        if operator == ">=":
            return current_tuple >= required_tuple
        if operator == "<=":
            return current_tuple <= required_tuple
        if operator == "==":
            return current_tuple == required_tuple
        if operator == ">":
            return current_tuple > required_tuple
        if operator == "<":
            return current_tuple < required_tuple

    # Bare versions are treated as exact pins.
    required_tuple = _parse_version_tuple(required)
    return current_tuple == required_tuple


def _enforce_python_version(cfg: dict) -> None:
    required = str(cfg.get("REPRODUCIBILITY", {}).get("python_version", "")).strip()
    current = platform.python_version()
    if not required:
        raise ValueError("REPRODUCIBILITY.python_version is required.")
    if not _is_version_satisfied(current, required):
        raise RuntimeError(
            f"Python version mismatch: required {required}, active {current}. "
            "Use the pinned interpreter before running prerequisites."
        )


def _validate_local_input_artifacts(cfg: dict, *, project_dir: Path) -> None:
    input_keys = ("DIR_DATASET", "DIR_CONTEXT", "DIR_JSON_MAP")
    allow_remote = bool(cfg.get("REPRODUCIBILITY", {}).get("allow_remote_inputs", False))
    missing_local_paths: list[str] = []

    for key in input_keys:
        raw_value = cfg.get(key)
        if not raw_value:
            continue

        value = str(raw_value).strip()
        if _is_remote_location(value):
            if not allow_remote:
                raise RuntimeError(
                    f"Remote input '{key}={value}' is not allowed in reproducible mode. "
                    "Set REPRODUCIBILITY.allow_remote_inputs=true only when you have explicit integrity controls."
                )
            continue

        candidate = resolve_path_or_remote(value, base_dir=project_dir)
        candidate_path = Path(candidate)
        if not (candidate_path.exists() and candidate_path.is_file()):
            missing_local_paths.append(f"{key}='{candidate_path}'")

    if missing_local_paths:
        raise FileNotFoundError(
            "Configured local input files are missing or invalid: " + ", ".join(missing_local_paths)
        )


def _resolve_project_dir(project_dir: str | Path | None, config_path: str | Path | None) -> Path:
    if project_dir is not None:
        return Path(project_dir).expanduser().resolve()
    if config_path is not None:
        resolved_config = Path(config_path).expanduser().resolve()
        if resolved_config.parent.name in ("config", "configs"):
            return resolved_config.parent.parent
        return resolved_config.parent
    return Path(__file__).resolve().parents[2]



def load_prerequisites(
    project_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    config_path: str | Path | None = None,
    requirements_path: str | Path | None = None,
    log_level: str | int | None = None,
    write_manifest: bool | None = None,
    resume_from_dir: str | Path | None = None,
) -> dict:
    """Bootstrap the full pipeline environment and return the resolved config dict.

    Order of operations:

    1. **Project root** — resolved from ``project_dir``, ``config_path``, or
       the package location as a fallback.
    2. **Config loading** — ``config/config.yaml`` is loaded and validated;
       ``config/training_config.yaml`` is merged in (takes precedence) if found.
    3. **Requirements validation** — packages are checked against the pinned
       ``requirements.txt`` (or the explicit ``requirements_path`` override).
    4. **Python-version check** — ``REPRODUCIBILITY.python_version`` is enforced.
    5. **Input-artifact validation** — configured local input paths are verified
       to exist on disk.
    6. **Run folder creation** — a timestamped output directory is created under
       ``DIR_OUTPUT`` and ``RUN_ID`` / ``DIR_MODEL`` are injected into ``cfg``.
       Skipped when ``resume_from_dir`` is given; in that case ``DIR_MODEL`` is
       pointed at the existing checkpoint directory instead.
    7. **Logging** — the root logger is configured with a file handler at
       ``log_path`` (or ``<run_root>/log.txt`` by default).
    8. **Reproducibility manifest** — ``run_manifest.json`` and
       ``run_config.yaml`` are written to ``DIR_REPRODUCIBILITY`` when the
       corresponding artifact toggles are enabled.

    **Side effects**: configures the root logger (via ``logging.basicConfig``
    with ``force=True``), sets global random seeds via
    ``enforce_reproducibility``, creates the run output directory on disk.

    Args:
        project_dir: Explicit project root directory. Inferred from
            ``config_path`` or the package location when ``None``.
        log_path: Override for the log file path. Falls back to
            ``LOG_PATH`` from config, then ``<run_root>/log.txt``.
        config_path: Path to the primary YAML config file. Defaults to
            ``<project_dir>/configs/main_config.yaml``.
        requirements_path: Override for the requirements file. Defaults to
            ``<project_dir>/requirements.txt``.
        log_level: Log level (name string or integer) passed to
            :func:`logging_config`. Overridden by ``LOG_LEVEL`` in config
            when that key is present.
        write_manifest: Whether to write ``run_manifest.json``. ``None`` defers
            to ``REPRODUCIBILITY.artifacts.run_manifest_json`` in config.
            ``False`` suppresses the manifest regardless of config.
        resume_from_dir: Path to an existing ``DIR_MODEL`` directory from a
            previous run. When provided, run-folder creation is skipped and
            ``DIR_MODEL`` / ``RUN_ID`` are set to point at the existing
            checkpoint.

    Returns:
        dict: Fully resolved configuration mapping. Notable injected keys:
            ``PROJECT_ROOT``, ``DIR_OUTPUT``, ``DIR_MODEL``, ``RUN_ID``,
            ``ACTIVE_REQUIREMENTS_PATH``, ``CONFIG_PATH``.

    Raises:
        FileNotFoundError: If ``project_dir``, ``config_path``,
            ``requirements_path``, or ``resume_from_dir`` does not exist on
            disk.
        RuntimeError: On Python-version mismatch, failed requirements
            validation, logging setup failure, or manifest write failure.
        ValueError: If ``REPRODUCIBILITY`` config block is missing or malformed.
    """
    project_dir = _resolve_project_dir(project_dir, config_path)
    check_path_exists(project_dir, kind="dir")

    config_path = Path(config_path).resolve() if config_path is not None else project_dir / "configs" / "main_config.yaml"
    check_path_exists(config_path, kind="file")

    try:
        logger.info("Loading configuration file: %s", config_path)
        cfg = load_config(config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file '{config_path}': {e}") from e

    training_config_key = cfg.get("training_config")
    resolved_training_config_path: Path | None = None
    if training_config_key:
        training_config_path = resolve_path_or_remote(training_config_key, base_dir=project_dir)
        training_config_path = Path(training_config_path)
        if training_config_path.exists():
            logger.info("Loading training configuration file: %s", training_config_path)
            with training_config_path.open("r", encoding="utf-8") as f:
                training_cfg = yaml.safe_load(f) or {}
            cfg.update(training_cfg)
            resolved_training_config_path = training_config_path
        else:
            logger.warning(
                f"Training config '{training_config_path}' specified in 'training_config' not found; "
                "training parameters must be present in config/config.yaml."
            )
    else:
        logger.warning(
            "No 'training_config' key found in main config; "
            "training parameters must be present in config/config.yaml."
        )

    requirements_path = Path(requirements_path).resolve() if requirements_path is not None else project_dir / "requirements.txt"
    check_path_exists(requirements_path, kind="file")

    cfg["PROJECT_ROOT"] = str(project_dir)
    cfg["DIR_OUTPUT"] = str(resolve_path_or_remote(cfg["DIR_OUTPUT"], base_dir=project_dir))
    cfg["ACTIVE_REQUIREMENTS_PATH"] = str(requirements_path)

    try:
        _enforce_python_version(cfg)
        _validate_local_input_artifacts(cfg, project_dir=project_dir)
        reproducibility_cfg = cfg.get("REPRODUCIBILITY", {})
        enforce_pins = bool(reproducibility_cfg.get("enforce_pinned_requirements", True))
        logger.info("Processing requirements file: %s", requirements_path)
        requirements_utils(requirements_path, enforce_pins=enforce_pins)
    except Exception as e:
        raise RuntimeError(f"Prerequisite validation failed: {e}") from e

    if resume_from_dir is not None:
        _resume_path = Path(resume_from_dir).resolve()
        if not _resume_path.exists():
            raise FileNotFoundError(f"resume_from_dir does not exist: {_resume_path}")
        cfg["DIR_MODEL"] = str(_resume_path)
        cfg["RUN_ID"]    = _resume_path.parent.name
        run_root         = _resume_path.parent
        print(f"Resuming into existing model directory: {_resume_path} (run_id: {cfg['RUN_ID']})")
    else:
        cfg = apply_timestamped_run_folder(cfg, output_root=cfg["DIR_OUTPUT"])
        run_root = Path(cfg["DIR_MODEL"]).parent.resolve()
        print(f"Created output folder at: {run_root} (run_id: {cfg['RUN_ID']})")
    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    artifact_toggles = reproducibility_cfg.get("artifacts", {}) or {}
    reproducibility_dir_raw = cfg.get("DIR_REPRODUCIBILITY")
    reproducibility_dir = Path(reproducibility_dir_raw).resolve() if reproducibility_dir_raw else None
    if reproducibility_dir is not None:
        reproducibility_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output folder at: %s (run_id: %s)", run_root, cfg["RUN_ID"])

    configured_log_path = cfg.get("LOG_PATH")
    if log_path is not None:
        log_path = Path(log_path).resolve()
    elif configured_log_path is not None and str(configured_log_path).strip():
        configured_text = str(configured_log_path).strip()
        if _is_remote_location(configured_text):
            raise RuntimeError("LOG_PATH must be a local file path, not a remote URL.")
        configured_candidate = Path(configured_text)
        log_path = configured_candidate if configured_candidate.is_absolute() else run_root / configured_candidate
    else:
        log_path = run_root / "log.txt"

    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=resume_from_dir is not None)

    effective_log_level = cfg.get("LOG_LEVEL", log_level)
    try:
        logging_config(log_path, log_level=effective_log_level)
    except Exception as e:
        raise RuntimeError(f"Failed to configure logging using '{log_path}': {e}") from e

    run_manifest_toggle = bool(artifact_toggles.get("run_manifest_json", True))
    if write_manifest is None:
        write_manifest = run_manifest_toggle
    else:
        write_manifest = bool(write_manifest) and run_manifest_toggle

    if write_manifest and reproducibility_dir is not None:
        try:
            manifest_path = reproducibility_dir / "run_manifest.json"

            write_reproducibility_manifest(
                config_path=config_path,
                requirements_path=requirements_path,
                manifest_path=manifest_path,
                training_config_path=resolved_training_config_path,
                extra={
                    "llm": cfg.get("llm"),
                    "llm_revision": cfg.get("llm_revision"),
                    "hf_local_files_only": cfg.get("hf_local_files_only", False),
                    "run_id": cfg.get("RUN_ID"),
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to write reproducibility manifest: {e}") from e

    if bool(artifact_toggles.get("run_config_yaml", True)) and reproducibility_dir is not None:
        try:
            config_snapshot_path = reproducibility_dir / "run_config.yaml"
            config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            config_snapshot_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to write run config snapshot: {e}") from e

    logger.info("All prerequisites loaded successfully.")
    return cfg
