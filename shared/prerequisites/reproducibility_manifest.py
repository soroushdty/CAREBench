from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
import yaml

from shared.endpoints import PROTOCOL_VERSION
from shared.utils.path_utils import is_remote_location as _is_remote_location


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit(project_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _infer_project_dir(config_path: Path) -> Path:
    if config_path.parent.name in ("config", "configs"):
        return config_path.parent.parent
    return config_path.parent


def _package_versions(requirements_path: Path) -> dict[str, dict[str, str | None]]:
    versions: dict[str, dict[str, str | None]] = {}
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==", 1)[0].strip()
        expected = line.split("==", 1)[1].strip() if "==" in line else None
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        versions[name] = {"expected": expected, "installed": installed}
    return versions


def _path_metadata(value: str, project_dir: Path) -> dict[str, Any]:
    if _is_remote_location(value):
        return {"source": "remote", "location": value}

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (project_dir / candidate).resolve()

    exists = candidate.exists() and candidate.is_file()
    try:
        rel_location = str(candidate.relative_to(project_dir))
    except ValueError:
        rel_location = str(candidate)
    payload: dict[str, Any] = {
        "source": "local",
        "location": rel_location,
        "exists": exists,
    }
    if exists:
        payload["sha256"] = _sha256_file(candidate)
    return payload


def _input_artifact_metadata(config_data: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    return {
        key: _path_metadata(str(config_data[key]), project_dir)
        for key in ("DIR_DATASET", "DIR_CONTEXT", "DIR_JSON_MAP")
        if key in config_data and config_data.get(key)
    }


def _output_artifact_hashes(output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not output_root.exists():
        return hashes

    for file_path in sorted(output_root.rglob("*")):
        if not file_path.is_file():
            continue
        hashes[str(file_path.relative_to(output_root))] = _sha256_file(file_path)
    return hashes


def write_reproducibility_manifest(
    *,
    config_path: str | Path,
    requirements_path: str | Path,
    manifest_path: str | Path,
    training_config_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    requirements_path = Path(requirements_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    project_dir = _infer_project_dir(config_path)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = yaml.safe_dump(config_data, sort_keys=False, default_flow_style=False)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(extra.get("run_id") or "") if extra and extra.get("run_id") else str(config_data.get("RUN_ID", "")),
        "protocol_version": PROTOCOL_VERSION,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": _git_commit(project_dir),
        "config": {
            "path": str(config_path.relative_to(project_dir)),
            "sha256": _sha256_file(config_path),
            "resolved_hash": _sha256_text(config_text),
        },
        "requirements": {
            "path": str(requirements_path.relative_to(project_dir)),
            "sha256": _sha256_file(requirements_path),
            "installed_versions": _package_versions(requirements_path),
        },
    }

    if training_config_path is not None:
        tc_path = Path(training_config_path).resolve()
        if tc_path.is_file():
            tc_text = yaml.safe_dump(
                yaml.safe_load(tc_path.read_text(encoding="utf-8")),
                sort_keys=False,
                default_flow_style=False,
            )
            try:
                tc_rel = str(tc_path.relative_to(project_dir))
            except ValueError:
                tc_rel = str(tc_path)
            payload["training_config"] = {
                "path": tc_rel,
                "sha256": _sha256_file(tc_path),
                "resolved_hash": _sha256_text(tc_text),
            }

    payload.update({
        "input_artifacts": _input_artifact_metadata(config_data, project_dir),
        "model": {
            "id": config_data.get("llm"),
            "revision": config_data.get("llm_revision"),
            "local_files_only": bool(config_data.get("hf_local_files_only", False)),
        },
    })

    output_dir = config_data.get("DIR_OUTPUT")
    if output_dir:
        payload["output_artifacts"] = _output_artifact_hashes(Path(output_dir))

    if extra:
        payload["extra"] = extra

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
