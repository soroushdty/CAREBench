from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.preprocessing import FunctionTransformer

from tracks.representation.models.EnsemblePredictor import EnsemblePredictor, load_ensemble_predictor
from tracks.representation.models.MultiLabelModel import MultiLabelModel
from shared.prerequisites.load_config import load_config
from shared.prerequisites.load_prerequisites import load_prerequisites
from shared.prerequisites.reproducibility_manifest import write_reproducibility_manifest
from shared.prerequisites.requirements_utils import requirements_utils
from shared.prerequisites.run_timestamping import apply_timestamped_run_folder


REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_repro_cfg(tmp_path: Path, *, python_version: str | None = None) -> dict:
    py_ver = python_version or platform.python_version()
    return {
        "DIR_DATASET": "data/dataset.xlsx",
        "DIR_CONTEXT": "data/context.json",
        "DIR_JSON_MAP": "data/mapping.json",
        "DIR_OUTPUT": "output",
        "llm": "local/model",
        "llm_revision": "main",
        "hf_local_files_only": True,
        "REPRODUCIBILITY": {
            "enforce_pinned_requirements": True,
            "python_version": py_ver,
            "allow_remote_inputs": False,
            "folder_enabled": True,
            "folder_path": "reproducibility_artifacts",
            "artifacts": {
                "ensemble_manifest_json": True,
                "run_manifest_json": True,
                "run_config_yaml": True,
                "index_map_json": True,
                "summary_config_json": True,
            },
        },
        "LOG_LEVEL": "INFO",
    }


def _write_project_files(tmp_path: Path, cfg: dict, requirements: str = "pip==0.0.0\n") -> tuple[Path, Path]:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "dataset.xlsx").write_text("placeholder", encoding="utf-8")
    (tmp_path / "data" / "context.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data" / "mapping.json").write_text("{}", encoding="utf-8")

    config_path = tmp_path / "config" / "config.yaml"
    requirements_path = tmp_path / "requirements.txt"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    requirements_path.write_text(requirements, encoding="utf-8")
    return config_path, requirements_path


def test_imports_work_with_repo_root_only_on_pythonpath(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [sys.executable, "-c", "import shared, tracks, adapters; print('ok')"]
    completed = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_compute_embeddings_import_is_cwd_independent(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [sys.executable, "-c", "import shared.embeddings.compute_embeddings as c; print(c.__name__)"]
    completed = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_load_config_resolves_relative_paths_against_config_root(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir(parents=True)
    cfg = _base_repro_cfg(project)
    config_path, _ = _write_project_files(project, cfg)

    monkeypatch.chdir(tmp_path)
    loaded = load_config(config_path)

    assert Path(loaded["DIR_OUTPUT"]).is_absolute()
    assert str(Path(loaded["DIR_OUTPUT"]).parent) == str(project)
    assert str(Path(loaded["DIR_DATASET"]).parent) == str(project / "data")


def test_requirements_utils_rejects_unpinned(tmp_path: Path):
    req = tmp_path / "requirements_input.txt"
    req.write_text("numpy\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unpinned requirements"):
        requirements_utils(req)


def test_requirements_utils_raises_for_missing_module(tmp_path: Path):
    req = tmp_path / "requirements_input.txt"
    req.write_text("this-package-does-not-exist-xyz==0.0.1\n", encoding="utf-8")

    with pytest.raises(ModuleNotFoundError, match="Missing required dependencies"):
        requirements_utils(req)


def test_requirements_utils_rejects_version_mismatch(tmp_path: Path):
    req = tmp_path / "requirements_input.txt"
    req.write_text("pytest==0.0.1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not match pinned requirements"):
        requirements_utils(req)


def test_load_prerequisites_enforces_python_version(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path, python_version="999.0.0")
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    with pytest.raises(RuntimeError, match="Python version mismatch"):
        load_prerequisites(
            project_dir=tmp_path,
            config_path=config_path,
            requirements_path=requirements_path,
        )


def test_load_prerequisites_accepts_minimum_python_version_specifier(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path, python_version=">=3.10")
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    loaded = load_prerequisites(
        project_dir=tmp_path,
        config_path=config_path,
        requirements_path=requirements_path,
        write_manifest=False,
    )

    assert loaded["PROJECT_ROOT"] == str(tmp_path)


def test_load_prerequisites_creates_log_file_by_default(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path)
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    loaded = load_prerequisites(
        project_dir=tmp_path,
        config_path=config_path,
        requirements_path=requirements_path,
        write_manifest=False,
    )

    run_root = Path(loaded["DIR_MODEL"]).parent
    assert (run_root / "log.txt").exists()


def test_load_prerequisites_respects_log_path_config_relative_to_run_root(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path)
    cfg["LOG_PATH"] = "logs/pipeline.log"
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    loaded = load_prerequisites(
        project_dir=tmp_path,
        config_path=config_path,
        requirements_path=requirements_path,
        write_manifest=False,
    )

    run_root = Path(loaded["DIR_MODEL"]).parent
    assert (run_root / "logs" / "pipeline.log").exists()
    assert not (run_root / "log.txt").exists()


def test_load_prerequisites_respects_log_path_config_absolute_path(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path)
    absolute_log_path = tmp_path / "external_logs" / "custom.log"
    cfg["LOG_PATH"] = str(absolute_log_path)
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    loaded = load_prerequisites(
        project_dir=tmp_path,
        config_path=config_path,
        requirements_path=requirements_path,
        write_manifest=False,
    )

    run_root = Path(loaded["DIR_MODEL"]).parent
    assert absolute_log_path.exists()
    assert not (run_root / "log.txt").exists()


def test_load_prerequisites_fails_fast_without_side_effect_dirs(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path, python_version="0.0.0")
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="pip==0.0.1\n")

    with pytest.raises(RuntimeError):
        load_prerequisites(
            project_dir=tmp_path,
            config_path=config_path,
            requirements_path=requirements_path,
        )

    assert not (tmp_path / "output").exists()


def test_run_folder_creation_is_collision_safe(tmp_path: Path):
    cfg = {"DIR_OUTPUT": str(tmp_path / "output")}
    cfg_1 = apply_timestamped_run_folder(dict(cfg), output_root=cfg["DIR_OUTPUT"])
    cfg_2 = apply_timestamped_run_folder(dict(cfg), output_root=cfg["DIR_OUTPUT"])

    assert cfg_1["RUN_ID"] != cfg_2["RUN_ID"]
    assert Path(cfg_1["DIR_MODEL"]).parent.exists()
    assert Path(cfg_2["DIR_MODEL"]).parent.exists()
    assert Path(cfg_1["DIR_REPRODUCIBILITY"]).exists()
    assert Path(cfg_2["DIR_REPRODUCIBILITY"]).exists()


def test_run_folder_uses_default_summary_subdir(tmp_path: Path):
    cfg = {"DIR_OUTPUT": str(tmp_path / "output")}
    updated = apply_timestamped_run_folder(dict(cfg), output_root=cfg["DIR_OUTPUT"])

    run_root = Path(updated["DIR_MODEL"]).parent
    assert Path(updated["DIR_SUMMARY"]) == run_root / "input_summary"


def test_run_folder_respects_configured_summary_subdir(tmp_path: Path):
    cfg = {
        "DIR_OUTPUT": str(tmp_path / "output"),
        "DIR_SUMMARY": "custom_summary_dir",
    }
    updated = apply_timestamped_run_folder(dict(cfg), output_root=cfg["DIR_OUTPUT"])

    run_root = Path(updated["DIR_MODEL"]).parent
    assert Path(updated["DIR_SUMMARY"]) == run_root / "custom_summary_dir"


def test_run_folder_can_disable_reproducibility_folder(tmp_path: Path):
    cfg = {
        "DIR_OUTPUT": str(tmp_path / "output"),
        "REPRODUCIBILITY": {
            "folder_enabled": False,
            "python_version": platform.python_version(),
            "enforce_pinned_requirements": True,
            "allow_remote_inputs": False,
        },
    }
    updated = apply_timestamped_run_folder(dict(cfg), output_root=cfg["DIR_OUTPUT"])

    assert "DIR_REPRODUCIBILITY" not in updated


def test_load_prerequisites_writes_manifest_and_snapshot_under_reproducibility(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path, python_version=platform.python_version())
    cfg["REPRODUCIBILITY"]["artifacts"]["run_manifest_json"] = True
    cfg["REPRODUCIBILITY"]["artifacts"]["run_config_yaml"] = True
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    loaded = load_prerequisites(
        project_dir=tmp_path,
        config_path=config_path,
        requirements_path=requirements_path,
    )

    run_root = Path(loaded["DIR_MODEL"]).parent
    reproducibility_dir = run_root / "reproducibility_artifacts"
    assert reproducibility_dir.exists()
    assert (reproducibility_dir / "run_manifest.json").exists()
    assert (reproducibility_dir / "run_config.yaml").exists()


def test_load_prerequisites_respects_run_manifest_toggle(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path, python_version=platform.python_version())
    cfg["REPRODUCIBILITY"]["artifacts"]["run_manifest_json"] = False
    cfg["REPRODUCIBILITY"]["artifacts"]["run_config_yaml"] = True
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="")

    loaded = load_prerequisites(
        project_dir=tmp_path,
        config_path=config_path,
        requirements_path=requirements_path,
    )

    run_root = Path(loaded["DIR_MODEL"]).parent
    reproducibility_dir = run_root / "reproducibility_artifacts"
    assert not (reproducibility_dir / "run_manifest.json").exists()
    assert (reproducibility_dir / "run_config.yaml").exists()


def test_manifest_contains_exact_version_and_hash_fields(tmp_path: Path):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "config" / "config.yaml"
    req_path = tmp_path / "requirements_input.txt"
    manifest = tmp_path / "run_manifest.json"

    cfg_payload = _base_repro_cfg(tmp_path)
    cfg_payload["RUN_ID"] = "run_abc"
    cfg_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding="utf-8")
    req_path.write_text("pip==0.0.1\n", encoding="utf-8")

    out = write_reproducibility_manifest(
        config_path=cfg_path,
        requirements_path=req_path,
        manifest_path=manifest,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_abc"
    assert payload["requirements"]["sha256"]
    assert payload["config"]["sha256"]
    assert "installed" in payload["requirements"]["installed_versions"]["pip"]
    assert "input_artifacts" in payload


def test_remote_inputs_disallowed_by_default(tmp_path: Path):
    cfg = _base_repro_cfg(tmp_path, python_version=platform.python_version())
    cfg["DIR_CONTEXT"] = "https://example.com/context.json"
    config_path, requirements_path = _write_project_files(tmp_path, cfg, requirements="pip==0.0.1\n")

    with pytest.raises(RuntimeError, match="Remote input"):
        load_prerequisites(
            project_dir=tmp_path,
            config_path=config_path,
            requirements_path=requirements_path,
        )


def test_ensemble_bundle_reload_in_fresh_process_without_manual_path_hacks(tmp_path: Path):
    prep = FunctionTransformer(validate=False)
    model = MultiLabelModel(2, 1, hidden_dims=[])

    predictor = EnsemblePredictor(
        models=[model.state_dict()],
        preprocessors=[prep],
        calibrators=[{"class_0": None}],
        thresholds=[np.array([0.5], dtype=np.float32)],
        class_list=["class_0"],
    )

    artifact_path = tmp_path / "ensemble_bundle.joblib"
    predictor.save_bundle(artifact_path, cfg_snapshot={"x": 1})

    loaded = load_ensemble_predictor(artifact_path)
    probs = loaded.predict_proba(np.array([[0.1, 0.9]], dtype=np.float64))
    assert probs.shape == (1, 1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [
        sys.executable,
        "-c",
        (
            "import numpy as np; "
            "from tracks.representation.models.EnsemblePredictor import load_ensemble_predictor; "
            f"m=load_ensemble_predictor(r'{artifact_path}'); "
            "p=m.predict_proba(np.array([[0.2,0.8]],dtype=float)); "
            "print(p.shape)"
        ),
    ]
    completed = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
