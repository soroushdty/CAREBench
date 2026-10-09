"""Manifest and artifact-index emission helpers.

Provides idempotent functions for writing the canonical run artifacts:
- run_manifest.json
- adapter_manifest.json
- artifact_index.json
- resolved_config.yaml

These are called at the end of a track run to capture provenance metadata.

Canonical path: shared/prerequisites/manifests.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_manifest(
    output_dir: str | Path,
    *,
    run_id: str,
    track: str,
    entry_command: str,
    config_path: str | None = None,
    strategy: str | None = None,
    dry_run: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write run_manifest.json to the output directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write the manifest into.
    run_id : str
        Unique run identifier.
    track : str
        Track name (e.g., "representation", "reasoning").
    entry_command : str
        The CLI command that launched this run.
    config_path : str | None
        Path to the config file used.
    strategy : str | None
        Strategy class name (Track 1 only).
    dry_run : bool
        Whether this was a dry-run execution.
    extra : dict | None
        Additional key-value pairs to include.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "python_version": sys.version,
        "platform": platform.platform(),
        "entry_command": entry_command,
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
    }

    if strategy is not None:
        manifest["strategy"] = strategy
    if dry_run:
        manifest["dry_run"] = True
        manifest["dry_run_warning"] = (
            "Synthetic scores — not scientific results"
        )
    if extra:
        manifest.update(extra)

    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def write_adapter_manifest(
    output_dir: str | Path,
    *,
    adapter_name: str,
    adapter_class: str,
    label_space: dict[str, str],
    reference_aggregation_policy: str = "paired_physician_consensus",
    dataset_path: str | None = None,
    patient_summaries_path: str | None = None,
    mapping_file_path: str | None = None,
    row_counts: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write adapter_manifest.json to the output directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write the manifest into.
    adapter_name : str
        Human-readable adapter name.
    adapter_class : str
        Fully qualified adapter class path.
    label_space : dict[str, str]
        Mapping of canonical_key → display_label.
    reference_aggregation_policy : str
        How reference labels are aggregated.
    dataset_path : str | None
        Path to primary dataset file.
    patient_summaries_path : str | None
        Path to patient summaries JSON.
    mapping_file_path : str | None
        Path to item mapping JSON.
    row_counts : dict[str, int] | None
        Dataset size metadata.
    extra : dict | None
        Additional key-value pairs to include.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "adapter_name": adapter_name,
        "adapter_class": adapter_class,
        "label_space": label_space,
        "reference_aggregation_policy": reference_aggregation_policy,
    }

    if dataset_path is not None:
        manifest["dataset_path"] = str(dataset_path)
    if patient_summaries_path is not None:
        manifest["patient_summaries_path"] = str(patient_summaries_path)
    if mapping_file_path is not None:
        manifest["mapping_file_path"] = str(mapping_file_path)
    if row_counts is not None:
        manifest["row_counts"] = row_counts
    if extra:
        manifest.update(extra)

    path = output_dir / "adapter_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def write_artifact_index(
    output_dir: str | Path,
    role_to_path: dict[str, str],
) -> Path:
    """Write artifact_index.json to the output directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write the index into.
    role_to_path : dict[str, str]
        Mapping of canonical artifact roles to relative file paths.

    Returns
    -------
    Path
        Path to the written index file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "artifact_index.json"
    path.write_text(json.dumps(role_to_path, indent=2), encoding="utf-8")
    return path


def write_resolved_config(
    output_dir: str | Path,
    config: dict[str, Any],
) -> Path:
    """Write resolved_config.yaml to the output directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write the config into.
    config : dict[str, Any]
        The fully resolved configuration dictionary.

    Returns
    -------
    Path
        Path to the written config file.
    """
    import yaml

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "resolved_config.yaml"
    path.write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
