"""Track 1 representation framework orchestrator.

Provides the `RepresentationTrack` class that composes adapter, strategy,
and artifact writers through dependency injection. Contains NO dataset-specific
logic, column names, or algorithm parameters.

Canonical path: tracks/representation/framework.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .strategies.base import (
    ArtifactWriter,
    RepresentationDataset,
    RepresentationDatasetAdapter,
    TrainingStrategy,
)

logger = logging.getLogger(__name__)


class RepresentationTrack:
    """Orchestrates Track 1: adapter -> strategy -> artifact writers.

    This class contains NO dataset-specific logic, column names, or algorithm
    parameters. All intelligence is delegated to the injected components.

    Parameters
    ----------
    adapter : RepresentationDatasetAdapter
        Loads and constructs the representation dataset.
    strategy : TrainingStrategy
        Trains/evaluates a representation model.
    artifact_writers : list[ArtifactWriter] | None
        Zero or more writers that persist strategy results.
    """

    def __init__(
        self,
        adapter: RepresentationDatasetAdapter,
        strategy: TrainingStrategy,
        artifact_writers: list[ArtifactWriter] | None = None,
    ) -> None:
        self._adapter = adapter
        self._strategy = strategy
        self._artifact_writers = artifact_writers or []

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute the full Track 1 pipeline.

        1. Load dataset via the injected adapter.
        2. Train/evaluate via the injected strategy.
        3. Write artifacts via each injected writer.
        4. Emit run manifests and artifact index.

        Parameters
        ----------
        config : dict[str, Any]
            Pipeline configuration dict.

        Returns
        -------
        dict[str, Any]
            The strategy result dict.
        """
        output_dir = Path(config.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading dataset via adapter...")
        dataset: RepresentationDataset = (
            self._adapter.load_representation_dataset(config)
        )

        logger.info("Running training strategy...")
        result: dict[str, Any] = self._strategy.fit_and_evaluate(dataset, config)

        logger.info("Writing artifacts (%d writers)...", len(self._artifact_writers))
        for writer in self._artifact_writers:
            writer.write(result, output_dir)

        # --- Emit manifests ---
        self._emit_manifests(output_dir, config, result, dataset)

        logger.info("Track 1 pipeline complete.")
        return result

    def _emit_manifests(
        self,
        output_dir: Path,
        config: dict[str, Any],
        result: dict[str, Any],
        dataset: RepresentationDataset,
    ) -> None:
        """Emit run_manifest.json, adapter_manifest.json, artifact_index.json."""
        from shared.prerequisites.manifests import (
            write_adapter_manifest,
            write_artifact_index,
            write_resolved_config,
            write_run_manifest,
        )

        run_id = config.get("RUN_ID", output_dir.name)
        strategy_name = type(self._strategy).__qualname__

        write_run_manifest(
            output_dir,
            run_id=run_id,
            track="representation",
            entry_command="python main.py --track representation",
            config_path=config.get("_config_path"),
            strategy=strategy_name,
            # The strategy's cross-validation splits, when it reports them.
            extra={"cv": result["cv"]} if result.get("cv") else None,
        )

        # Adapter manifest — use adapter.manifest() if available, else minimal
        adapter_name = getattr(self._adapter, "name", "unknown")
        adapter_class = (
            f"{type(self._adapter).__module__}.{type(self._adapter).__qualname__}"
        )
        # The adapter may supply a canonical key → display label mapping;
        # otherwise the output dimensions map to themselves.
        label_space = (dataset.metadata or {}).get("label_space") or {
            name: name for name in dataset.output_dimensions
        }

        write_adapter_manifest(
            output_dir,
            adapter_name=adapter_name,
            adapter_class=adapter_class,
            label_space=label_space,
        )

        # Artifact index — map canonical roles to files that exist
        role_to_path: dict[str, str] = {}
        # Check for common Track 1 output patterns
        for candidate in output_dir.rglob("*"):
            if candidate.is_file():
                rel = str(candidate.relative_to(output_dir))
                name_lower = candidate.name.lower()
                if "context_free" in name_lower and "pred" in name_lower:
                    role_to_path["track1.predictions.context_free"] = rel
                elif "correct_context" in name_lower and "pred" in name_lower:
                    role_to_path["track1.predictions.correct_context"] = rel
                elif "metrics" in name_lower and candidate.suffix in (".csv", ".json"):
                    role_to_path["track1.metrics.per_dimension"] = rel
                elif "ensemble_bundle" in name_lower:
                    role_to_path["track1.bundle"] = rel

        write_artifact_index(output_dir, role_to_path)
        write_resolved_config(output_dir, config)
