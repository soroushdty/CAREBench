"""High-level summary orchestration wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..logic.audits import (
    write_missingness_matrix,
    write_physician_disagreement,
    write_split_overlap,
    write_stage_audit,
    write_standardization_impact,
)
from ..utils.cleanup import remove_obsolete_outputs
from ..logic.delta import write_delta_outputs
from ..logic.divisions import collect_stage_stats, prepare_stage_data
from ..outputs.final_outputs import write_final_outputs
from ..utils.io_utils import deep_merge, warn
from ..utils.logging_utils import log_high_level_event, setup_summary_logger
from ..outputs.notes_outputs import write_runtime_summary_config, write_summary_index
from ..utils.settings_loader import load_default_summary
from ..outputs.stage_outputs import write_item_frequency_outputs, write_stage_split_outputs
from ..logic.divisions import FIXED_STAGES


def summary(
    cfg: dict[str, Any],
    raw_splits: dict[str, Any],
    standardized_splits: dict[str, Any],
    post_physician_splits: dict[str, Any],
    final_splits: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Generate stage-aware summary artifacts from preprocessing snapshots."""
    summary_cfg = deep_merge(load_default_summary(), cfg.get("SUMMARY") or {})

    strict = bool(summary_cfg.get("strict", False))
    summary_dir = Path(cfg.get("DIR_SUMMARY", "output/summary"))
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Initialize summary logger
    summary_logger = setup_summary_logger(
        summary_dir=summary_dir,
        summary_log_path=cfg.get("SUMMARY_LOG_PATH"),
    )
    root_logger = logging.getLogger()

    result: dict[str, Any] = {
        "summary_dir": str(summary_dir),
        "generated_files": [],
        "generated_tables": [],
        "generated_plots": [],
        "removed_files": [],
        "warnings": [],
        "skipped": [],
        "stage_stats": [],
    }

    # Log high-level start event
    log_high_level_event(
        summary_logger, root_logger, logging.INFO,
        "Starting overall summary generation process..."
    )

    try:
        item_col = str(metadata.get("item_col") or cfg.get("item_col") or "Item")  # PAIRED-CONTEXT SCHEMA
        patient_col = metadata.get("patient_col")
        physician_col = metadata.get("physician_col")

        plot_cfg = summary_cfg["plots"]
        dpi = int(plot_cfg.get("dpi", 150))
        figsize_raw = plot_cfg.get("figsize", [10, 6])
        if (
            isinstance(figsize_raw, (list, tuple))
            and len(figsize_raw) == 2
            and all(isinstance(v, (int, float)) for v in figsize_raw)
        ):
            figsize = (float(figsize_raw[0]), float(figsize_raw[1]))
        else:
            figsize = (10.0, 6.0)
            warn(
                result,
                strict,
                "Invalid SUMMARY.plots.figsize provided; expected two numeric values. Falling back to [10, 6].",
            )
        bar_top_n = int(plot_cfg.get("bar_top_n", 50))
        text_length_bins = int(plot_cfg.get("text_length_bins", 20))
        heatmap_max_labels = int(plot_cfg.get("heatmap_max_labels", 100))

        stage_data, selected_stages, split_names = prepare_stage_data(
            raw_splits=raw_splits,
            standardized_splits=standardized_splits,
            post_physician_splits=post_physician_splits,
            final_splits=final_splits,
            metadata=metadata,
        )

        summary_logger.debug(f"Selected stages: {', '.join(FIXED_STAGES)}")

        stage_stats = collect_stage_stats(
            stage_data={s: stage_data[s] for s in selected_stages},
            item_col=item_col,
            patient_col=patient_col,
            metadata=metadata,
        )
        result["stage_stats"] = stage_stats

        write_stage_audit(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            stage_stats=stage_stats,
        )
        summary_logger.debug("Finished stage audit outputs")

        remove_obsolete_outputs(
            result=result,
            strict=strict,
            summary_dir=summary_dir,
            selected_stages=selected_stages,
            stage_data=stage_data,
        )

        write_item_frequency_outputs(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            split_names=split_names,
            stage_data=stage_data,
            item_col=item_col,
        )
        summary_logger.debug("Finished item frequency outputs")

        write_stage_split_outputs(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            stage_data=stage_data,
            selected_stages=selected_stages,
            summary_dir=summary_dir,
            item_col=item_col,
            patient_col=patient_col,
            physician_col=physician_col,
            metadata=metadata,
            figsize=figsize,
            dpi=dpi,
            bar_top_n=bar_top_n,
            heatmap_max_labels=heatmap_max_labels,
            text_length_bins=text_length_bins,
        )
        for stage in selected_stages:
            log_high_level_event(
                summary_logger, root_logger, logging.INFO,
                f"Finished processing stage: {stage}"
            )

        write_final_outputs(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            stage_data=stage_data,
        )
        summary_logger.debug("Finished final outputs")

        write_delta_outputs(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            cfg=cfg,
            stage_data=stage_data,
        )
        summary_logger.debug("Finished delta outputs")

        write_standardization_impact(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            split_names=split_names,
            stage_data=stage_data,
            item_col=item_col,
        )
        summary_logger.debug("Finished standardization impact outputs")

        write_split_overlap(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            selected_stages=selected_stages,
            stage_data=stage_data,
            item_col=item_col,
        )
        summary_logger.debug("Finished split overlap outputs")

        write_physician_disagreement(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            stage_data=stage_data,
            patient_col=patient_col,
            physician_col=physician_col,
            item_col=item_col,
            metadata=metadata,
        )
        summary_logger.debug("Finished physician disagreement outputs")

        write_missingness_matrix(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            stage_data=stage_data,
        )
        summary_logger.debug("Finished missingness matrix outputs")

        write_summary_index(
            result=result,
            strict=strict,
            cfg=cfg,
            summary_dir=summary_dir,
        )
        summary_logger.debug("Finished summary index outputs")

        write_runtime_summary_config(
            result=result,
            strict=strict,
            summary_cfg=summary_cfg,
            summary_dir=summary_dir,
            cfg=cfg,
        )
        summary_logger.debug("Finished runtime summary config outputs")

        log_high_level_event(
            summary_logger, root_logger, logging.INFO,
            "Finished overall summary generation process"
        )

    except Exception as e:
        error_msg = f"Error during summary generation: {type(e).__name__}: {str(e)}"
        log_high_level_event(summary_logger, root_logger, logging.ERROR, error_msg)
        summary_logger.exception("Full traceback:")
        raise

    return result
