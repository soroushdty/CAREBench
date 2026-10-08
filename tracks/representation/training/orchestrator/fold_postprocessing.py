"""Fold postprocessing: calibration, threshold tuning, metrics, and checkpoint writes."""

from __future__ import annotations

import logging
import os
import tempfile
import joblib
import pandas as pd

from ..shared.apply_calibrators import apply_calibrators
from ..shared.threshold_tuning import threshold_tuning
from .compute_metrics_and_save import compute_metrics_df

logger = logging.getLogger(__name__)


CHECKPOINT_VERSION = "pdm_fold_v3"

_REQUIRED_V3_KEYS = {
    "checkpoint_version", "fold_idx", "val_ix",
    "model_state", "preprocessor", "calibrators",
    "thresholds", "pos_weights",
    "thresh_inner", "oof_probs", "oof_labels", "best_hp",
}


def validate_fold_checkpoint(ckpt: dict, n_classes: int) -> tuple[bool, str]:
    """Return (is_valid, reason) for a fold checkpoint dict."""
    if not isinstance(ckpt, dict):
        return False, "not a dict"
    if ckpt.get("checkpoint_version") != CHECKPOINT_VERSION:
        return False, f"version mismatch: {ckpt.get('checkpoint_version')!r}"
    missing = _REQUIRED_V3_KEYS - set(ckpt.keys())
    if missing:
        return False, f"missing keys: {missing}"
    oof_probs  = ckpt["oof_probs"]
    oof_labels = ckpt["oof_labels"]
    val_ix     = ckpt["val_ix"]
    if not hasattr(oof_probs, "shape") or oof_probs.shape != (len(val_ix), n_classes):
        return False, f"oof_probs shape {getattr(oof_probs, 'shape', '?')} != ({len(val_ix)}, {n_classes})"
    if not hasattr(oof_labels, "shape") or oof_labels.shape != oof_probs.shape:
        return False, f"oof_labels shape {getattr(oof_labels, 'shape', '?')} != oof_probs shape {oof_probs.shape}"
    return True, "ok"


def load_fold_checkpoint(path: Path, n_classes: int) -> dict | None:
    """Load and validate a fold checkpoint. Returns None if missing or invalid."""
    if not path.exists():
        return None
    try:
        ckpt = joblib.load(path)
    except Exception as exc:
        logging.warning("Checkpoint %s failed to load (%s); will re-train fold.", path, exc)
        return None
    valid, reason = validate_fold_checkpoint(ckpt, n_classes)
    if not valid:
        logging.warning("Checkpoint %s invalid (%s); will re-train fold.", path, reason)
        return None
    return ckpt


def detect_completed_folds(
    checkpoints_dir: Path,
    n_folds: int,
    n_classes: int,
) -> dict[int, dict]:
    """Scan the checkpoints directory for valid completed folds.

    Returns a dict mapping 0-based fold_idx to its loaded checkpoint dict,
    containing only folds whose checkpoint passes validation.
    """
    if not checkpoints_dir.exists():
        return {}
    completed = {}
    for fold_idx in range(n_folds):
        ckpt = load_fold_checkpoint(
            checkpoints_dir / f"fold_{fold_idx + 1}.joblib", n_classes
        )
        if ckpt is not None:
            completed[fold_idx] = ckpt
    return completed


def _fold_already_in_csv(csv_path: Path, fold_number: int) -> bool:
    """Return True if the Fold column in csv_path already contains fold_number."""
    if not csv_path.exists():
        return False
    try:
        return int(fold_number) in pd.read_csv(csv_path, usecols=["Fold"])["Fold"].values
    except Exception:
        return False


def _postprocess_fold(fold_idx, val_ix, raw_val_probs, Y_val, class_list, cfg, *, calibrators=None, best_hp=None, patient_ids_val=None):
    """CPU-bound postprocessing for one outer fold.

    Runs in a background thread while the next fold's inner HP search runs on
    GPU. Contains: calibration fitting, calibration application, threshold
    tuning, and per-fold metrics computation.

    Returns a dict consumed by _apply_fold_result on the main thread.
    """
    if patient_ids_val is None:
        raise ValueError("patient_ids_val required for bootstrap CI")

    # Use provided calibrators (from inner OOF). When inner OOF is unavailable
    # (HP loaded from crash-recovery CSV), fall back to identity per class rather
    # than fitting on outer validation labels — that would leak held-out patient
    # data into calibration and violate the Stage 1 spec.
    if calibrators is not None:
        calibs = calibrators
    else:
        calibs = {cls: None for cls in class_list}
    val_probs_cal = apply_calibrators(calibs, raw_val_probs, class_list)
    thresh, thresh_report = threshold_tuning(
        val_probs_cal, Y_val, class_list, cfg, return_report=True
    )
    thresh_report.insert(0, "Fold", int(fold_idx + 1))

    fold_df = compute_metrics_df(val_probs_cal, Y_val, thresh, class_list, cfg, patient_ids_val)
    fold_df.insert(0, "Fold", int(fold_idx + 1))

    return {
        "fold_idx":      fold_idx,
        "val_ix":        val_ix,
        "val_probs_cal": val_probs_cal,
        "calibs":        calibs,
        "thresh":        thresh,
        "thresh_report": thresh_report,
        "fold_df":       fold_df,
        "Y_val":         Y_val,
        "best_hp":       best_hp,
    }


def write_fold_artifacts(result: dict, ensemble_artifacts: dict, paths) -> None:
    """Write all I/O for a completed fold result.

    Separated from accumulator mutations so reads (compute) and writes (I/O)
    are in distinct call sites. Called on the main thread after accumulators
    are updated.
    """
    _fold_num = int(result["fold_idx"])
    logger.debug("write_fold_artifacts: writing fold %d artifacts.", _fold_num + 1)

    # CSVs are opened in append mode so each fold extends the same file
    # incrementally rather than requiring all folds to be held in memory first.
    # The header is written only when the file does not yet exist (fold 0).
    # Thread-safety: this function is always called from the main thread via
    # _apply_fold_result, so concurrent header writes cannot occur.
    if not _fold_already_in_csv(paths.cv_folds_csv, _fold_num + 1):
        _write_header = not paths.cv_folds_csv.exists()
        result["fold_df"].to_csv(paths.cv_folds_csv, mode="a", header=_write_header, index=False)
        logger.debug("write_fold_artifacts: fold %d metrics appended to %s.", _fold_num + 1, paths.cv_folds_csv)
    else:
        logger.debug("write_fold_artifacts: fold %d already in %s — skipping duplicate write.", _fold_num + 1, paths.cv_folds_csv)

    if not _fold_already_in_csv(paths.threshold_report_csv, _fold_num + 1):
        _write_header = not paths.threshold_report_csv.exists()
        result["thresh_report"].to_csv(
            paths.threshold_report_csv, mode="a", header=_write_header, index=False
        )
        logger.debug("write_fold_artifacts: fold %d threshold report appended to %s.", _fold_num + 1, paths.threshold_report_csv)
    else:
        logger.debug("write_fold_artifacts: fold %d already in %s — skipping duplicate write.", _fold_num + 1, paths.threshold_report_csv)

    _stage2_list = ensemble_artifacts.get("stage2", [])
    _fold_stage2 = _stage2_list[_fold_num] if _fold_num < len(_stage2_list) else None
    # ycf_test_probs: calibrated Stage 1 context-free predictions on test items.
    # Populated only when Stage 2 fusion is active; None otherwise.
    _ycf_list = ensemble_artifacts.get("ycf_test_probs", [])
    _ycf = _ycf_list[_fold_num] if _fold_num < len(_ycf_list) else None
    _checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "fold_idx":       _fold_num,
        "val_ix":         result["val_ix"],
        "model_state":    ensemble_artifacts["models"][_fold_num],
        "preprocessor":   ensemble_artifacts["preps"][_fold_num],
        "calibrators":    result["calibs"],
        "thresholds":     result["thresh"],
        "pos_weights":    ensemble_artifacts["pos_weights"][_fold_num],
        "thresh_inner":   ensemble_artifacts["thresh_inner"][_fold_num],
        "stage2":         _fold_stage2,
        "oof_probs":      result["val_probs_cal"],
        "oof_labels":     result["Y_val"],
        "best_hp":        result.get("best_hp"),
        "ycf_test_probs": _ycf,
    }
    target = paths.fold_checkpoint(_fold_num)
    _fd, _tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(_fd)
    try:
        joblib.dump(_checkpoint, _tmp)
        os.replace(_tmp, target)
    except Exception:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise


def _apply_fold_result(result, ensemble_artifacts, oof_preds, *, paths, use_inner_calibrators=False):
    """Apply a completed _postprocess_fold result to shared accumulators.

    Always called on the main thread to avoid concurrent list mutation.
    Accumulator mutations happen first; artifact writes are delegated to
    write_fold_artifacts so the two concerns stay separate.
    """
    oof_preds[result["val_ix"]] = result["val_probs_cal"]
    # Always use calibrators from inner OOF fit if available
    if (
        "calibrators_inner" in ensemble_artifacts
        and ensemble_artifacts["calibrators_inner"][result["fold_idx"]] is not None
    ):
        ensemble_artifacts["calibs"].append(ensemble_artifacts["calibrators_inner"][result["fold_idx"]])
    else:
        ensemble_artifacts["calibs"].append(result["calibs"])
    ensemble_artifacts["thresh"].append(result["thresh"])

    write_fold_artifacts(result, ensemble_artifacts, paths)
