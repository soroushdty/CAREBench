import pandas as pd
import numpy as np
from ..shared.soft_label_utils import (
    masked_class_data,
    _soft_scores,
    _soft_average_precision,
    _soft_roc_auc,
    _soft_pr_curve,
    soft_ece_with_ci,
    soft_brier_with_ci,
)


def compute_metrics_df(probs, y_true, thresholds, class_list, cfg, patient_ids):
    import logging
    logger = logging.getLogger(__name__)
    """Compute per-class and aggregate classification metrics and return a DataFrame.

    Args:
        probs: Float array of shape ``(n_samples, n_classes)`` — calibrated
            model probabilities.
        y_true: Array-like of shape ``(n_samples, n_classes)`` — ground-truth
            labels (may be soft/float).
        thresholds: Float array of length ``n_classes`` — per-class decision
            thresholds (e.g. from :func:`~tracks.representation.training.shared.threshold_tuning.threshold_tuning`).
        class_list: Sequence of class name strings, length ``n_classes``.
        cfg: Config mapping forwarded to
            :func:`~tracks.representation.training.shared.soft_label_utils.masked_class_data`.
        patient_ids: 1-D array of patient identifiers aligned with the rows of
            ``probs`` / ``y_true``.  Required for cluster-bootstrap CIs.

    Returns:
        pd.DataFrame: One row per class plus two summary rows
            (``"Macro Average"`` and ``"Micro Aggregate"``).
            Columns: ``Class``, ``Precision``, ``Recall``, ``MCC``,
            ``AUC ROC``, ``AUC PR``, ``AP``, ``F1``, ``TP``, ``FP``, ``TN``,
            ``FN``, ``Threshold``, ``Valid Count``, ``Prevalence``,
            ``Pred Pos Rate``.  Metric columns are rounded to 4 decimal places;
            count columns (``TP``, ``FP``, ``TN``, ``FN``, ``Valid Count``) are
            cast to ``int``.
    """
    _n_bins      = int(cfg.get("ece_n_bins",      15))   if isinstance(cfg, dict) else 15
    _n_resamples = int(cfg.get("ece_n_resamples", 1000)) if isinstance(cfg, dict) else 1000

    metrics_rows = []
    micro_tp = micro_fp = micro_tn = micro_fn = 0
    micro_valid = 0

    for i, cls in enumerate(class_list):
        t = thresholds[i]
        mask, y_c, probs_c = masked_class_data(y_true, probs, i, cfg)
        patient_ids_c = patient_ids[mask]

        max_prob = float(np.max(probs_c)) if probs_c.size > 0 else float('nan')
        prevalence = float(np.mean(y_c)) if y_c.size > 0 else float('nan')
        pred_pos_rate = float(np.mean(probs_c >= t)) if probs_c.size > 0 else float('nan')

        # Log per-class threshold and stats
        logger.info(f"[THRESH] Class '{cls}': threshold={t:.4f}, max_prob={max_prob:.4f}, prevalence={prevalence:.4f}, pred_pos_rate={pred_pos_rate:.4f}")

        # Sanity checks and warnings
        if max_prob < t:
            logger.warning(f"[THRESH] Class '{cls}': max predicted probability ({max_prob:.4f}) is below threshold ({t:.4f}) — all predictions will be negative.")
        if pred_pos_rate == 0.0:
            logger.warning(f"[THRESH] Class '{cls}': all predictions are negative at threshold {t:.4f}.")
        if pred_pos_rate > 0.5:
            logger.warning(f"[THRESH] Class '{cls}': overprediction — pred_pos_rate={pred_pos_rate:.4f} at threshold {t:.4f} (prevalence={prevalence:.4f}).")
        if t > 0.8 and prevalence < 0.1:
            logger.warning(f"[THRESH] Class '{cls}': threshold {t:.4f} is high relative to prevalence ({prevalence:.4f}).")

        if y_c.size == 0:
            tn = fp = fn = tp = 0.0
            prec = rec = mcc = f1 = np.nan
            ap = auc_roc = auc_pr = np.nan
            ece_point = ece_ci_lower = ece_ci_upper = np.nan
            brier_point = brier_ci_lower = brier_ci_upper = np.nan
        else:
            soft_scores = _soft_scores(y_c, probs_c, t)
            tp = soft_scores["TP"]
            fp = soft_scores["FP"]
            tn = soft_scores["TN"]
            fn = soft_scores["FN"]
            prevalence = float(np.mean(y_c))
            pred_pos_rate = float(np.mean(probs_c >= t))
            micro_tp += tp
            micro_fp += fp
            micro_tn += tn
            micro_fn += fn
            micro_valid += int(y_c.size)

            prec = soft_scores["Precision"]
            rec = soft_scores["Recall"]
            mcc = soft_scores["MCC"]
            f1 = soft_scores["F1"]
            ap = _soft_average_precision(y_c, probs_c)
            auc_roc = _soft_roc_auc(y_c, probs_c)
            p_curve, r_curve = _soft_pr_curve(y_c, probs_c)
            auc_pr = float(np.trapezoid(p_curve, r_curve)) if r_curve.size > 1 else 0.0
            ece_point, ece_ci_lower, ece_ci_upper = soft_ece_with_ci(
                y_c, probs_c, patient_ids_c,
                n_bins=_n_bins,
                n_resamples=_n_resamples,
            )
            brier_point, brier_ci_lower, brier_ci_upper = soft_brier_with_ci(
                y_c, probs_c, patient_ids_c,
                n_resamples=_n_resamples,
            )

        metrics_rows.append({
            "Class": cls,
            "Precision": prec, "Recall": rec, "MCC": mcc,
            "AUC ROC": auc_roc, "AUC PR": auc_pr, "AP": ap, "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "Threshold": t,
            "Valid Count": int(y_c.size),
            "Prevalence": prevalence,
            "Pred Pos Rate": pred_pos_rate,
            "ECE": ece_point,
            "ECE CI Lower": ece_ci_lower,
            "ECE CI Upper": ece_ci_upper,
            "Brier Score": brier_point,
            "Brier CI Lower": brier_ci_lower,
            "Brier CI Upper": brier_ci_upper,
        })

    df = pd.DataFrame(metrics_rows)

    macro_metric_cols = [
        "Precision", "Recall", "MCC", "AUC ROC", "AUC PR", "AP", "F1",
        "ECE", "ECE CI Lower", "ECE CI Upper",
        "Brier Score", "Brier CI Lower", "Brier CI Upper",
        "Valid Count", "Prevalence", "Pred Pos Rate",
    ]
    means = df[macro_metric_cols].mean()
    macro_row = means.to_dict()
    macro_row['Class'] = 'Macro Average'
    macro_row["TP"] = np.nan
    macro_row["FP"] = np.nan
    macro_row["TN"] = np.nan
    macro_row["FN"] = np.nan
    macro_row["Threshold"] = np.nan

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_denom_mcc = float(np.sqrt(max(
        (micro_tp + micro_fp) * (micro_tp + micro_fn) *
        (micro_tn + micro_fp) * (micro_tn + micro_fn), 1e-8
    )))
    micro_mcc = float((micro_tp * micro_tn - micro_fp * micro_fn) / micro_denom_mcc)
    micro_f1 = (
        0.0
        if (micro_precision + micro_recall) == 0
        else (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall)
    )
    micro_prevalence = (micro_tp + micro_fn) / max(micro_valid, 1)
    micro_pred_pos_rate = (micro_tp + micro_fp) / max(micro_valid, 1)

    micro_row = {
        "Class": "Micro Aggregate",
        "Precision": micro_precision,
        "Recall": micro_recall,
        "MCC": micro_mcc,
        "AUC ROC": np.nan,
        "AUC PR": np.nan,
        "AP": np.nan,
        "F1": micro_f1,
        "TP": int(micro_tp),
        "FP": int(micro_fp),
        "TN": int(micro_tn),
        "FN": int(micro_fn),
        "Threshold": np.nan,
        "Valid Count": int(micro_valid),
        "Prevalence": float(micro_prevalence),
        "Pred Pos Rate": float(micro_pred_pos_rate),
        "ECE": np.nan,
        "ECE CI Lower": np.nan,
        "ECE CI Upper": np.nan,
        "Brier Score": np.nan,
        "Brier CI Lower": np.nan,
        "Brier CI Upper": np.nan,
    }

    df = pd.concat([df, pd.DataFrame([macro_row, micro_row])], ignore_index=True)

    # Keep confusion-matrix count columns as integers, and round metric-like values to 4 decimals.
    count_cols = ["TP", "TN", "FP", "FN"]
    for col in count_cols + ["Valid Count"]:
        if col in df.columns:
            if col in count_cols:
                row_mask = df["Class"] != "Macro Average"
            else:
                row_mask = pd.Series(True, index=df.index)
            df.loc[row_mask, col] = pd.to_numeric(df.loc[row_mask, col], errors="coerce").fillna(0).round(0).astype(int)

    decimal_cols = [
        col for col in df.columns
        if col not in ["Class", *count_cols, "Valid Count"]
    ]
    for col in decimal_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    return df


def write_metrics_csv(df, path) -> None:
    """Write a metrics DataFrame to a CSV file, creating parent directories if needed.

    The file is always overwritten (not appended). Parent directories are
    created via :func:`~shared.utils.file_utils.ensure_parent_dir`.

    Args:
        df: ``pd.DataFrame`` to write (typically the output of
            :func:`compute_metrics_df`).
        path: Destination file path for the CSV.
    """
    from shared.utils.file_utils import write_csv_file
    write_csv_file(df, path, index=False)
