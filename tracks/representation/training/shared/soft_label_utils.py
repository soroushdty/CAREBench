# Soft-label metric utilities
#
# sklearn.metrics functions (roc_auc_score, average_precision_score, etc.)
# reject non-binary y_true arrays, so they cannot be used directly here.
# Labels in this pipeline are continuous values in [0, 1] — physician agreement
# fractions — not one-hot integers.  The implementations below accept these
# soft labels natively by treating them as fractional positives.
#
# Tie-handling convention for _soft_roc_auc and _soft_pr_curve:
#   Samples with identical predicted probabilities are grouped.  Within each
#   tie-group the positives are credited with half the negatives in the group
#   (standard trapezoidal / Wilcoxon-Mann-Whitney convention), matching the
#   behaviour of sklearn when hard labels happen to be present.
import numpy as np


def macro_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """GPU-accelerated macro-averaged Brier Score.

    Used by the inner HP-selection loop; called hundreds of times per pipeline
    run, so GPU throughput matters even though each individual call is small.
    Falls back to CPU if CUDA is unavailable.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(np.asarray(y_true, dtype=np.float32), device=device)
    p = torch.tensor(np.asarray(probs,  dtype=np.float32), device=device)
    if y.numel() == 0:
        return float("nan")
    per_class = ((y - p) ** 2).mean(dim=0)
    return float(per_class.mean().item())


def masked_class_data(y_true, probs, class_idx, eval_pos_threshold_or_cfg):
    y_arr = np.asarray(y_true, dtype=np.float32)
    p_arr = np.asarray(probs, dtype=np.float32)
    if isinstance(eval_pos_threshold_or_cfg, dict):
        threshold = float(eval_pos_threshold_or_cfg.get("eval_pos_threshold", 0.5))
    else:
        threshold = float(eval_pos_threshold_or_cfg)
    y_col = y_arr[:, class_idx]
    # Keep samples that are clearly positive (>= threshold) or clearly negative
    # (<= 1 - threshold). Samples in the ambiguous middle band are excluded.
    # threshold=0.5 (the default) includes all samples: y>=0.5 | y<=0.5 is always True.
    mask = (y_col >= threshold) | (y_col <= 1.0 - threshold)
    return mask, y_col[mask], p_arr[mask, class_idx]


def _soft_confusion_counts(y_true_col, probs_col, threshold):
    y_col = np.asarray(y_true_col, dtype=np.float32)
    p_col = np.asarray(probs_col, dtype=np.float32)
    y_pred = (p_col >= float(threshold)).astype(np.float32)

    tp = float(np.sum(y_col * y_pred))
    fp = float(np.sum((1.0 - y_col) * y_pred))
    fn = float(np.sum(y_col * (1.0 - y_pred)))
    tn = float(np.sum((1.0 - y_col) * (1.0 - y_pred)))
    return tp, fp, tn, fn


def _soft_scores(y_true_col, probs_col, threshold):
    tp, fp, tn, fn = _soft_confusion_counts(y_true_col, probs_col, threshold)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    denom_mcc = float(np.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-8)
    ))
    mcc = float((tp * tn - fp * fn) / denom_mcc)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": float(precision),
        "Recall": float(recall),
        "MCC": mcc,
        "F1": float(f1),
    }


def _soft_pr_curve(y_true_col, probs_col):
    y_col = np.asarray(y_true_col, dtype=np.float32)
    p_col = np.asarray(probs_col, dtype=np.float32)
    if y_col.size == 0:
        return np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    order = np.argsort(p_col, kind="mergesort")[::-1]
    y_sorted = y_col[order]
    p_sorted = p_col[order]
    total_pos = float(y_sorted.sum())
    if total_pos <= 0.0:
        return np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    distinct = np.r_[np.where(np.diff(p_sorted))[0], y_sorted.size - 1]
    tp_cum = np.cumsum(y_sorted)[distinct]
    fp_cum = np.cumsum(1.0 - y_sorted)[distinct]
    precision = np.divide(tp_cum, tp_cum + fp_cum, out=np.ones_like(tp_cum), where=(tp_cum + fp_cum) > 0)
    recall = np.divide(tp_cum, total_pos, out=np.zeros_like(tp_cum), where=total_pos > 0)
    precision = np.concatenate(([1.0], precision.astype(np.float32)))
    recall = np.concatenate(([0.0], recall.astype(np.float32)))
    return precision, recall


def _soft_average_precision(y_true_col, probs_col):
    precision, recall = _soft_pr_curve(y_true_col, probs_col)
    if recall.size < 2:
        return 0.0
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _soft_roc_auc(y_true_col, probs_col):
    y_col = np.asarray(y_true_col, dtype=np.float32)
    p_col = np.asarray(probs_col, dtype=np.float32)
    pos_mass = float(y_col.sum())
    neg_mass = float(np.sum(1.0 - y_col))
    if pos_mass <= 0.0 or neg_mass <= 0.0:
        return 0.5

    order = np.argsort(p_col, kind="mergesort")
    y_sorted = y_col[order]
    p_sorted = p_col[order]
    neg_sorted = 1.0 - y_sorted

    # Wilcoxon-Mann-Whitney U-statistic formulation of AUC.
    # Iterate over tie-groups (samples with identical predicted probability)
    # in ascending order.  For each group, a positive receives credit for:
    #   - every negative that scored strictly lower  (neg_before), plus
    #   - half of every negative in the same tie-group  (0.5 * neg_group).
    # Dividing by pos_mass * neg_mass normalises to [0, 1].
    auc_mass = 0.0
    neg_before = 0.0
    start = 0
    while start < p_sorted.size:
        end = start + 1
        while end < p_sorted.size and p_sorted[end] == p_sorted[start]:
            end += 1
        pos_group = float(y_sorted[start:end].sum())
        neg_group = float(neg_sorted[start:end].sum())
        auc_mass += pos_group * (neg_before + 0.5 * neg_group)
        neg_before += neg_group
        start = end

    return float(auc_mass / (pos_mass * neg_mass))


def _soft_roc_curve(y_true_col, probs_col):
    y_col = np.asarray(y_true_col, dtype=np.float32)
    p_col = np.asarray(probs_col, dtype=np.float32)
    pos_mass = float(y_col.sum())
    neg_mass = float(np.sum(1.0 - y_col))
    if pos_mass <= 0.0 or neg_mass <= 0.0:
        return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)

    order = np.argsort(p_col, kind="mergesort")
    y_sorted = y_col[order]
    p_sorted = p_col[order]
    neg_sorted = 1.0 - y_sorted

    fpr_points = [0.0]
    tpr_points = [0.0]
    neg_before = 0.0
    pos_before = 0.0
    start = 0
    while start < p_sorted.size:
        end = start + 1
        while end < p_sorted.size and p_sorted[end] == p_sorted[start]:
            end += 1
        pos_group = float(y_sorted[start:end].sum())
        neg_group = float(neg_sorted[start:end].sum())
        pos_before += pos_group
        neg_before += neg_group
        tpr_points.append(float(pos_before / pos_mass))
        fpr_points.append(float(neg_before / neg_mass))
        start = end

    return np.asarray(fpr_points, dtype=np.float32), np.asarray(tpr_points, dtype=np.float32)


def masked_macro_f1(y_true, probs, thresholds, eval_pos_threshold_or_cfg):
    y_arr = np.asarray(y_true, dtype=np.float32)
    p_arr = np.asarray(probs, dtype=np.float32)
    t_arr = np.asarray(thresholds, dtype=np.float32)

    scores = []
    for idx in range(y_arr.shape[1]):
        _, y_col, p_col = masked_class_data(y_arr, p_arr, idx, eval_pos_threshold_or_cfg)
        if y_col.size == 0:
            continue

        scores.append(_soft_scores(y_col, p_col, t_arr[idx])["F1"])

    return float(np.mean(scores)) if scores else float("nan")


def masked_macro_metric(y_true, probs, thresholds, eval_pos_threshold_or_cfg, objective):
    """
    Compute a macro-averaged metric using soft labels directly.
    """
    y_arr = np.asarray(y_true, dtype=np.float32)
    p_arr = np.asarray(probs, dtype=np.float32)
    t_arr = np.asarray(thresholds, dtype=np.float32)
    objective = str(objective).strip()
    objective_upper = objective.upper()

    if objective_upper == "F1":
        return masked_macro_f1(y_true, probs, thresholds, eval_pos_threshold_or_cfg)

    scores = []
    for idx in range(y_arr.shape[1]):
        _, y_col, p_col = masked_class_data(y_arr, p_arr, idx, eval_pos_threshold_or_cfg)
        if y_col.size == 0:
            continue

        soft_scores = _soft_scores(y_col, p_col, t_arr[idx])
        if objective_upper == "PRECISION":
            scores.append(soft_scores["Precision"])
        elif objective_upper == "RECALL":
            scores.append(soft_scores["Recall"])
        elif objective_upper == "MCC":
            scores.append(soft_scores["MCC"])
        elif objective_upper == "F0.5":
            prec = soft_scores["Precision"]
            rec = soft_scores["Recall"]
            beta_sq = 0.5 ** 2
            denom = (beta_sq * prec) + rec
            scores.append(0.0 if denom == 0 else ((1 + beta_sq) * prec * rec) / denom)
        elif objective_upper == "AP":
            scores.append(_soft_average_precision(y_col, p_col))
        elif objective_upper == "AUC ROC":
            scores.append(_soft_roc_auc(y_col, p_col))
        elif objective_upper == "AUC PR":
            precision_arr, recall_arr = _soft_pr_curve(y_col, p_col)
            scores.append(float(np.trapezoid(precision_arr, recall_arr)) if recall_arr.size > 1 else 0.0)
        elif objective_upper == "YOUDEN":
            tp = soft_scores["TP"]
            fp = soft_scores["FP"]
            tn = soft_scores["TN"]
            fn = soft_scores["FN"]
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            scores.append(tpr - fpr)

    return float(np.mean(scores)) if scores else float("nan")


def _soft_ece(
    y_true_col: np.ndarray,
    probs_col: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error for a single class column with soft labels.

    Bins predicted probabilities into ``n_bins`` equal-width intervals over
    [0, 1] and computes the weighted absolute gap between mean predicted
    probability and mean ground-truth label within each occupied bin.

    Empty bins contribute 0 (not nan) to ECE.  Returns nan when the input is
    empty or all predicted probabilities are identical (constant predictions).
    """
    y_arr = np.asarray(y_true_col, dtype=np.float32)
    p_arr = np.asarray(probs_col,  dtype=np.float32)
    if y_arr.size == 0:
        return float("nan")
    if y_arr.size > 1 and np.all(p_arr == p_arr[0]):
        return float("nan")

    # n_bins-1 inner edges → values in [0, n_bins-1]
    inner_edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bin_indices = np.digitize(p_arr, inner_edges)
    N = float(y_arr.size)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / N) * abs(float(p_arr[mask].mean()) - float(y_arr[mask].mean()))
    return float(ece)


def soft_ece_with_ci(
    y_true_col: np.ndarray,
    probs_col: np.ndarray,
    patient_ids: np.ndarray,
    n_bins: int = 15,
    n_resamples: int = 1000,
    rng: "np.random.Generator | None" = None,
) -> "tuple[float, float, float]":
    """ECE point estimate with a 95% cluster-bootstrap confidence interval.

    Resamples whole patient blocks (cluster bootstrap) so that the CI
    reflects between-patient variability rather than within-patient
    observation noise.

    Returns ``(ece_point, ci_lower, ci_upper)``.  All three values are nan
    when the array is empty or ECE is undefined.  When fewer than 2 unique
    patients are present the point estimate is returned but CI values are nan.
    """
    import logging

    y_arr = np.asarray(y_true_col, dtype=np.float32)
    p_arr = np.asarray(probs_col,  dtype=np.float32)
    nan3 = (float("nan"), float("nan"), float("nan"))

    if y_arr.size == 0:
        return nan3

    if patient_ids is None:
        raise ValueError("patient_ids is required for cluster-bootstrap CI")

    ece_point = _soft_ece(y_arr, p_arr, n_bins)
    if np.isnan(ece_point):
        return nan3

    unique_pts = np.unique(patient_ids)
    if len(unique_pts) < 2:
        logging.warning(
            "ECE bootstrap CI skipped: fewer than 2 unique patients. "
            "Returning point estimate only."
        )
        return (ece_point, float("nan"), float("nan"))

    if rng is None:
        rng = np.random.default_rng()

    pt_to_ix = {pt: np.where(patient_ids == pt)[0] for pt in unique_pts}

    boot_eces = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sampled = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        idx = np.concatenate([pt_to_ix[pt] for pt in sampled])
        b = _soft_ece(y_arr[idx], p_arr[idx], n_bins)
        boot_eces[i] = b if not np.isnan(b) else ece_point

    return (
        ece_point,
        float(np.percentile(boot_eces, 2.5)),
        float(np.percentile(boot_eces, 97.5)),
    )


def _soft_brier_score(
    y_true_col: np.ndarray,
    probs_col: np.ndarray,
) -> float:
    """Brier Score (mean squared error) for a single class column with soft labels.

    Returns nan if the array is empty.  No binning required — this is simply
    ``mean((y - p)^2)``.
    """
    y_arr = np.asarray(y_true_col, dtype=np.float32)
    p_arr = np.asarray(probs_col,  dtype=np.float32)
    if y_arr.size == 0:
        return float("nan")
    return float(np.mean((y_arr - p_arr) ** 2))


def soft_brier_with_ci(
    y_true_col: np.ndarray,
    probs_col: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: "np.random.Generator | None" = None,
) -> "tuple[float, float, float]":
    """Brier Score point estimate with a 95% cluster-bootstrap confidence interval.

    Resamples whole patient blocks (cluster bootstrap) so that the CI
    reflects between-patient variability rather than within-patient
    observation noise.

    Returns ``(brier_point, ci_lower, ci_upper)``.  All three values are nan
    when the array is empty.  When fewer than 2 unique patients are present
    the point estimate is returned but CI values are nan.
    """
    import logging

    y_arr = np.asarray(y_true_col, dtype=np.float32)
    p_arr = np.asarray(probs_col,  dtype=np.float32)
    nan3 = (float("nan"), float("nan"), float("nan"))

    if y_arr.size == 0:
        return nan3

    if patient_ids is None:
        raise ValueError("patient_ids is required for cluster-bootstrap CI")

    brier_point = _soft_brier_score(y_arr, p_arr)

    unique_pts = np.unique(patient_ids)
    if len(unique_pts) < 2:
        logging.warning(
            "Brier bootstrap CI skipped: fewer than 2 unique patients. "
            "Returning point estimate only."
        )
        return (brier_point, float("nan"), float("nan"))

    if rng is None:
        rng = np.random.default_rng()

    pt_to_ix = {pt: np.where(patient_ids == pt)[0] for pt in unique_pts}

    boot_briers = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sampled = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        idx = np.concatenate([pt_to_ix[pt] for pt in sampled])
        b = _soft_brier_score(y_arr[idx], p_arr[idx])
        boot_briers[i] = b if not np.isnan(b) else brier_point

    return (
        brier_point,
        float(np.percentile(boot_briers, 2.5)),
        float(np.percentile(boot_briers, 97.5)),
    )
