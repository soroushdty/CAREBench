import logging

import numpy as np
import pandas as pd

from .soft_label_utils import masked_class_data

logger = logging.getLogger(__name__)


def _resolve_threshold_grid(cfg):
    explicit_grid = cfg.get("threshold_search_grid")
    if explicit_grid is not None:
        grid = np.asarray(explicit_grid, dtype=np.float32)
    else:
        low = float(cfg.get("threshold_search_min", 0.05))
        high = float(cfg.get("threshold_search_max", 0.99))
        points = int(cfg.get("threshold_search_points", 99))
        points = max(points, 2)
        grid = np.linspace(low, high, points, dtype=np.float32)

    min_t = float(cfg.get("threshold_min_value", 0.0))
    max_t = float(cfg.get("threshold_max_value", 1.0))
    grid = np.clip(grid, min_t, max_t)
    grid = np.unique(grid)
    if grid.size == 0:
        grid = np.array([float(np.clip(0.5, min_t, max_t))], dtype=np.float32)
    return grid



def _objective_score(metrics, objective):
    normalized = str(objective).strip().upper()
    if normalized == "F1":
        return float(metrics["F1"])
    if normalized in {"F0.5", "FBETA"}:
        return float(metrics["Fbeta"])
    if normalized == "PRECISION":
        return float(metrics["Precision"])
    if normalized == "RECALL":
        return float(metrics["Recall"])
    if normalized == "YOUDEN":
        return float(metrics["Youden"])
    raise ValueError(
        f"Unsupported threshold objective: {objective}. Supported: F1, F0.5, FBETA, PRECISION, RECALL, YOUDEN"
    )


def _threshold_tuning_gpu(
    probs_c: np.ndarray,
    y_bin: np.ndarray,
    grid: np.ndarray,
    beta: float,
    min_precision: float | None,
    min_recall: float | None,
    max_pred_pos_rate: float | None,
    min_t: float,
    max_t: float,
    objective: str,
) -> tuple[float, dict]:
    """Vectorised threshold search, evaluated in a single GPU broadcast pass.

    Builds ``(N, G)`` tensors where N = samples and G = grid candidates, then
    computes all metrics simultaneously without a Python loop over the grid.

    Constraint masking: grid candidates that violate ``min_precision``,
    ``min_recall``, or ``max_pred_pos_rate`` are set to ``-inf`` before
    ``argmax``. If **all** candidates are masked (i.e. no feasible point
    exists), the constraint is silently relaxed and the unconstrained best is
    returned instead.

    Falls back gracefully to CPU if CUDA is unavailable at runtime.

    Args:
        probs_c: 1-D float array of shape ``(N,)`` — predicted probabilities
            for a single class.
        y_bin: 1-D float array of shape ``(N,)`` — binarised ground-truth
            labels for the same class.
        grid: 1-D float array of shape ``(G,)`` — threshold candidates.
        beta: Beta parameter for the F-beta score.
        min_precision: Minimum acceptable precision; ``None`` to skip.
        min_recall: Minimum acceptable recall; ``None`` to skip.
        max_pred_pos_rate: Maximum acceptable predicted-positive rate; ``None``
            to skip.
        min_t: Lower bound applied to the grid before evaluation.
        max_t: Upper bound applied to the grid before evaluation.
        objective: Optimisation objective — one of ``"F1"``, ``"F0.5"``,
            ``"FBETA"``, ``"PRECISION"``, ``"RECALL"``, ``"YOUDEN"``.

    Returns:
        tuple[float, dict]: ``(best_threshold, metrics)`` where ``metrics``
            contains ``Precision``, ``Recall``, ``F1``, ``Fbeta``, ``Youden``,
            and ``Pred Pos Rate`` at the selected threshold.

    Raises:
        ValueError: If ``objective`` is not a recognised optimisation target.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = torch.tensor(probs_c, dtype=torch.float32, device=device)   # (N,)
    y = torch.tensor(y_bin,   dtype=torch.float32, device=device)   # (N,)
    g = torch.tensor(
        np.clip(grid, min_t, max_t), dtype=torch.float32, device=device
    )                                                                 # (G,)

    # Broadcast p (N,) against g (G,) to produce a boolean prediction matrix.
    # p.unsqueeze(1) → (N, 1); g.unsqueeze(0) → (1, G); result → (N, G).
    # Each column j is the binary prediction vector for threshold g[j],
    # letting us score all G thresholds in a single GPU kernel.
    pred = (p.unsqueeze(1) >= g.unsqueeze(0)).float()                # (N, G)

    # Sum over the N dimension → per-threshold confusion counts, shape (G,).
    tp = (y.unsqueeze(1) * pred).sum(dim=0)                          # (G,)
    fp = ((1 - y).unsqueeze(1) * pred).sum(dim=0)
    fn = (y.unsqueeze(1) * (1 - pred)).sum(dim=0)
    tn = ((1 - y).unsqueeze(1) * (1 - pred)).sum(dim=0)

    precision = torch.where((tp + fp) > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall    = torch.where((tp + fn) > 0, tp / (tp + fn), torch.zeros_like(tp))
    beta_sq   = float(beta) ** 2
    fb_denom  = beta_sq * precision + recall
    fbeta     = torch.where(fb_denom > 0,
                            (1 + beta_sq) * precision * recall / fb_denom,
                            torch.zeros_like(precision))
    f1_denom  = precision + recall
    f1        = torch.where(f1_denom > 0,
                            2 * precision * recall / f1_denom,
                            torch.zeros_like(precision))
    tpr      = recall
    fpr      = torch.where((fp + tn) > 0, fp / (fp + tn), torch.zeros_like(fp))
    youden   = tpr - fpr
    pred_pos = pred.mean(dim=0)

    score_map = {
        "F1":        f1,
        "F0.5":      fbeta,
        "FBETA":     fbeta,
        "PRECISION": precision,
        "RECALL":    recall,
        "YOUDEN":    youden,
    }
    obj_upper = str(objective).strip().upper()
    if obj_upper not in score_map:
        raise ValueError(
            f"Unsupported threshold objective: {objective}. "
            f"Supported: {list(score_map)}"
        )
    scores = score_map[obj_upper].clone()

    # Enforce operational constraints by masking violating grid points.
    # Setting the score to -inf removes them from argmax without changing the
    # scores tensor shape.  All three constraints are hard floors/ceilings —
    # a threshold that violates any one of them is inadmissible regardless of
    # its objective score.
    if min_precision is not None:
        scores[precision < min_precision] = float("-inf")
    if min_recall is not None:
        scores[recall < min_recall] = float("-inf")
    if max_pred_pos_rate is not None:
        scores[pred_pos > max_pred_pos_rate] = float("-inf")

    # If every grid point violates at least one constraint, fall back to the
    # unconstrained best. This means the constraint floor cannot be satisfied
    # on this class/fold; downstream calibration will see a lower-precision
    # threshold than requested.
    if not torch.isfinite(scores).any():
        logger.warning(
            "threshold_tuning: no grid point satisfies all constraints "
            "(min_precision=%s, min_recall=%s, max_pred_pos_rate=%s); "
            "falling back to unconstrained best for objective '%s'.",
            min_precision, min_recall, max_pred_pos_rate, objective,
        )
        scores = score_map[obj_upper].clone()

    best_idx = int(scores.argmax().item())
    best_threshold = float(g[best_idx].item())

    metrics = {
        "Precision":     float(precision[best_idx].item()),
        "Recall":        float(recall[best_idx].item()),
        "F1":            float(f1[best_idx].item()),
        "Fbeta":         float(fbeta[best_idx].item()),
        "Youden":        float(youden[best_idx].item()),
        "Pred Pos Rate": float(pred_pos[best_idx].item()),
    }
    return best_threshold, metrics


def threshold_tuning(probs, Y_val, class_list, cfg, return_report=False):
    """Select per-class decision thresholds by optimising a configurable objective.

    Delegates per-class search to :func:`_threshold_tuning_gpu`. Classes with
    no valid samples after masking receive the default threshold
    ``clip(0.5, min_t, max_t)`` and ``NaN`` metric values in the report.

    Config keys consumed (all optional with shown defaults):

    - ``threshold_objective`` (``"F0.5"``): optimisation target.
    - ``threshold_beta`` (``0.5``): beta for F-beta computation.
    - ``threshold_min_precision_floor`` (``None``): minimum precision constraint.
    - ``threshold_min_recall_floor`` (``None``): minimum recall constraint.
    - ``threshold_max_predicted_positive_rate`` (``None``): max PPR constraint.
    - ``threshold_min_value`` / ``threshold_max_value`` (``0.0`` / ``1.0``):
      bounds applied to the search grid.
    - ``threshold_search_grid``, ``threshold_search_min/max/points``: grid (default min=0.05)
      definition (see :func:`_resolve_threshold_grid`).

    Args:
        probs: Float array of shape ``(n_samples, n_classes)`` — calibrated
            model probabilities from the validation fold.
        Y_val: Array-like of shape ``(n_samples, n_classes)`` — ground-truth
            labels (may be soft/float).
        class_list: Sequence of class name strings, length ``n_classes``.
        cfg: Config mapping (see keys above).
        return_report: If ``True``, also return a per-class tuning report.

    Returns:
        np.ndarray | tuple:
            - If ``return_report=False``: float32 array of shape
              ``(n_classes,)`` — selected threshold per class.
            - If ``return_report=True``: ``(thresholds, report_df)`` where
              ``report_df`` is a ``pd.DataFrame`` with columns ``Class``,
              ``Valid Count``, ``Soft Prevalence``, ``Threshold Objective``,
              ``Threshold Beta``, ``Selected Threshold``,
              ``Selected Objective Score``, ``Precision@Selected``,
              ``Recall@Selected``, ``F1@Selected``, ``Fbeta@Selected``,
              ``Pred Pos Rate@Selected``.
    """
    objective = cfg.get("threshold_objective", "F0.5")
    beta = float(cfg.get("threshold_beta", 0.5))
    min_precision = cfg.get("threshold_min_precision_floor")
    min_recall = cfg.get("threshold_min_recall_floor")
    max_pred_pos_rate = cfg.get("threshold_max_predicted_positive_rate")
    min_t = float(cfg.get("threshold_min_value", 0.0))
    max_t = float(cfg.get("threshold_max_value", 1.0))

    min_precision = None if min_precision is None else float(min_precision)
    min_recall = None if min_recall is None else float(min_recall)
    max_pred_pos_rate = None if max_pred_pos_rate is None else float(max_pred_pos_rate)

    grid = _resolve_threshold_grid(cfg)
    best_thresh = np.full(len(class_list), float(np.clip(0.5, min_t, max_t)), dtype=np.float32)
    report_rows = []

    for i, cls_name in enumerate(class_list):
        _, y_bin, probs_c = masked_class_data(Y_val, probs, i, cfg)
        valid_count = int(y_bin.size)
        prevalence = float(np.mean(y_bin)) if valid_count > 0 else float("nan")

        if valid_count == 0:
            selected_threshold = float(best_thresh[i])
            selected_metrics = {
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
                "Fbeta": float("nan"),
                "Youden": float("nan"),
                "Pred Pos Rate": float("nan"),
            }
            selected_score = float("nan")
        else:
            selected_threshold, selected_metrics = _threshold_tuning_gpu(
                probs_c=probs_c,
                y_bin=y_bin,
                grid=grid,
                beta=beta,
                min_precision=min_precision,
                min_recall=min_recall,
                max_pred_pos_rate=max_pred_pos_rate,
                min_t=min_t,
                max_t=max_t,
                objective=objective,
            )
            best_thresh[i] = float(selected_threshold)
            selected_score = _objective_score(selected_metrics, objective=objective)

        report_rows.append(
            {
                "Class": cls_name,
                "Valid Count": valid_count,
                "Soft Prevalence": prevalence,
                "Threshold Objective": str(objective),
                "Threshold Beta": beta,
                "Selected Threshold": float(best_thresh[i]),
                "Selected Objective Score": float(selected_score),
                "Precision@Selected": selected_metrics["Precision"],
                "Recall@Selected": selected_metrics["Recall"],
                "F1@Selected": selected_metrics["F1"],
                "Fbeta@Selected": selected_metrics["Fbeta"],
                "Pred Pos Rate@Selected": selected_metrics["Pred Pos Rate"],
            }
        )

    if return_report:
        return best_thresh, pd.DataFrame(report_rows)
    return best_thresh
