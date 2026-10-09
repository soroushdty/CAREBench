import numpy as np
import logging

def resolve_thresholds(cfg, class_list, tuned_thresholds=None, logger=logging):
    """
    Resolve per-class thresholds for final metric computation.
    Args:
        cfg: config dict
        class_list: list of class names
        tuned_thresholds: np.ndarray or None, per-class thresholds (from inner-F1 or tuning)
        logger: logger for warnings/info
    Returns:
        np.ndarray of thresholds, shape (n_classes,)
    """
    n_classes = len(class_list)
    mode = cfg.get("primary_threshold", "fixed_f1")
    thresholds = None
    meta = {}

    if mode == "tau":
        tau = float(cfg.get("tau", 0.5))
        thresholds = np.full(n_classes, tau, dtype=np.float32)
        meta["mode"] = "tau"
        meta["value"] = tau
    elif mode == "fixed_f1":
        tau = float(cfg.get("fixed_f1_threshold", 0.3))
        thresholds = np.full(n_classes, tau, dtype=np.float32)
        meta["mode"] = "fixed_f1"
        meta["value"] = tau
    elif mode in ("inner_f1_opt", "tuned"):
        if tuned_thresholds is not None:
            thresholds = np.array(tuned_thresholds, dtype=np.float32)
            meta["mode"] = mode
            meta["value"] = "tuned"
        else:
            logger.warning(f"primary_threshold '{mode}' requested but tuned_thresholds unavailable. Aborting.")
            raise ValueError(f"primary_threshold '{mode}' requested but tuned_thresholds unavailable.")
    else:
        logger.warning(f"Unknown primary_threshold mode '{mode}', defaulting to tau (0.5)")
        thresholds = np.full(n_classes, 0.5, dtype=np.float32)
        meta["mode"] = "tau"
        meta["value"] = 0.5

    return thresholds, meta
