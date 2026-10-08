import logging

import numpy as np

logger = logging.getLogger(__name__)


def apply_calibrators(calibrators, probs, class_list):
    """Apply fitted per-class calibrators to raw model probabilities.

    Column ordering of ``probs`` must match the order of ``class_list``, which
    must also match the order used when the calibrators were fitted with
    :func:`~tracks.representation.training.shared.fit_calibrators.fit_calibrators`. Mixing up the
    order produces silently wrong results.

    Args:
        calibrators: Dict ``{class_name: calibrator}`` as returned by
            :func:`~tracks.representation.training.shared.fit_calibrators.fit_calibrators`.
        probs: Float array of shape ``(n_samples, n_classes)`` — raw model
            output probabilities.
        class_list: Sequence of class name strings, length ``n_classes``.

    Returns:
        np.ndarray: Calibrated probability array of shape
            ``(n_samples, n_classes)``, clipped to ``[0, 1]``.
    """
    res = np.zeros_like(probs)
    for i, cls in enumerate(class_list):
        cal = calibrators[cls]
        p_c = probs[:, i].reshape(-1, 1)
        if cal is None:
            logger.debug("apply_calibrators: %s — no calibrator (passthrough).", cls)
            res[:, i] = probs[:, i]
        elif hasattr(cal, 'predict_proba'):
            logger.debug("apply_calibrators: %s — predict_proba calibrator.", cls)
            # Predict probability for the positive class (index 1)
            res[:, i] = cal.predict_proba(p_c)[:, 1]
        else:
            logger.debug("apply_calibrators: %s — predict calibrator (%s).", cls, type(cal).__name__)
            res[:, i] = np.asarray(cal.predict(p_c.reshape(-1))).reshape(-1)
    return np.clip(res, 0.0, 1.0)
