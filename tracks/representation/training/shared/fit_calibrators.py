import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

from tracks.representation.models.ConstantCalibrator import ConstantCalibrator
from .soft_label_utils import masked_class_data

logger = logging.getLogger(__name__)


def fit_calibrators(probs, Y_true, class_list, cfg):
    """Fit per-class probability calibrators on held-out validation predictions.

    Supported calibration methods (``cfg["calibration_method"]``): ``"isotonic"``
    (default) and ``"none"`` (no calibration — stores ``None`` per class).

    Falls back to :class:`~models.ConstantCalibrator.ConstantCalibrator` for a
    class when any of the following conditions hold on the masked validation
    subset: no valid samples, a single unique label value (all-positive or
    all-negative), or fewer samples than ``cfg["calibration_min_samples"]``
    (default 10). The constant value is the empirical label mean of the subset.

    Args:
        probs: Float array of shape ``(n_samples, n_classes)`` — raw model
            output probabilities from the validation fold.
        Y_true: Array-like of shape ``(n_samples, n_classes)`` — ground-truth
            labels (may be soft/float).
        class_list: Sequence of class name strings, length ``n_classes``.
            Determines the keys of the returned dict.
        cfg: Config mapping. Relevant keys: ``calibration_method``,
            ``calibration_min_samples``, and any keys consumed by
            :func:`~tracks.representation.training.shared.soft_label_utils.masked_class_data`.

    Returns:
        dict: ``{class_name: calibrator}`` mapping with one entry per class.
            Values are fitted :class:`~sklearn.isotonic.IsotonicRegression`
            instances, :class:`~models.ConstantCalibrator.ConstantCalibrator`
            instances, or ``None`` (when ``calibration_method == "none"``).

    Raises:
        ValueError: If ``calibration_method`` is not ``"none"`` or
            ``"isotonic"``.
    """
    method = str(cfg.get("calibration_method", "isotonic")).strip().lower()
    min_samples = int(cfg.get("calibration_min_samples", 10))
    calibrators = {}
    for i, cls in enumerate(class_list):
        _, Y_bin, probs_c = masked_class_data(Y_true, probs, i, cfg)
        if method == "none":
            calibrators[cls] = None
            continue

        # ConstantCalibrator is the safe fallback for three degenerate cases:
        #   (a) no valid samples in this fold's validation set for the class,
        #   (b) all validation labels are the same value (all-positive or all-negative),
        #   (c) too few samples to fit a monotonic regression reliably.
        # In all cases Y_bin.mean() is the maximum-likelihood estimate of label
        # prevalence and is numerically stable; it produces constant calibrated
        # scores, which is the least-wrong option when the fold provides no
        # calibration signal.  A ConstantCalibrator(0.5) is used only when no
        # samples exist at all, since mean() would be undefined.
        if Y_bin.size == 0:
            logger.warning("Calibration fallback to ConstantCalibrator(0.5) for %s: no valid samples.", cls)
            calibrators[cls] = ConstantCalibrator(0.5)
        elif len(np.unique(Y_bin)) < 2:
            logger.warning(
                "Calibration fallback to ConstantCalibrator for %s: validation subset has a single class.",
                cls,
            )
            calibrators[cls] = ConstantCalibrator(float(Y_bin.mean()))
        elif Y_bin.size < min_samples:
            logger.warning(
                "Calibration fallback to ConstantCalibrator for %s: only %d validation samples (< %d).",
                cls,
                int(Y_bin.size),
                min_samples,
            )
            calibrators[cls] = ConstantCalibrator(float(Y_bin.mean()))
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs_c.reshape(-1), Y_bin)
            calibrators[cls] = iso
        else:
            raise ValueError(f"Unsupported calibration_method '{method}'. Use one of: 'none', 'isotonic'.")
    return calibrators
