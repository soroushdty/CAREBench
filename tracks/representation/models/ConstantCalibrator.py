import numpy as np


class ConstantCalibrator:
    """Degenerate calibrator that always returns the training label prevalence.

    Used as a fallback by :func:`~tracks.representation.training.shared.fit_calibrators.fit_calibrators`
    when the validation subset for a class is all-positive, all-negative, or
    smaller than ``calibration_min_samples``. In those cases a proper sigmoid
    or isotonic calibrator cannot be fitted reliably, so this class returns a
    fixed probability equal to the observed prevalence.

    The interface mirrors scikit-learn calibrators so that
    :func:`~tracks.representation.training.shared.apply_calibrators.apply_calibrators` and
    :meth:`~tracks.representation.models.EnsemblePredictor.EnsemblePredictor.predict_proba` can
    dispatch to it without special-casing.
    """

    def __init__(self, prob):
        """
        Args:
            prob: Constant probability to return for the positive class.
                Typically the empirical label mean of the validation subset.
        """
        self.prob = prob

    def predict(self, X):
        """Return a constant probability for every sample.

        Args:
            X: Array-like of shape ``(n_samples,)`` or ``(n_samples, 1)``.
                Values are ignored; only the length is used.

        Returns:
            np.ndarray: 1-D float array of shape ``(n_samples,)`` filled with
                ``self.prob``.
        """
        return np.full(X.shape[0], self.prob)

    def predict_proba(self, X):
        """Return constant class probabilities for every sample.

        Args:
            X: Array-like of shape ``(n_samples,)`` or ``(n_samples, 1)``.
                Values are ignored; only the length is used.

        Returns:
            np.ndarray: Float array of shape ``(n_samples, 2)`` where column 0
                is ``1 - self.prob`` (negative class) and column 1 is
                ``self.prob`` (positive class).
        """
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])
