"""Array coercion helpers shared across the pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def as_float32_array(values, name: str) -> np.ndarray:
    """Coerce array-like values to float32, handling pandas nullable/object dtypes.

    NaN values after coercion are filled with 0.0 and a warning is emitted.
    """
    arr = np.asarray(values)
    if arr.dtype == np.object_:
        flat = pd.to_numeric(pd.Series(arr.reshape(-1)), errors="coerce")
        arr = flat.to_numpy(dtype=np.float32).reshape(arr.shape)
    else:
        arr = arr.astype(np.float32, copy=False)

    nan_count = int(np.isnan(arr).sum())
    if nan_count > 0:
        logger.warning(
            "%s contains %d NaN values after coercion; filling with 0.0.", name, nan_count
        )
        arr = np.nan_to_num(arr, nan=0.0)
    return arr
