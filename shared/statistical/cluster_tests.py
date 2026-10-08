"""Patient-cluster sign-flip randomization test.

Cells (for example (patient, item, class) observations) from the same
patient share a context snapshot and a reference observer group, so they
are correlated.  Tests that treat cells as independent understate the
p-value.  The sign-flip test here keeps each patient's cells together: under
the null, each patient's summed contribution is symmetric about zero, so its
sign can be flipped as a block.

With few patients the test is exact: all ``2 ** n_clusters`` sign patterns
are enumerated when that number does not exceed ``n_permutations``.  This
also means the smallest attainable one-sided p-value is ``2 ** -n_clusters``
(1/64 with six patients), which is the honest limit of a design with that
many patients.
"""
from __future__ import annotations

import itertools

import numpy as np


def cluster_sign_flip_test(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """One-sided cluster sign-flip test of ``mean(values) > 0``.

    The statistic is the mean over all cells.  Each null replicate multiplies
    every cell of a cluster by the same random sign, so within-cluster
    correlation is preserved.

    Args:
        values: (n,) cell-level values whose mean is zero under the null
            (for example ``agree - 0.5`` or a paired difference).
        cluster_ids: (n,) cluster (patient) ID for each cell.
        n_permutations: Monte Carlo replicates when exact enumeration is too
            large.  Exact enumeration is used when
            ``2 ** n_clusters <= n_permutations``.
        rng: Generator for the Monte Carlo path.  Not used when the test is
            exact.

    Returns:
        One-sided p-value.  NaN if there are no cells or fewer than two
        clusters; 1.0 if every cluster sums to zero.
    """
    values = np.asarray(values, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape != cluster_ids.shape:
        raise ValueError(
            f"values and cluster_ids must have the same shape, got "
            f"{values.shape} and {cluster_ids.shape}"
        )
    if values.size == 0:
        return float("nan")

    _, inverse = np.unique(cluster_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)  # (n_clusters,)
    n_clusters = sums.size
    if n_clusters < 2:
        return float("nan")
    if np.all(sums == 0.0):
        return 1.0

    observed = float(sums.sum())
    tol = 1e-12 * max(1.0, float(np.abs(sums).sum()))

    if 2 ** n_clusters <= n_permutations:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n_clusters)))
        null = signs @ sums
        return float(np.mean(null >= observed - tol))

    if rng is None:
        rng = np.random.default_rng()
    signs = rng.choice(np.array([1.0, -1.0]), size=(n_permutations, n_clusters))
    null = signs @ sums
    return float((1 + np.sum(null >= observed - tol)) / (n_permutations + 1))
