"""
Hypothesis Analyzer for Track 3 (reasoning).

Computes Track 3's endpoints (shared/endpoints.py) with patient-cluster
bootstrap CIs and permutation-test p-values: context sensitivity (formerly
H1), directional alignment (H2), class-level correspondence (H3) and context
specificity (H4, patient-cluster sign-flip test). Method names keep the old
numbers (compute_h1 ... compute_h4).

Reuses ``shared.statistical.bootstrap.patient_block_bootstrap`` for
all bootstrap confidence intervals.

Canonical location: shared/evaluation/hypothesis_analyzer.py
"""

# Terminology mapping (canonical terms):
#   physician / human rater → reference_observer
#   Patient                 → context_entity_id
#   Item / input_text       → task_instance
#   class / category        → output_dimension
#   model_id / llm_*        → candidate_id
#   physician_survey/survey → reference_context_free
#   physician_interview/interview → reference_correct_context
#   delta_physician/human_shift   → delta_reference

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import scipy.stats

from shared.label_space import DEFAULT_LABEL_SPACE
from shared.statistical.bootstrap import patient_block_bootstrap
from shared.statistical.cluster_tests import cluster_sign_flip_test

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Privacy category names (canonical order)
# ---------------------------------------------------------------------------

# Default label space; the analyzer itself uses ``paired_dataset.category_names``.
CATEGORY_NAMES: list[str] = DEFAULT_LABEL_SPACE.keys()


# ---------------------------------------------------------------------------
# Standalone bootstrap / permutation helpers
# ---------------------------------------------------------------------------


def patient_cluster_bootstrap(
    statistic_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Patient-cluster bootstrap CI wrapping ``patient_block_bootstrap``.

    Resamples entire patient clusters with replacement.  For each resample,
    all items belonging to the drawn patients are collected and passed to
    ``statistic_fn`` as an index array into ``data``.

    Parameters
    ----------
    statistic_fn:
        Callable that accepts an index array (subset of ``0..len(data)-1``)
        and returns a scalar statistic.
    data:
        Array of shape ``(N, ...)`` — the full dataset.  Not used directly
        here; the caller's ``statistic_fn`` closes over it.
    patient_ids:
        1-D array of patient IDs (context_entity_id), one per row of ``data``.
    n_resamples:
        Number of bootstrap resamples (default 1000).
    rng:
        Optional ``numpy.random.Generator``; created fresh if ``None``.

    Returns
    -------
    (ci_lower, ci_upper)
        2.5th and 97.5th percentiles of the bootstrap distribution.
    """
    patient_ids = np.asarray(patient_ids)
    return patient_block_bootstrap(
        metric_fn=statistic_fn,
        patient_ids=patient_ids,
        n_resamples=n_resamples,
        rng=rng,
        ci_level=0.95,
    )


def permutation_test_h3(
    mean_delta_physician: np.ndarray,
    mean_delta_llm: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Permutation test for H3 (class-level Pearson correlation).

    Permutes the D class indices of ``mean_delta_llm`` (not the raw data)
    and recomputes Pearson r with the fixed ``mean_delta_physician``
    (canonical: delta_reference for reference_observer).

    Parameters
    ----------
    mean_delta_physician:
        Shape ``(D,)`` — class-level mean reference_observer deltas
        (delta_reference per output_dimension).
    mean_delta_llm:
        Shape ``(D,)`` — class-level mean candidate_id deltas.
    n_permutations:
        Number of permutations (default 10 000).
    rng:
        Optional ``numpy.random.Generator``; created fresh if ``None``.

    Returns
    -------
    float
        p-value = fraction of permuted r values ≥ observed r.
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_delta_physician = np.asarray(mean_delta_physician, dtype=float)
    mean_delta_llm = np.asarray(mean_delta_llm, dtype=float)

    observed_r, _ = scipy.stats.pearsonr(mean_delta_physician, mean_delta_llm)

    permuted_r = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(mean_delta_llm)
        r_perm, _ = scipy.stats.pearsonr(mean_delta_physician, shuffled)
        permuted_r[i] = r_perm

    p_value = float(np.mean(permuted_r >= observed_r))
    return p_value


def permutation_test_h4(
    align_correct: np.ndarray,
    align_shuffled: np.ndarray,
    patient_ids: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Patient-cluster sign-flip test for H4 (correct vs shuffled alignment).

    Under the null, correct and shuffled context are exchangeable for a
    patient, so the sign of each patient's summed difference
    ``align_correct - align_shuffled`` can be flipped.  All of a patient's
    cells (context_entity_id) are flipped together, which preserves
    within-patient correlation.  Exact when ``2 ** n_patients <=
    n_permutations``; see ``shared.statistical.cluster_tests``.

    Parameters
    ----------
    align_correct:
        1-D array of alignment values for the reference_correct_context
        condition (already restricted to non-zero delta_reference cells).
    align_shuffled:
        1-D array of alignment values for the shuffled-context condition,
        parallel to ``align_correct``.
    patient_ids:
        1-D array of context_entity_id values parallel to ``align_correct``.
    n_permutations:
        Monte Carlo replicates when exact enumeration is too large
        (default 10 000).
    rng:
        Optional ``numpy.random.Generator`` for the Monte Carlo path.

    Returns
    -------
    float
        One-sided p-value for mean(correct - shuffled) > 0.
    """
    align_correct = np.asarray(align_correct, dtype=float)
    align_shuffled = np.asarray(align_shuffled, dtype=float)
    return cluster_sign_flip_test(
        align_correct - align_shuffled,
        np.asarray(patient_ids),
        n_permutations=n_permutations,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# HypothesisAnalyzer
# ---------------------------------------------------------------------------


class HypothesisAnalyzer:
    """Computes Track 3's four endpoints (formerly H1–H4) for Track 3 (reasoning).

    Parameters
    ----------
    paired_dataset:
        A ``PairedDataset`` instance (from ``tracks.reasoning.dataset_loader``).
        Used to extract ``patient_ids`` (context_entity_id) and
        ``category_names`` (output_dimension).
    cfg:
        Configuration dict with keys:
        - ``bootstrap_seed`` (int)
        - ``permutation_seed`` (int)
        - ``n_bootstrap_resamples`` (int, default 1000)
        - ``n_permutations`` (int, default 10 000)
    """

    def __init__(self, paired_dataset: Any, cfg: dict[str, Any]) -> None:
        self._patient_ids: np.ndarray = np.asarray(paired_dataset.patient_ids)
        self._category_names: list[str] = list(paired_dataset.category_names)
        self._cfg = cfg

        self._n_bootstrap: int = int(cfg.get("n_bootstrap_resamples", 1000))
        self._n_permutations: int = int(cfg.get("n_permutations", 10_000))
        self._bootstrap_seed: int = int(cfg.get("bootstrap_seed", 42))
        self._permutation_seed: int = int(cfg.get("permutation_seed", 42))

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, *arrays: np.ndarray) -> int:
        """Validate shape consistency across input arrays.

        Checks that all arrays have the same number of rows (N) and that
        all 2-D arrays have the same second dimension (D).

        Parameters
        ----------
        *arrays:
            One or more numpy arrays to validate.

        Returns
        -------
        int
            The output dimension count D (second axis of 2-D arrays).

        Raises
        ------
        ValueError
            If row counts or output dimensions are inconsistent.
        """
        if not arrays:
            raise ValueError("_validate_inputs requires at least one array.")

        row_counts = [a.shape[0] for a in arrays]
        if len(set(row_counts)) > 1:
            raise ValueError(
                f"Row count mismatch across input arrays: {row_counts}"
            )

        dims = set()
        for a in arrays:
            if a.ndim >= 2:
                dims.add(a.shape[1])

        if len(dims) > 1:
            raise ValueError(
                f"Output dimension (column count) mismatch across input "
                f"arrays: {sorted(dims)}"
            )

        if not dims:
            raise ValueError(
                "No 2-D arrays provided; cannot determine output dimension D."
            )

        return dims.pop()

    # ------------------------------------------------------------------
    # H1 — Context Sensitivity
    # ------------------------------------------------------------------

    def compute_h1(
        self,
        context_free_scores: np.ndarray,
        correct_context_scores: np.ndarray,
    ) -> dict[str, Any]:
        """Compute H1: mean absolute Delta_LLM_Correct with bootstrap CI.

        Parameters
        ----------
        context_free_scores:
            Shape ``(N, D)`` — candidate_id scores under
            reference_context_free condition.
        correct_context_scores:
            Shape ``(N, D)`` — candidate_id scores under
            reference_correct_context condition.

        Returns
        -------
        dict with keys:
            - ``mean_abs_delta`` (float): aggregate point estimate
            - ``ci_lower`` (float): 2.5th percentile of bootstrap distribution
            - ``ci_upper`` (float): 97.5th percentile
            - ``per_category`` (dict): per output_dimension results, each with
              ``mean_abs_delta``, ``ci_lower``, ``ci_upper``
        """
        context_free_scores = np.asarray(context_free_scores, dtype=float)
        correct_context_scores = np.asarray(correct_context_scores, dtype=float)
        D = self._validate_inputs(context_free_scores, correct_context_scores)

        delta_llm_correct = correct_context_scores - context_free_scores  # (N, D)
        abs_delta = np.abs(delta_llm_correct)  # (N, D)

        # --- Aggregate ---
        aggregate_point = float(np.mean(abs_delta))

        rng_boot = np.random.default_rng(self._bootstrap_seed)

        def _agg_stat(idx: np.ndarray) -> float:
            return float(np.mean(abs_delta[idx]))

        ci_lower, ci_upper = patient_cluster_bootstrap(
            statistic_fn=_agg_stat,
            data=abs_delta,
            patient_ids=self._patient_ids,
            n_resamples=self._n_bootstrap,
            rng=rng_boot,
        )

        # --- Per-category ---
        per_category: dict[str, dict[str, float]] = {}
        for c_idx, cat_name in enumerate(self._category_names):
            cat_abs_delta = abs_delta[:, c_idx]  # (N,)
            cat_point = float(np.mean(cat_abs_delta))

            # Fresh RNG seeded deterministically per category
            rng_cat = np.random.default_rng(self._bootstrap_seed + c_idx + 1)

            def _cat_stat(idx: np.ndarray, _col: np.ndarray = cat_abs_delta) -> float:
                return float(np.mean(_col[idx]))

            cat_lo, cat_hi = patient_cluster_bootstrap(
                statistic_fn=_cat_stat,
                data=cat_abs_delta,
                patient_ids=self._patient_ids,
                n_resamples=self._n_bootstrap,
                rng=rng_cat,
            )
            per_category[cat_name] = {
                "mean_abs_delta": cat_point,
                "ci_lower": cat_lo,
                "ci_upper": cat_hi,
            }

        return {
            "mean_abs_delta": aggregate_point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "per_category": per_category,
        }

    # ------------------------------------------------------------------
    # H2 — Directional Physician Alignment
    # ------------------------------------------------------------------

    def compute_h2(
        self,
        context_free_scores: np.ndarray,
        correct_context_scores: np.ndarray,
        delta_physician: np.ndarray,
    ) -> dict[str, Any]:
        """Compute H2: mean alignment and sign agreement rate with bootstrap CIs.

        Restricted to cells where ``delta_physician`` (delta_reference) != 0.

        Parameters
        ----------
        context_free_scores:
            Shape ``(N, D)``.
        correct_context_scores:
            Shape ``(N, D)``.
        delta_physician:
            Shape ``(N, D)`` — reference_observer delta matrix
            (delta_reference: reference_correct_context minus
            reference_context_free).

        Returns
        -------
        dict with keys:
            - ``mean_alignment`` (float)
            - ``ci_lower_alignment`` (float)
            - ``ci_upper_alignment`` (float)
            - ``sign_agreement_rate`` (float)
            - ``ci_lower_sign_agree`` (float)
            - ``ci_upper_sign_agree`` (float)

        All values are ``NaN`` if there are no non-zero delta_reference cells.
        """
        context_free_scores = np.asarray(context_free_scores, dtype=float)
        correct_context_scores = np.asarray(correct_context_scores, dtype=float)
        delta_physician = np.asarray(delta_physician, dtype=float)
        D = self._validate_inputs(context_free_scores, correct_context_scores, delta_physician)

        delta_llm_correct = correct_context_scores - context_free_scores  # (N, D)
        nonzero_mask = delta_physician != 0.0  # (N, D) bool

        if not np.any(nonzero_mask):
            logger.warning("H2: no non-zero Delta_Physician cells found. Returning NaN.")
            nan = float("nan")
            return {
                "mean_alignment": nan,
                "ci_lower_alignment": nan,
                "ci_upper_alignment": nan,
                "sign_agreement_rate": nan,
                "ci_lower_sign_agree": nan,
                "ci_upper_sign_agree": nan,
            }

        # Flatten to 1-D arrays of non-zero cells
        # Repeat patient_ids to match (N, D) shape, then mask
        patient_ids_2d = np.tile(self._patient_ids[:, None], (1, D))  # (N, D)
        patient_ids_nz = patient_ids_2d[nonzero_mask]  # (M,)

        alignment_2d = np.sign(delta_physician) * delta_llm_correct  # (N, 10)
        alignment_nz = alignment_2d[nonzero_mask]  # (M,)

        sign_agree_2d = (
            np.sign(delta_physician) == np.sign(delta_llm_correct)
        ).astype(float)  # (N, 10)
        sign_agree_nz = sign_agree_2d[nonzero_mask]  # (M,)

        # --- Mean alignment ---
        mean_align = float(np.mean(alignment_nz))

        rng_align = np.random.default_rng(self._bootstrap_seed)

        def _align_stat(idx: np.ndarray) -> float:
            return float(np.mean(alignment_nz[idx]))

        ci_lo_align, ci_hi_align = patient_cluster_bootstrap(
            statistic_fn=_align_stat,
            data=alignment_nz,
            patient_ids=patient_ids_nz,
            n_resamples=self._n_bootstrap,
            rng=rng_align,
        )

        # --- Sign agreement rate ---
        sign_agree_rate = float(np.mean(sign_agree_nz))

        rng_sign = np.random.default_rng(self._bootstrap_seed + 1)

        def _sign_stat(idx: np.ndarray) -> float:
            return float(np.mean(sign_agree_nz[idx]))

        ci_lo_sign, ci_hi_sign = patient_cluster_bootstrap(
            statistic_fn=_sign_stat,
            data=sign_agree_nz,
            patient_ids=patient_ids_nz,
            n_resamples=self._n_bootstrap,
            rng=rng_sign,
        )

        return {
            "mean_alignment": mean_align,
            "ci_lower_alignment": ci_lo_align,
            "ci_upper_alignment": ci_hi_align,
            "sign_agreement_rate": sign_agree_rate,
            "ci_lower_sign_agree": ci_lo_sign,
            "ci_upper_sign_agree": ci_hi_sign,
        }

    # ------------------------------------------------------------------
    # H3 — Class-Level Correspondence
    # ------------------------------------------------------------------

    def compute_h3(
        self,
        context_free_scores: np.ndarray,
        correct_context_scores: np.ndarray,
        delta_physician: np.ndarray,
    ) -> dict[str, Any]:
        """Compute H3: Pearson correlation of class-level mean deltas.

        Parameters
        ----------
        context_free_scores:
            Shape ``(N, D)``.
        correct_context_scores:
            Shape ``(N, D)``.
        delta_physician:
            Shape ``(N, D)`` — delta_reference matrix.

        Returns
        -------
        dict with keys:
            - ``pearson_r`` (float)
            - ``p_value`` (float) — permutation test p-value
            - ``mean_delta_physician`` (np.ndarray, shape (D,))
            - ``mean_delta_llm`` (np.ndarray, shape (D,))
        """
        context_free_scores = np.asarray(context_free_scores, dtype=float)
        correct_context_scores = np.asarray(correct_context_scores, dtype=float)
        delta_physician = np.asarray(delta_physician, dtype=float)
        D = self._validate_inputs(context_free_scores, correct_context_scores, delta_physician)

        delta_llm_correct = correct_context_scores - context_free_scores  # (N, D)

        mean_delta_physician = delta_physician.mean(axis=0)   # (D,)
        mean_delta_llm = delta_llm_correct.mean(axis=0)       # (D,)

        pearson_r, _ = scipy.stats.pearsonr(mean_delta_physician, mean_delta_llm)

        rng_perm = np.random.default_rng(self._permutation_seed)
        p_value = permutation_test_h3(
            mean_delta_physician=mean_delta_physician,
            mean_delta_llm=mean_delta_llm,
            n_permutations=self._n_permutations,
            rng=rng_perm,
        )

        return {
            "pearson_r": float(pearson_r),
            "p_value": p_value,
            "mean_delta_physician": mean_delta_physician,
            "mean_delta_llm": mean_delta_llm,
        }

    # ------------------------------------------------------------------
    # H4 — Correct vs Shuffled Context Control
    # ------------------------------------------------------------------

    def compute_h4(
        self,
        context_free_scores: np.ndarray,
        correct_context_scores: np.ndarray,
        shuffled_context_scores: np.ndarray,
        delta_physician: np.ndarray,
    ) -> dict[str, Any]:
        """Compute H4: mean alignment difference (correct vs shuffled) with CI.

        Parameters
        ----------
        context_free_scores:
            Shape ``(N, D)``.
        correct_context_scores:
            Shape ``(N, D)``.
        shuffled_context_scores:
            Shape ``(N, D)``.
        delta_physician:
            Shape ``(N, D)`` — delta_reference matrix.

        Returns
        -------
        dict with keys:
            - ``mean_diff`` (float)
            - ``ci_lower`` (float)
            - ``ci_upper`` (float)
            - ``p_value`` (float) — paired permutation test p-value

        All values are ``NaN`` if there are no non-zero delta_reference cells.
        """
        context_free_scores = np.asarray(context_free_scores, dtype=float)
        correct_context_scores = np.asarray(correct_context_scores, dtype=float)
        shuffled_context_scores = np.asarray(shuffled_context_scores, dtype=float)
        delta_physician = np.asarray(delta_physician, dtype=float)
        D = self._validate_inputs(
            context_free_scores, correct_context_scores,
            shuffled_context_scores, delta_physician,
        )

        delta_llm_correct = correct_context_scores - context_free_scores    # (N, D)
        delta_llm_shuffled = shuffled_context_scores - context_free_scores  # (N, D)

        align_correct_2d = np.sign(delta_physician) * delta_llm_correct    # (N, D)
        align_shuffled_2d = np.sign(delta_physician) * delta_llm_shuffled  # (N, D)

        nz = delta_physician != 0.0  # (N, D) bool

        if not np.any(nz):
            logger.warning("H4: no non-zero Delta_Physician cells found. Returning NaN.")
            nan = float("nan")
            return {
                "mean_diff": nan,
                "ci_lower": nan,
                "ci_upper": nan,
                "p_value": nan,
            }

        # Flatten to 1-D arrays of non-zero cells
        patient_ids_2d = np.tile(self._patient_ids[:, None], (1, D))  # (N, D)
        patient_ids_nz = patient_ids_2d[nz]  # (M,)

        align_correct_nz = align_correct_2d[nz]    # (M,)
        align_shuffled_nz = align_shuffled_2d[nz]  # (M,)
        diff_nz = align_correct_nz - align_shuffled_nz  # (M,)

        mean_diff = float(np.mean(diff_nz))

        # --- Bootstrap CI ---
        rng_boot = np.random.default_rng(self._bootstrap_seed)

        def _diff_stat(idx: np.ndarray) -> float:
            return float(np.mean(diff_nz[idx]))

        ci_lower, ci_upper = patient_cluster_bootstrap(
            statistic_fn=_diff_stat,
            data=diff_nz,
            patient_ids=patient_ids_nz,
            n_resamples=self._n_bootstrap,
            rng=rng_boot,
        )

        # --- Paired permutation test ---
        rng_perm = np.random.default_rng(self._permutation_seed)
        p_value = permutation_test_h4(
            align_correct=align_correct_nz,
            align_shuffled=align_shuffled_nz,
            patient_ids=patient_ids_nz,
            n_permutations=self._n_permutations,
            rng=rng_perm,
        )

        return {
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
        }
