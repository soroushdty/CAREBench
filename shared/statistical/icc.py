"""ICC(2,1) rater-level analysis and Lin's CCC (§3.7 / §5.5).

ICC(2,1): two-way mixed effects, absolute agreement, single measurement.
Implemented analytically using the standard SS decomposition — no external
psychometrics library required.

Lin's CCC reported as the secondary/sensitivity metric.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def icc21(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way mixed, absolute agreement, single measurement.

    Args:
        ratings: (n_subjects, n_raters) array.  Rows = subjects, cols = raters.

    Returns:
        ICC(2,1) scalar, or nan if degenerate.

    Formula (Shrout & Fleiss 1979, eq. 2):
        MS_R = between-subject mean square
        MS_E = error mean square
        MS_C = between-rater mean square
        k    = n_raters, n = n_subjects
        ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + k*(MS_C - MS_E)/n)
    """
    r = np.asarray(ratings, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 2:
        return float("nan")

    n, k = r.shape
    grand_mean = r.mean()

    # Sum of squares
    ss_total = np.sum((r - grand_mean) ** 2)
    ss_rows = k * np.sum((r.mean(axis=1) - grand_mean) ** 2)    # between subjects
    ss_cols = n * np.sum((r.mean(axis=0) - grand_mean) ** 2)    # between raters
    ss_error = ss_total - ss_rows - ss_cols

    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_error / ((n - 1) * (k - 1))

    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if abs(denom) < 1e-12:
        return float("nan")
    return float((ms_r - ms_e) / denom)


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (§3.7 secondary metric).

    CCC = 2·cov(x,y) / (var(x) + var(y) + (mean(x) - mean(y))²)

    Returns nan for degenerate (constant) inputs.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    mu_x, mu_y = x.mean(), y.mean()
    var_x = x.var(ddof=0)
    var_y = y.var(ddof=0)
    cov_xy = np.cov(x, y, ddof=0)[0, 1]
    denom = var_x + var_y + (mu_x - mu_y) ** 2
    if abs(denom) < 1e-12:
        return float("nan")
    return float(2.0 * cov_xy / denom)


# ---------------------------------------------------------------------------
# Load individual physician labels from source dataset
# ---------------------------------------------------------------------------

def load_individual_physician_labels(
    dataset_path: Path,
    sheet_names: dict,
    patient_col: str,
    physician_col: str,
    item_col: str,
    class_list: list[str],
    item_standardizer=None,
) -> dict:
    """Load per-physician binary labels from the raw Excel dataset.

    Reads both the test sheet (survey phase) and interview sheet so that
    individual rater labels are available for ICC computation.

    Args:
        item_standardizer: Optional callable ``str -> str`` that maps raw Excel
            item strings to the same canonical form used in the preprocessed
            pipeline (i.e. the output of the JSON-based item standardization
            stage).  When provided, dict keys are canonical names so that
            ``rater_icc_analysis`` can match them against ``item_texts_test``.

    Returns:
        {
            "test":      {patient_id: {physician_id: {item_text: {class: 0|1}}}},
            "interview": {patient_id: {physician_id: {item_text: {class: 0|1}}}},
        }
    """
    result: dict[str, dict] = {"test": {}, "interview": {}}  # PAIRED-CONTEXT SCHEMA

    for split_key in ("test", "interview"):  # PAIRED-CONTEXT SCHEMA
        sheet = sheet_names.get(split_key)
        if not sheet:
            continue
        try:
            df = pd.read_excel(dataset_path, sheet_name=sheet)
        except Exception as exc:
            logger.warning("Could not load sheet %r: %s", sheet, exc)
            continue

        for _, row in df.iterrows():
            pid = row.get(patient_col)
            phid = row.get(physician_col)
            raw_item = str(row.get(item_col, "")).strip()
            item = item_standardizer(raw_item) if item_standardizer is not None else raw_item

            if pid not in result[split_key]:
                result[split_key][pid] = {}
            if phid not in result[split_key][pid]:
                result[split_key][pid][phid] = {}
            if item not in result[split_key][pid][phid]:
                result[split_key][pid][phid][item] = {}

            for cls in class_list:
                val = row.get(cls)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    result[split_key][pid][phid][item][cls] = int(val)

    return result


# ---------------------------------------------------------------------------
# Main rater ICC analysis
# ---------------------------------------------------------------------------

def rater_icc_analysis(
    y_hat_ca: np.ndarray,
    individual_labels: dict,
    patient_ids_test: np.ndarray,
    item_texts_test: np.ndarray,
    class_list: list[str],
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Rater-level ICC(2,1) analysis (§3.7 / §5.5).

    The model is treated as an additional rater.  For each (patient, physician)
    cell, ICC(2,1) is computed between:
      - Model's continuous predictions in [0, 1] for that patient's items × 10 classes
      - That physician's individual binary labels {0, 1}

    Produces:
      - One model-vs-physician ICC per (patient, physician) cell
      - One within-pair human-human ICC per patient

    Args:
        y_hat_ca:          (n_test, n_classes) context-aware predictions.
        individual_labels: From load_individual_physician_labels() — use interview split.
        patient_ids_test:  (n_test,) patient IDs for each test item.
        item_texts_test:   (n_test,) item text strings.
        class_list:        10 class names.
        n_resamples:       Bootstrap CI resamples.

    Returns dict with keys:
        model_physician_iccs: list of {patient, physician, icc, ccc}
        human_human_iccs:     list of {patient, icc, ccc}
        model_physician_icc_mean: float (+ ci_lower, ci_upper)
        human_human_icc_mean:     float (+ ci_lower, ci_upper)
        model_physician_ccc_mean: float
        human_human_ccc_mean:     float
    """
    if rng is None:
        rng = np.random.default_rng()

    ca = np.asarray(y_hat_ca, dtype=np.float64)
    pid_arr = np.asarray(patient_ids_test)
    items_arr = np.asarray(item_texts_test)
    n_classes = len(class_list)

    interview_labels = individual_labels.get("interview", {})  # PAIRED-CONTEXT SCHEMA

    model_physician_iccs: list[dict] = []
    human_human_iccs: list[dict] = []

    for pid in np.unique(pid_arr):
        pid_mask = pid_arr == pid
        patient_items = items_arr[pid_mask]
        patient_preds = ca[pid_mask]   # (n_items_for_patient, n_classes)

        if pid not in interview_labels:
            logger.warning("No interview labels for patient %s — skipping ICC.", pid)
            continue

        physicians = list(interview_labels[pid].keys())
        if len(physicians) < 2:
            logger.warning(
                "Patient %s has fewer than 2 physicians in interview sheet — "
                "skipping within-pair ICC.",
                pid,
            )

        def _labels_matrix(phid: object) -> np.ndarray | None:
            """Build (n_items, n_classes) binary label matrix for one physician."""
            ph_data = interview_labels[pid].get(phid, {})
            rows = []
            for item_text in patient_items:
                item_data = ph_data.get(str(item_text).strip(), {})
                row = [float(item_data.get(cls, float("nan"))) for cls in class_list]
                rows.append(row)
            mat = np.array(rows, dtype=np.float64)
            return mat if mat.size > 0 else None

        # Model-vs-physician ICCs
        for phid in physicians:
            ph_mat = _labels_matrix(phid)
            if ph_mat is None:
                continue
            # Keep only rows where physician label is non-NaN
            valid = ~np.isnan(ph_mat).all(axis=1)
            if valid.sum() < 2:
                continue
            ph_valid = ph_mat[valid]
            pred_valid = patient_preds[valid]

            # Flatten to (n_items * n_classes, 2) for ICC computation
            n_items = ph_valid.shape[0]
            ratings = np.stack([
                pred_valid.ravel(),
                ph_valid.ravel(),
            ], axis=1)  # (n_items * n_classes, 2)

            model_physician_iccs.append({
                "patient": pid,  # PAIRED-CONTEXT SCHEMA
                "physician": phid,  # PAIRED-CONTEXT SCHEMA
                "icc": icc21(ratings),
                "ccc": lins_ccc(pred_valid.ravel(), ph_valid.ravel()),
                "n_items": n_items,
            })

        # Within-pair human-human ICC (uses the two physicians together)
        if len(physicians) >= 2:
            ph0_mat = _labels_matrix(physicians[0])
            ph1_mat = _labels_matrix(physicians[1])
            if ph0_mat is not None and ph1_mat is not None:
                valid = ~(np.isnan(ph0_mat).all(axis=1) | np.isnan(ph1_mat).all(axis=1))
                if valid.sum() >= 2:
                    r0 = ph0_mat[valid].ravel()
                    r1 = ph1_mat[valid].ravel()
                    ratings_hh = np.stack([r0, r1], axis=1)
                    human_human_iccs.append({
                        "patient": pid,  # PAIRED-CONTEXT SCHEMA
                        "icc": icc21(ratings_hh),
                        "ccc": lins_ccc(r0, r1),
                        "n_items": int(valid.sum()),
                    })

    # Aggregate summary with bootstrap CIs
    mp_icc_vals = np.array([d["icc"] for d in model_physician_iccs], dtype=float)
    hh_icc_vals = np.array([d["icc"] for d in human_human_iccs], dtype=float)
    mp_ccc_vals = np.array([d["ccc"] for d in model_physician_iccs], dtype=float)
    hh_ccc_vals = np.array([d["ccc"] for d in human_human_iccs], dtype=float)

    def _nanmean(arr: np.ndarray) -> float:
        return float(np.nanmean(arr)) if len(arr) > 0 else float("nan")

    from shared.statistical.bootstrap import patient_block_bootstrap

    # Bootstrap CI on model-physician ICC mean, clustered by patient
    mp_patient_ids = np.array([d["patient"] for d in model_physician_iccs])  # PAIRED-CONTEXT SCHEMA
    mp_icc_finite = np.where(np.isnan(mp_icc_vals), 0.0, mp_icc_vals)

    def _mp_icc_mean(idx: np.ndarray) -> float:
        return float(np.mean(mp_icc_finite[idx]))

    if len(mp_patient_ids) >= 2:
        mp_ci_lo, mp_ci_hi = patient_block_bootstrap(
            _mp_icc_mean, mp_patient_ids, n_resamples, rng
        )
    else:
        mp_ci_lo = mp_ci_hi = float("nan")

    # Bootstrap CI on human-human ICC mean, clustered by patient
    hh_patient_ids = np.array([d["patient"] for d in human_human_iccs])  # PAIRED-CONTEXT SCHEMA
    hh_icc_finite = np.where(np.isnan(hh_icc_vals), 0.0, hh_icc_vals)

    def _hh_icc_mean(idx: np.ndarray) -> float:
        return float(np.mean(hh_icc_finite[idx]))

    if len(hh_patient_ids) >= 2:
        hh_ci_lo, hh_ci_hi = patient_block_bootstrap(
            _hh_icc_mean, hh_patient_ids, n_resamples, rng
        )
    else:
        hh_ci_lo = hh_ci_hi = float("nan")

    return {
        "model_physician_iccs": model_physician_iccs,
        "human_human_iccs": human_human_iccs,
        "model_physician_icc_mean": _nanmean(mp_icc_vals),
        "model_physician_icc_ci_lower": mp_ci_lo,
        "model_physician_icc_ci_upper": mp_ci_hi,
        "model_physician_icc_sd": float(np.nanstd(mp_icc_vals)),
        "human_human_icc_mean": _nanmean(hh_icc_vals),
        "human_human_icc_ci_lower": hh_ci_lo,
        "human_human_icc_ci_upper": hh_ci_hi,
        "human_human_icc_sd": float(np.nanstd(hh_icc_vals)),
        "human_human_icc_min": float(np.nanmin(hh_icc_vals)) if len(hh_icc_vals) else float("nan"),
        "human_human_icc_max": float(np.nanmax(hh_icc_vals)) if len(hh_icc_vals) else float("nan"),
        "model_physician_ccc_mean": _nanmean(mp_ccc_vals),
        "human_human_ccc_mean": _nanmean(hh_ccc_vals),
        "n_model_physician_pairs": len(model_physician_iccs),
        "n_human_human_pairs": len(human_human_iccs),
    }
