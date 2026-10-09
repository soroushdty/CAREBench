import logging
import numpy as np
import torch
import torch.nn as nn

from ..shared.lopo_cv import lopo_splits

logger = logging.getLogger(__name__)


class _PassthroughSentinel:
    """Returned by fit_stage2_fusion_fold when passthrough beats all learned fusion candidates."""
    def eval(self): return self
    def parameters(self): return iter([])


class Stage2FusionModel(nn.Module):
    """Ridge-regularized linear head: ŷ_ca = σ(V^T z + b).

    forward() returns raw logits (no sigmoid). Callers that need probabilities
    must apply sigmoid explicitly. This allows BCEWithLogitsLoss to be used for
    numerically stable training with correct pos_weight semantics.
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.linear(Z)


class LowRankBilinearFusionModel(nn.Module):
    """Per-class low-rank bilinear head: ŷ_c = e_i^T (U_c V_c^T) c_p + b_c.

    Input Z must be the concatenation [e_i ‖ c_p] with shape (batch, 2d).
    U_c and V_c are learned per class: shape (n_classes, d, r).
    This gives 2·d·r parameters per class.
    forward() returns raw logits.
    """

    def __init__(self, input_dim: int, n_classes: int, r: int = 8):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(
                f"LowRankBilinearFusionModel requires input_dim divisible by 2 "
                f"(got {input_dim}). Z must be [e_i ‖ c_p]."
            )
        d = input_dim // 2
        self.d = d
        self.U = nn.Parameter(torch.randn(n_classes, d, r) * 0.01)
        self.V = nn.Parameter(torch.randn(n_classes, d, r) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        e_i = Z[:, : self.d]   # (batch, d)
        c_p = Z[:, self.d :]   # (batch, d)
        # h_e[b, c, r] = e_i[b] @ U[c], same for h_c
        h_e = torch.einsum("bd,cdr->bcr", e_i, self.U)   # (batch, n_classes, r)
        h_c = torch.einsum("bd,cdr->bcr", c_p, self.V)   # (batch, n_classes, r)
        return (h_e * h_c).sum(-1) + self.bias             # (batch, n_classes)


def _compute_pos_weights(y: np.ndarray, weight_cap: float, cfg: dict | None = None) -> np.ndarray:
    """Per-class positive-class weights capped to [1.0, weight_cap].

    Uses Cui 2019 Effective Number of Samples (default) or inverse-frequency
    weighting according to cfg['class_weight_mode'] and cfg['class_weight_beta'].
    Lower bound of 1.0 matches Stage 1 (hp_search.py) for consistency.
    """
    weight_mode = cfg.get("class_weight_mode", "cui") if cfg else "cui"
    beta = float(cfg.get("class_weight_beta", 0.999)) if cfg else 0.999
    weights = []
    for c in range(y.shape[1]):
        valid = y[:, c][~np.isnan(y[:, c])]
        # Use soft sum (matching Stage 1 hp_search.py) so physician-disagreement
        # labels (0.5) contribute 0.5 to positive mass rather than being
        # binarized to 0 by a "> 0.5" threshold.
        n_pos = float(valid.sum())
        pos_mass = n_pos
        # Use per-class valid count (non-NaN rows) for inverse_frequency so that
        # classes with many missing interview labels are not artificially
        # upweighted by the total row count n_total.  Cui mode is unaffected
        # because it does not use a total-count denominator.
        n_valid_c = int(valid.size)
        if weight_mode == "inverse_frequency":
            w = n_valid_c / (2.0 * max(pos_mass, 1)) if pos_mass > 0 else 1.0
        else:
            w = (1 - beta) / (1 - beta ** max(pos_mass, 1))
        weights.append(min(max(w, 1.0), weight_cap))
    return np.array(weights, dtype=np.float32)


def _train_one_model(
    Z_tr: np.ndarray,
    y_int_tr: np.ndarray,
    y_survey_tr: np.ndarray,
    y_cf_tr: np.ndarray,
    alpha: float,
    lr: float,
    num_epochs: int,
    patience: int,
    weight_cap: float,
    device: torch.device,
    cfg: dict | None = None,
    model_cls=None,
    model_kwargs: dict | None = None,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> "Stage2FusionModel | LowRankBilinearFusionModel":
    """Train one Stage 2 model on the given data.

    Early stopping:
    - If Z_val / y_val are provided: monitors held-out Brier score (correct).
    - Otherwise (final retraining on full data): trains for the full num_epochs
      without early stopping, since ridge regularisation (alpha) prevents
      overfitting without needing a held-out monitor.

    model_cls defaults to Stage2FusionModel.
    """
    n, input_dim = Z_tr.shape
    n_classes = y_int_tr.shape[1]

    if model_cls is None:
        model_cls = Stage2FusionModel
    if model_kwargs is None:
        model_kwargs = {}

    # Build a boolean mask for valid (non-NaN) interview labels.
    # NaN positions are excluded from the BCE loss rather than filled with
    # survey labels, so the model trains only on actual interview observations.
    nan_mask = ~np.isnan(y_int_tr)
    # Fill NaN cells with 0.0 for tensor arithmetic; the mask zeroes them in the loss.
    y_int_filled = np.where(nan_mask, y_int_tr, 0.0).astype(np.float32)

    # Unweighted BCE by default: Stage 2's input (ŷ_cf) is already calibrated by
    # Stage 1, so pos_weight would shift probabilities away from the CF baseline
    # rather than correcting for class imbalance. Weighted mode is available via
    # cfg["stage2_weighted_loss"] = True for ablation only.
    _use_weighted = bool((cfg or {}).get("stage2_weighted_loss", False))
    if _use_weighted:
        pos_w = _compute_pos_weights(y_int_tr, weight_cap, cfg)
        pos_w_t: torch.Tensor | None = torch.tensor(pos_w, device=device)
    else:
        pos_w_t = None

    Z_t        = torch.tensor(Z_tr, dtype=torch.float32, device=device)
    y_int_t    = torch.tensor(y_int_filled, device=device)
    nan_mask_t = torch.tensor(nan_mask, dtype=torch.bool, device=device)

    # Prevalence-shift regularisation: penalise the batch-mean prediction per class
    # from drifting away from the Stage 1 mean. This prevents the head from learning
    # a global upward shift (positive bias for every class) instead of patient-specific
    # corrections. Computed once as a constant reference; does not penalise per-item
    # deviations that cancel across patients.
    _delta_reg = float((cfg or {}).get("stage2_delta_reg_weight", 0.0))
    _mean_cf: torch.Tensor | None = None
    if _delta_reg > 0.0:
        _y_cf_t_reg = torch.tensor(y_cf_tr.astype(np.float32), dtype=torch.float32, device=device)
        _mean_cf = _y_cf_t_reg.mean(dim=0)  # (n_classes,) constant reference

    model = model_cls(input_dim, n_classes, **model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    use_val = Z_val is not None and y_val is not None
    if use_val:
        Z_val_t = torch.tensor(Z_val, dtype=torch.float32, device=device)

    best_score = float("inf")
    no_improve = 0
    best_state: dict | None = None  # only populated when use_val=True

    for _ in range(num_epochs):
        model.train()
        opt.zero_grad()
        logits = model(Z_t)
        # Masked, normalized BCE loss: exclude NaN interview-label positions so
        # the gradient comes only from observed (item, patient, class) triples.
        # reduction="none" + manual mask avoids survey-label substitution and
        # keeps gradient scale stable across folds regardless of NaN density.
        loss_per_elem = nn.functional.binary_cross_entropy_with_logits(
            logits, y_int_t, pos_weight=pos_w_t, reduction="none"
        )
        valid_count = nan_mask_t.sum().clamp(min=1)
        loss = (loss_per_elem * nan_mask_t).sum() / valid_count
        if _delta_reg > 0.0 and _mean_cf is not None:
            mean_preds = torch.sigmoid(logits).mean(dim=0)  # (n_classes,)
            loss = loss + _delta_reg * ((mean_preds - _mean_cf) ** 2).mean()
        loss.backward()
        opt.step()

        if use_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(Z_val_t)
                val_preds = torch.sigmoid(val_logits).cpu().numpy()
            monitor = _brier_score(val_preds, y_val)
        else:
            # No held-out data: skip early stopping, rely on ridge regularisation.
            continue

        if monitor < best_score - 1e-6:
            best_score = monitor
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Only restore the early-stopped best state when validation monitoring was
    # active.  Without validation, best_state is never populated, so restoring
    # it would reset the model to its initial zero weights — the bug this block
    # corrects.
    if use_val and best_state is not None:
        model.load_state_dict(best_state)
    return model


def _brier_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean per-class Brier Score, ignoring NaN targets."""
    scores = []
    for c in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, c])
        if mask.sum() == 0:
            continue
        scores.append(np.mean((y_pred[mask, c] - y_true[mask, c]) ** 2))
    return float(np.mean(scores)) if scores else 1.0


def fit_stage2_fusion_fold(
    Z_tr: np.ndarray,            # (n_tr, fusion_dim) fusion features
    y_int_tr: np.ndarray,        # (n_tr, n_classes) interview labels (may contain NaN)
    y_survey_tr: np.ndarray,     # (n_tr, n_classes) survey labels
    y_cf_tr: np.ndarray,         # (n_tr, n_classes) Stage 1 calibrated predictions
    patient_ids_tr: np.ndarray,  # (n_tr,)
    class_list: list,
    cfg: dict,
    baseline_metrics: dict = None,
    metrics_df: "pd.DataFrame" = None,
) -> "Stage2FusionModel | LowRankBilinearFusionModel":
    """Fit Stage 2 fusion head with inner LOPO HP selection.

    HP grid: stage2_alpha_options only.
    Selection criterion: mean per-class Brier Score on inner held-out patient.
    Returns the best model retrained on all n_tr items.
    """
    alpha_options = cfg.get("stage2_alpha_options", [0.01, 0.1, 1.0, 10.0, 100.0])
    lr            = float(cfg.get("stage2_lr", 0.01))
    num_epochs    = int(cfg.get("stage2_num_epochs", 200))
    patience      = int(cfg.get("stage2_early_stopping_patience", 20))
    weight_cap    = float(cfg.get("weight_cap", 50.0))
    fusion_strategy = cfg.get("fusion_strategy", "2d")
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class based on fusion strategy
    if fusion_strategy == "lowrank_bilinear":
        model_cls    = LowRankBilinearFusionModel
        model_kwargs = {"r": int(cfg.get("lowrank_bilinear_r", 8))}
    else:
        model_cls    = Stage2FusionModel
        model_kwargs = {}

    unique_patients = np.unique(patient_ids_tr)
    hp_grid = list(alpha_options)

    best_hp    = alpha_options[0]
    best_score = float("inf")

    if metrics_df is not None:
        logger.info("Stage 2: Received full metrics DataFrame for fold with shape %s", metrics_df.shape)
    elif baseline_metrics is not None:
        logger.info("Stage 2: Received baseline metrics for fold: %s", baseline_metrics)

    def _passthrough_baseline(y_cf_val):
        return y_cf_val.copy()

    def _item_plus_patient_onehot(Z, patient_ids, reference_ids=None):
        unique_ids = np.unique(reference_ids) if reference_ids is not None else np.unique(patient_ids)
        id_to_idx = {pid: idx for idx, pid in enumerate(unique_ids)}
        onehot = np.zeros((len(patient_ids), len(unique_ids)), dtype=np.float32)
        for i, pid in enumerate(patient_ids):
            if pid in id_to_idx:
                onehot[i, id_to_idx[pid]] = 1.0
        return np.concatenate([Z, onehot], axis=1)

    if metrics_df is not None:
        logger.info(
            "Stage 2: Using new metrics_df input for fold (full DataFrame available). "
            "Legacy arrays still used for fitting."
        )

    if len(unique_patients) >= 2:
        inner_splits = lopo_splits(patient_ids_tr)
        hp_scores = {hp: [] for hp in hp_grid}
        hp_deltas = {hp: [] for hp in hp_grid}  # per-fold per-class mean (preds - y_cf)
        baseline_scores = {"passthrough": [], "item_plus_patient_onehot": []}

        for train_ix, val_ix in inner_splits:
            for alpha in hp_grid:
                m = _train_one_model(
                    Z_tr[train_ix], y_int_tr[train_ix],
                    y_survey_tr[train_ix], y_cf_tr[train_ix],
                    alpha=alpha, lr=lr,
                    num_epochs=num_epochs, patience=patience,
                    weight_cap=weight_cap, device=device,
                    cfg=cfg,
                    model_cls=model_cls, model_kwargs=model_kwargs,
                    Z_val=Z_tr[val_ix], y_val=y_int_tr[val_ix],
                )
                m.eval()
                with torch.no_grad():
                    val_logits = m(torch.tensor(Z_tr[val_ix], dtype=torch.float32, device=device))
                    preds_np = torch.sigmoid(val_logits).cpu().numpy()
                score = _brier_score(preds_np, y_int_tr[val_ix])
                hp_scores[alpha].append(score)
                hp_deltas[alpha].append(np.nanmean(preds_np - y_cf_tr[val_ix], axis=0))

            passthrough_preds = _passthrough_baseline(y_cf_tr[val_ix])
            baseline_scores["passthrough"].append(_brier_score(passthrough_preds, y_int_tr[val_ix]))

            Z_aug_tr  = _item_plus_patient_onehot(Z_tr[train_ix],  patient_ids_tr[train_ix], reference_ids=unique_patients)
            Z_aug_val = _item_plus_patient_onehot(Z_tr[val_ix],    patient_ids_tr[val_ix],   reference_ids=unique_patients)
            m_aug = _train_one_model(
                Z_aug_tr, y_int_tr[train_ix],
                y_survey_tr[train_ix], y_cf_tr[train_ix],
                alpha=0.01, lr=lr,
                num_epochs=num_epochs, patience=patience,
                weight_cap=weight_cap, device=device,
                cfg=cfg,
                Z_val=Z_aug_val, y_val=y_int_tr[val_ix],
            )
            m_aug.eval()
            with torch.no_grad():
                preds_aug_logits = m_aug(torch.tensor(Z_aug_val, dtype=torch.float32, device=device))
                preds_aug_np = torch.sigmoid(preds_aug_logits).cpu().numpy()
            baseline_scores["item_plus_patient_onehot"].append(_brier_score(preds_aug_np, y_int_tr[val_ix]))

        logger.info("Stage 2 Baseline (passthrough) mean Brier: %.4f", float(np.mean(baseline_scores["passthrough"])))
        logger.info("Stage 2 Baseline (item+patient onehot) mean Brier: %.4f", float(np.mean(baseline_scores["item_plus_patient_onehot"])))

        for hp, scores in hp_scores.items():
            mean_s = float(np.mean(scores))
            logger.info("Stage 2 inner HP α=%.4g → Brier=%.4f", hp, mean_s)
            if mean_s < best_score:
                best_score = mean_s
                best_hp    = hp

        # Per-class delta diagnostic: compare model mean delta to physician mean delta.
        # Flags classes where the model shifts in the opposite direction to the physician,
        # which indicates the model is learning a spurious global bias rather than a
        # patient-specific correction.
        _model_deltas = np.nanmean(hp_deltas[best_hp], axis=0) if hp_deltas[best_hp] else None
        if _model_deltas is not None:
            _valid_int = ~np.isnan(y_int_tr)
            _phys_deltas = np.where(
                _valid_int.sum(axis=0) > 0,
                np.nanmean(y_int_tr - y_cf_tr, axis=0),
                np.nan,
            )
            _mismatch_classes = []
            for _c, _cls in enumerate(class_list):
                _md  = float(_model_deltas[_c])
                _phd = float(_phys_deltas[_c]) if not np.isnan(_phys_deltas[_c]) else float("nan")
                if not np.isnan(_phd) and ((_md > 0) != (_phd > 0)):
                    _mismatch_classes.append(_cls)
                    logger.warning(
                        "Stage 2 per-class delta [%s]: model=%.4f, physician=%.4f "
                        "(model shifts %s, physician shifts %s — direction mismatch)",
                        _cls, _md, _phd,
                        "up" if _md > 0 else "down",
                        "up" if _phd > 0 else "down",
                    )
                else:
                    logger.info(
                        "Stage 2 per-class delta [%s]: model=%.4f, physician=%.4f",
                        _cls, _md, _phd,
                    )
            if _mismatch_classes:
                logger.warning(
                    "Stage 2: %d class(es) show direction mismatch between model and "
                    "physician deltas: %s. Consider increasing stage2_delta_reg_weight "
                    "to suppress global prevalence shifts.",
                    len(_mismatch_classes), _mismatch_classes,
                )

        passthrough_mean = float(np.mean(baseline_scores["passthrough"]))
        if passthrough_mean <= best_score:
            logger.warning(
                "Stage 2: passthrough beats best fusion α=%.4g "
                "(passthrough Brier=%.4f vs fusion Brier=%.4f). "
                "H2 not supported — skipping learned context head.",
                best_hp, passthrough_mean, best_score,
            )
            return _PassthroughSentinel(), None
    else:
        logger.warning(
            "Stage 2: only %d patient(s) in training set — skipping inner HP search, "
            "using α=%.4g.", len(unique_patients), alpha_options[0]
        )

    best_alpha = best_hp
    logger.info(
        "Stage 2: selected α=%.4g (inner Brier=%.4f). "
        "Retraining on all %d items.",
        best_alpha, best_score, len(Z_tr),
    )

    # Collect LOPO OOF predictions with best_alpha for Stage 2 post-hoc calibration.
    oof_preds_s2 = np.full_like(y_int_tr, np.nan, dtype=np.float32)
    if len(unique_patients) >= 2:
        for tr_ix, val_ix in lopo_splits(patient_ids_tr):
            m_oof = _train_one_model(
                Z_tr[tr_ix], y_int_tr[tr_ix],
                y_survey_tr[tr_ix], y_cf_tr[tr_ix],
                alpha=best_alpha, lr=lr,
                num_epochs=num_epochs, patience=patience,
                weight_cap=weight_cap, device=device, cfg=cfg,
                model_cls=model_cls, model_kwargs=model_kwargs,
                Z_val=Z_tr[val_ix], y_val=y_int_tr[val_ix],
            )
            m_oof.eval()
            with torch.no_grad():
                oof_preds_s2[val_ix] = torch.sigmoid(
                    m_oof(torch.tensor(Z_tr[val_ix], dtype=torch.float32, device=device))
                ).cpu().numpy()

    from ..shared.fit_calibrators import fit_calibrators as _fit_cals
    s2_calibrators = None
    valid_oof = ~np.isnan(oof_preds_s2).all(axis=1)
    if valid_oof.sum() >= int(cfg.get("calibration_min_samples", 10)):
        s2_calibrators = _fit_cals(
            oof_preds_s2[valid_oof], y_int_tr[valid_oof], class_list, cfg
        )
    else:
        logger.warning(
            "Stage 2 calibration skipped: only %d valid OOF rows (< %d).",
            int(valid_oof.sum()), int(cfg.get("calibration_min_samples", 10)),
        )

    final_model = _train_one_model(
        Z_tr, y_int_tr, y_survey_tr, y_cf_tr,
        alpha=best_alpha, lr=lr,
        num_epochs=num_epochs, patience=patience,
        weight_cap=weight_cap, device=device,
        cfg=cfg,
        model_cls=model_cls, model_kwargs=model_kwargs,
        # No Z_val/y_val: final retraining on all data, ridge handles regularisation.
    )
    return final_model, s2_calibrators


def apply_stage2_fusion(
    artifact: dict,
    Z: np.ndarray,        # (n, fusion_dim)
    y_cf: np.ndarray | None = None,  # (n, n_classes) Stage-1 predictions for passthrough
) -> np.ndarray:          # (n, n_classes) float32, clipped to [0, 1]
    """Apply fitted fusion model at inference time.

    ``artifact`` must be a dict with keys ``"model"``, ``"use_cf_passthrough"``,
    ``"fusion_strategy"``, and ``"r"``.  When ``use_cf_passthrough`` is True,
    ``y_cf`` is prepended to ``Z`` before forwarding through the linear head.
    """
    model = artifact["model"]
    if isinstance(model, _PassthroughSentinel):
        if y_cf is None:
            raise ValueError("Passthrough sentinel selected but y_cf was not provided.")
        return np.clip(y_cf, 0.0, 1.0).astype(np.float32)
    if artifact.get("use_cf_passthrough") and y_cf is not None:
        Z = np.concatenate([y_cf.astype(np.float32), Z], axis=1)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        Z_t = torch.tensor(Z, dtype=torch.float32, device=device)
        out = torch.sigmoid(model(Z_t)).cpu().numpy()
    s2_calibrators = artifact.get("calibrators")
    s2_class_list  = artifact.get("class_list")
    if s2_calibrators is not None and s2_class_list is not None:
        from ..shared.apply_calibrators import apply_calibrators as _apply_cal
        out = _apply_cal(s2_calibrators, out, s2_class_list)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def project_stage2_embeddings(
    artifact: dict,
    X_items: np.ndarray,
    context_vectors: dict,
) -> tuple[np.ndarray, dict]:
    """Apply the artifact's stored PCA (if any) to item embeddings and context vectors.

    Returns ``(X_projected, ctx_projected)``.  If the artifact has no PCA,
    the inputs are returned unchanged.
    """
    pca = artifact.get("pca")
    if pca is None:
        return X_items, context_vectors
    X_proj = pca.transform(X_items)
    ctx_proj = {
        pid: pca.transform(cv.reshape(1, -1))[0]
        for pid, cv in context_vectors.items()
    }
    return X_proj, ctx_proj


def _build_patient_id_baseline_features(
    X_items: np.ndarray,
    patient_ids: np.ndarray,
    reference_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Build [e_i ‖ one-hot(patient)] features for the patient-ID baseline.

    X_items must be raw item embeddings (not fusion features Z).
    one-hot encodes over reference_ids (the training-patient set for this fold).
    Unseen-patient rows receive an all-zero one-hot block.
    """
    unique_ids = np.unique(reference_ids if reference_ids is not None else patient_ids)
    id_to_idx = {int(float(pid)): idx for idx, pid in enumerate(unique_ids)}
    onehot = np.zeros((len(patient_ids), len(unique_ids)), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        key = int(float(pid))
        if key in id_to_idx:
            onehot[i, id_to_idx[key]] = 1.0
    return np.concatenate([X_items, onehot], axis=1)


def fit_all_stage2_architectures_fold(
    X_items_tr: np.ndarray,
    patient_ids_tr: np.ndarray,
    context_vectors: dict,
    y_int_tr: np.ndarray,
    y_survey_tr: np.ndarray,
    y_cf_tr: np.ndarray,
    class_list: list,
    cfg: dict,
    r: int = 8,
) -> dict:
    """Fit all alternative Stage 2 fusion architectures under their own inner LOPO.

    Trains '2d', '3d', 'lowrank_bilinear', and 'patient_id' under the full
    nested patient-level CV protocol (same outer fold, independent inner LOPO
    alpha search per architecture).  Passthrough/stage1_only require no
    training and are assembled by the caller from Stage 1 predictions.

    Returns {arch_name: artifact_dict} where each artifact_dict is compatible
    with apply_stage2_fusion (has at least the "model" key).
    """
    from .stage2_context import build_fusion_matrix as _bfm
    from ..shared.lopo_cv import lopo_splits as _lopo

    lr          = float(cfg.get("stage2_lr", 0.01))
    num_epochs  = int(cfg.get("stage2_num_epochs", 200))
    patience    = int(cfg.get("stage2_early_stopping_patience", 20))
    weight_cap  = float(cfg.get("weight_cap", 50.0))
    alpha_opts  = list(cfg.get("stage2_alpha_options", [0.01, 0.1, 1.0, 10.0]))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_patients = np.unique(patient_ids_tr)
    inner_splits    = list(_lopo(patient_ids_tr)) if len(unique_patients) >= 2 else []

    def _select_and_fit(Z_inner: np.ndarray, model_cls, model_kwargs: dict):
        """Run inner LOPO alpha search then retrain on full data."""
        best_alpha = float(alpha_opts[0])
        best_score = float("inf")
        for alpha in alpha_opts:
            fold_scores = []
            for tr_ix, val_ix in inner_splits:
                m = _train_one_model(
                    Z_inner[tr_ix], y_int_tr[tr_ix],
                    y_survey_tr[tr_ix], y_cf_tr[tr_ix],
                    alpha=alpha, lr=lr, num_epochs=num_epochs, patience=patience,
                    weight_cap=weight_cap, device=device, cfg=cfg,
                    model_cls=model_cls, model_kwargs=model_kwargs,
                    Z_val=Z_inner[val_ix], y_val=y_int_tr[val_ix],
                )
                m.eval()
                with torch.no_grad():
                    preds = torch.sigmoid(
                        m(torch.tensor(Z_inner[val_ix], dtype=torch.float32, device=device))
                    ).cpu().numpy()
                fold_scores.append(_brier_score(preds, y_int_tr[val_ix]))
            mean_s = float(np.mean(fold_scores)) if fold_scores else float("inf")
            if mean_s < best_score:
                best_score = mean_s
                best_alpha = float(alpha)
        logger.debug("Alt arch inner α=%.4g, Brier=%.4f", best_alpha, best_score)
        return _train_one_model(
            Z_inner, y_int_tr, y_survey_tr, y_cf_tr,
            alpha=best_alpha, lr=lr, num_epochs=num_epochs, patience=patience,
            weight_cap=weight_cap, device=device, cfg=cfg,
            model_cls=model_cls, model_kwargs=model_kwargs,
        )

    models: dict = {}

    for strategy, m_cls, m_kw in [
        ("2d",              Stage2FusionModel,          {}),
        ("3d",              Stage2FusionModel,          {}),
        ("lowrank_bilinear", LowRankBilinearFusionModel, {"r": r}),
    ]:
        Z = _bfm(X_items_tr, patient_ids_tr, context_vectors, fusion_strategy=strategy, r=r)
        models[strategy] = {"model": _select_and_fit(Z, m_cls, m_kw)}
        logger.info("Alt Stage 2 arch '%s' fitted.", strategy)

    # Patient-ID baseline: [e_i ‖ one-hot(patient)] — raw embeddings, not fusion features.
    X_pid = _build_patient_id_baseline_features(X_items_tr, patient_ids_tr, reference_ids=patient_ids_tr)
    models["patient_id"] = {"model": _select_and_fit(X_pid, Stage2FusionModel, {})}
    logger.info("Patient-ID baseline fitted (dim=%d).", X_pid.shape[1])

    return models
