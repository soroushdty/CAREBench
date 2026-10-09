"""HP candidate evaluation: inner-fold hyperparameter grid search worker."""

from __future__ import annotations

import logging

import numpy as np

from .train_single_model import train_single_model

logger = logging.getLogger(__name__)
from ..shared.soft_label_utils import macro_brier_score
from tracks.representation.models.Preprocessor import Preprocessor


def _run_hp_candidate(
    hp_t,
    X_tr, Y_tr,
    inner_splits,
    cfg,
    seed_base: int,
):
    """Evaluate one HP candidate across all inner LOPO folds.

    Designed to run in a ThreadPoolExecutor thread. Uses a unique seed derived
    from seed_base to avoid global RNG collisions with sibling threads.

    Args:
        hp_t:        1-tuple of (head_config,) where head_config is a hidden_dims list.
        X_tr, Y_tr:  Outer-fold training data (numpy float32).
        inner_splits: List of (train_ix, val_ix) index arrays from lopo_splits.
        cfg:         Pipeline config dict.
        seed_base:   Unique integer per HP candidate; added to global_seed.

    Returns:
        (hp dict, mean Brier score across inner folds, inner OOF predictions)
        The OOF array has shape Y_tr.shape; each row i_v is filled with the
        Stage 1 val probabilities produced for that inner fold's held-out rows.
    """
    head_config = hp_t[0]
    hp = {
        "lr":           float(cfg.get("lr", cfg.get("lr_options", [0.001])[0])),
        "weight_decay": float(cfg.get("weight_decay", 0.0001)),
        "batch_size":   int(cfg.get("batch_size", 2048)),
        "weight_cap":   float(cfg.get("weight_cap", cfg.get("weight_cap_options", [10.0])[0])),
        "selected_stage1_head_config": head_config,
    }
    logger.debug(
        "HP candidate: selected_stage1_head_config=%r — evaluating %d inner folds.",
        head_config, len(inner_splits),
    )
    scores = []
    inner_oof_preds = np.full(Y_tr.shape, -1.0, dtype=np.float32)
    base_seed = int(cfg["global_seed"]) + seed_base

    for split_idx, (i_tr, i_v) in enumerate(inner_splits):
        xt, yt = X_tr[i_tr], Y_tr[i_tr]
        xv, yv = X_tr[i_v],  Y_tr[i_v]

        prep    = Preprocessor(
            n_components=int(cfg.get("pca_n_components", 200)),
            whiten=bool(cfg.get("pca_whiten", True)),
        ).fit(xt)
        xt_p    = prep.transform(xt)
        xv_p    = prep.transform(xv)

        weight_mode = cfg.get("class_weight_mode", "cui")
        beta = float(cfg.get("class_weight_beta", 0.999))
        valid_counts = yt.shape[0]
        pos_mass     = yt.sum(axis=0)
        if weight_mode == "inverse_frequency":
            pos_w = np.clip(
                valid_counts / (2.0 * np.maximum(pos_mass, 1.0)),
                1.0,
                hp["weight_cap"],
            )
        else:  # Cui 2019 (default)
            pos_w = (1 - beta) / (1 - np.power(beta, np.maximum(pos_mass, 1)))
            pos_w = np.clip(pos_w, 1.0, hp["weight_cap"])

        # Pass a unique seed: global_seed + hp_candidate_offset + split_offset
        # This prevents threads from overwriting each other's global RNG state.
        worker_seed = base_seed + split_idx * 10

        inner_cfg = {**cfg, "hidden_dims": head_config}
        _, val_p, _ = train_single_model(
            xt_p, yt, xv_p, yv, inner_cfg, hp, pos_w, seed=worker_seed
        )
        inner_oof_preds[i_v] = val_p
        scores.append(macro_brier_score(yv, val_p))

    mean_score = float(np.mean(scores))
    logger.debug(
        "HP candidate: selected_stage1_head_config=%r — mean Brier=%.4f.",
        head_config, mean_score,
    )
    return hp, mean_score, inner_oof_preds
