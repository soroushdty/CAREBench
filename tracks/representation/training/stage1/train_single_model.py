import logging
import random
import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .compute_bce_loss import compute_weighted_bce_loss
from ..shared.set_seeds import set_seeds
from ..shared.threshold_tuning import threshold_tuning
from ..shared.soft_label_utils import masked_macro_metric
from shared.utils.array_utils import as_float32_array as _as_float32_array

from tracks.representation.models.MultiLabelModel import MultiLabelModel

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_single_model(
    X_train, Y_train, X_val, Y_val, cfg, hp, pos_weight_vector, *,
    seed=None,
    log_context: str = "",
    epoch_curve_callback=None,
):
    """Train one MultiLabelModel with early stopping.

    Args:
        X_train, Y_train:      Training data (numpy float32).
        X_val, Y_val:          Validation data (numpy float32).
        cfg:                   Pipeline config dict. Reads: global_seed,
                               hidden_dims, dropout, activation, use_torch_compile,
                               num_epochs, early_stopping_patience, objective.
        hp:                    Hyperparameter dict: lr, weight_decay,
                               batch_size, weight_cap.
        pos_weight_vector:     Per-class positive weights (numpy float32).
        seed:                  Optional integer seed. Overrides cfg['global_seed']
                               when provided. Pass unique values per thread to avoid
                               RNG collisions in threaded callers.
                               Defaults to None (use cfg['global_seed']).
        log_context:           Optional string prepended to DEBUG epoch log lines
                               to identify which training run the log belongs to
                               (e.g. "Fold 3 / 12 final"). Defaults to "" (no
                               prefix). Inner HP search calls should leave this
                               unset.
        epoch_curve_callback:  Optional callable invoked after every epoch with
                               signature (epoch, loss, val_score, patience, is_best).
                               The caller owns all I/O; this function stays
                               filesystem-free. Inner HP search calls should
                               leave this unset (None).

    Returns:
        (model, final_probs, best_thresholds): Trained model and val outputs.
    """
    effective_seed = cfg["global_seed"] if seed is None else int(seed)
    set_seeds(effective_seed)

    # Thread-local RNG objects so concurrent threaded callers don't race on
    # global np.random / random state. The torch.Generator provides independent
    # GPU random state for mini-batch permutations.
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(effective_seed)
    rng_np = np.random.default_rng(effective_seed)
    rng_py = random.Random(effective_seed)

    X_train           = _as_float32_array(X_train,           "X_train")
    Y_train           = _as_float32_array(Y_train,           "Y_train")
    X_val             = _as_float32_array(X_val,             "X_val")
    Y_val             = _as_float32_array(Y_val,             "Y_val")
    pos_weight_vector = _as_float32_array(pos_weight_vector, "pos_weight_vector")

    X_t   = torch.tensor(X_train,           dtype=torch.float32).to(DEVICE)
    Y_t   = torch.tensor(Y_train,           dtype=torch.float32).to(DEVICE)
    X_v   = torch.tensor(X_val,             dtype=torch.float32).to(DEVICE)
    pos_w = torch.tensor(pos_weight_vector, dtype=torch.float32).to(DEVICE)
    class_list = [f"class_{idx}" for idx in range(Y_val.shape[1])]

    # Build model from cfg architecture params.
    # hidden_dims=[] → logistic regression (linear.weight key, backward compat)
    # hidden_dims=[64] → 1-hidden-layer MLP, etc.
    model = MultiLabelModel(
        n_features  = X_train.shape[1],
        n_classes   = Y_train.shape[1],
        hidden_dims = list(cfg.get("hidden_dims", [])),
        dropout     = float(cfg.get("dropout", 0.3)),
        activation  = str(cfg.get("activation", "gelu")),
    ).to(DEVICE)

    # torch.compile — fuses forward pass + loss into fewer CUDA kernel launches.
    # Falls back silently to eager mode if unavailable (CPU, Windows, PyTorch < 2).
    if bool(cfg.get("use_torch_compile", True)) and DEVICE.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as exc:
            logger.warning(
                "torch.compile unavailable (%s); running in eager mode.", exc
            )

    optimizer = optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )

    # Clamp batch size to dataset length so values like 2048 become full-batch
    # on small inner-fold datasets, cutting kernel launches to 1/epoch.
    effective_batch = min(int(hp["batch_size"]), X_train.shape[0])
    full_batch      = effective_batch == X_train.shape[0]

    best_score      = -1.0
    best_state      = deepcopy(model.state_dict())
    best_thresholds = np.full(Y_val.shape[1], 0.5, dtype=np.float32)
    patience        = 0
    objective       = cfg.get("objective", "F1")
    num_epochs = int(cfg.get("num_epochs", cfg.get("epochs", 50)))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 5))

    if num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be a positive integer.")

    _log_interval = int(cfg.get("log_epoch_interval", 10))
    _log_prefix   = f"{log_context} — " if log_context else ""

    use_amp = bool(cfg.get("use_amp", True)) and DEVICE.type == "cuda"
    # Use torch.amp.GradScaler for PyTorch 2.4+ (torch.cuda.amp.GradScaler is deprecated)
    scaler  = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        if full_batch:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = compute_weighted_bce_loss(model(X_t), Y_t, pos_w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            perm = torch.randperm(X_train.shape[0], device=DEVICE, generator=generator)
            for start in range(0, X_train.shape[0], effective_batch):
                idx = perm[start : start + effective_batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=use_amp):
                    loss = compute_weighted_bce_loss(model(X_t[idx]), Y_t[idx], pos_w)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            v_probs = torch.sigmoid(model(X_v)).cpu().numpy()

        tuned_thresholds = threshold_tuning(v_probs, Y_val, class_list, cfg)

        # --- Configurable primary threshold for early stopping metric ---
        primary_threshold_mode = cfg.get("primary_threshold", "tau")
        if primary_threshold_mode == "tau":
            tau = float(cfg.get("tau", 0.5))
            thresholds = np.full(Y_val.shape[1], tau, dtype=np.float32)
        elif primary_threshold_mode in ("fixed_f1", "f1_sweep"):
            if primary_threshold_mode == "f1_sweep":
                import warnings
                warnings.warn(
                    "primary_threshold='f1_sweep' is deprecated; use 'fixed_f1' with "
                    "'fixed_f1_threshold' scalar instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _f1_sweep_val = cfg.get("f1_sweep")
            threshold = float(cfg.get(
                "fixed_f1_threshold",
                _f1_sweep_val[0] if isinstance(_f1_sweep_val, list) else 0.3,
            ))
            thresholds = np.full(Y_val.shape[1], threshold, dtype=np.float32)
        else:
            thresholds = tuned_thresholds

        val_score = masked_macro_metric(
            Y_val, v_probs, thresholds, cfg, objective
        )

        if val_score > best_score:
            best_score      = val_score
            best_state      = deepcopy(model.state_dict())
            best_thresholds = thresholds.copy()
            patience        = 0
        else:
            patience += 1
        # Logging: epoch heartbeat
        if _log_interval > 0 and (epoch + 1) % _log_interval == 0:
            _batch_note = " (last batch)" if not full_batch else ""
            logger.debug(
                f"{_log_prefix}Epoch {epoch + 1} / {num_epochs} — "
                f"loss: {loss.item():.4f}{_batch_note}, "
                f"val_{objective}: {val_score:.4f}, "
                f"patience: {patience} / {early_stopping_patience}."
            )
        if epoch_curve_callback is not None:
            _batch_loss = loss.item() if hasattr(loss, "item") else float(loss)
            epoch_curve_callback(
                epoch + 1, round(_batch_loss, 6),
                round(float(val_score), 6), patience, patience == 0,
            )
        if patience >= early_stopping_patience:
            logger.debug(
                f"{_log_prefix}Early stopping at epoch {epoch + 1} / "
                f"{num_epochs}. "
                f"Best val_{objective}: {best_score:.4f}."
            )
            break

    if best_state is None:
        logger.warning(
            "best_state was never updated (val_score was NaN every epoch). "
            "Loading final epoch state as fallback."
        )
        best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
        final_probs = torch.sigmoid(model(X_v)).cpu().numpy()
    return model, final_probs, best_thresholds
