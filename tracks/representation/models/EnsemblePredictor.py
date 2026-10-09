from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from tracks.representation.models.MultiLabelModel import MultiLabelModel

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Container for portable ensemble artifacts."""

    def __init__(
        self,
        models,
        preprocessors,
        calibrators,
        thresholds,
        device="cpu",
        class_list=None,
        model_arch=None,
        stage2=None,
        stage2_context_vectors=None,
    ):
        self.models = models
        self.preprocessors = preprocessors
        self.calibrators = calibrators
        self.thresholds = thresholds
        self.device = device
        self.class_list = list(class_list) if class_list is not None else None
        # model_arch: {"hidden_dims": [], "dropout": 0.3, "activation": "gelu"}
        # Empty dict means logistic regression.
        self.model_arch = model_arch or {}
        # stage2: list of per-fold PyTorch fusion head models (Stage2FusionModel or
        # LowRankBilinearFusionModel), one per outer LOPO fold (or None when that
        # fold's Stage 2 training was skipped or failed).  Each fold's model is
        # applied to fusion features before averaging, so the held-out patient for
        # fold F never influenced fold F's Stage 2 head.  None entries are skipped.
        self.stage2 = stage2
        self.stage2_context_vectors = stage2_context_vectors
        self._live_models: list[MultiLabelModel] | None = None

    def to_bundle_dict(
        self,
        *,
        cfg_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize the ensemble to a portable dict suitable for joblib persistence.

        Args:
            cfg_snapshot: Optional config dict to embed as a reproducibility record.
            metadata: Optional free-form metadata dict (e.g. run_id, timestamp).

        Returns:
            dict: Bundle payload with keys ``bundle_format``, ``models``,
                ``preprocessors``, ``calibrators``, ``thresholds``, ``device``,
                ``class_list``, ``model_arch``, ``config_snapshot``, ``metadata``.
        """
        return {
            "bundle_format": "pdm_ensemble_v1",
            "models": self.models,
            "preprocessors": self.preprocessors,
            "calibrators": self.calibrators,
            "thresholds": self.thresholds,
            "device": str(self.device),
            "class_list": self.class_list,
            "model_arch": self.model_arch,
            "stage2": self.stage2,
            "stage2_context_vectors": self.stage2_context_vectors,
            "config_snapshot": cfg_snapshot or {},
            "metadata": metadata or {},
        }

    @classmethod
    def from_bundle_dict(
        cls, payload: dict[str, Any], *, device: str | None = None
    ) -> "EnsemblePredictor":
        """Reconstruct an EnsemblePredictor from a bundle dict.

        Required bundle keys: ``models``, ``preprocessors``, ``calibrators``,
        ``thresholds``. Optional keys: ``device``, ``class_list``, ``model_arch``.
        The ``bundle_format`` key is not checked here; callers that care about
        the format string should validate it before calling this method (see
        :meth:`load_bundle`).

        Args:
            payload: Dict produced by :meth:`to_bundle_dict`.
            device: Override the device stored in the bundle (e.g. ``"cpu"`` or
                ``"cuda"``). If ``None``, the value from the bundle is used.

        Returns:
            EnsemblePredictor: Fully initialized predictor.
        """
        return cls(
            models=payload["models"],
            preprocessors=payload["preprocessors"],
            calibrators=payload["calibrators"],
            thresholds=payload["thresholds"],
            device=device or payload.get("device", "cpu"),
            class_list=payload.get("class_list"),
            model_arch=payload.get("model_arch"),
            stage2=payload.get("stage2"),
            stage2_context_vectors=payload.get("stage2_context_vectors"),
        )

    def save_bundle(
        self,
        bundle_path: str | Path,
        *,
        cfg_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize the ensemble to disk via joblib.

        Creates parent directories as needed. The saved file can be reloaded
        with :meth:`load_bundle` or the top-level :func:`load_ensemble_predictor`.

        Args:
            bundle_path: Destination file path (e.g. ``run/model/ensemble.joblib``).
            cfg_snapshot: Optional config snapshot forwarded to :meth:`to_bundle_dict`.
            metadata: Optional metadata dict forwarded to :meth:`to_bundle_dict`.

        Returns:
            Path: Resolved path to the saved file.
        """
        bundle_path = Path(bundle_path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_bundle_dict(cfg_snapshot=cfg_snapshot, metadata=metadata)
        joblib.dump(payload, bundle_path)
        return bundle_path

    @staticmethod
    def load_bundle(
        bundle_path: str | Path, *, device: str | None = None
    ) -> "EnsemblePredictor":
        """Load an ensemble from a joblib file created by :meth:`save_bundle`.

        The file must contain a dict with ``bundle_format == "pdm_ensemble_v1"``
        and the required keys ``models``, ``preprocessors``, ``calibrators``,
        ``thresholds``.

        Args:
            bundle_path: Path to the ``.joblib`` bundle file.
            device: Device override forwarded to :meth:`from_bundle_dict`.

        Returns:
            EnsemblePredictor: Loaded predictor.

        Raises:
            ValueError: If the file does not contain a recognised bundle format.
        """
        payload = joblib.load(bundle_path)
        if not isinstance(payload, dict) or payload.get("bundle_format") != "pdm_ensemble_v1":
            raise ValueError(f"Unsupported ensemble bundle format in {bundle_path}")
        return EnsemblePredictor.from_bundle_dict(payload, device=device)

    def materialize(self) -> None:
        """Instantiate and warm all ensemble member models, storing them for reuse.

        Idempotent — subsequent calls return immediately without re-allocating.
        After this call, ``predict_proba`` skips per-call model reconstruction.
        Call :meth:`release` to free the live models from memory.
        """
        if self._live_models is not None:
            return
        arch        = self.model_arch
        hidden_dims = list(arch.get("hidden_dims", []))
        dropout     = float(arch.get("dropout", 0.3))
        activation  = str(arch.get("activation", "gelu"))
        live = []
        for state_dict in self.models:
            # Infer input/output dims from state dict.
            # Logistic regression bundles (new and legacy) store "linear.weight".
            # MLP bundles store "net.0.weight" for the first layer.
            if "linear.weight" in state_dict:
                n_in   = state_dict["linear.weight"].shape[1]
                n_out  = state_dict["linear.weight"].shape[0]
                h_dims = []   # logistic regression regardless of arch setting
            else:
                first_w = next(k for k in state_dict if k.endswith(".weight"))
                last_w  = next(
                    k for k in reversed(list(state_dict)) if k.endswith(".weight")
                )
                n_in   = state_dict[first_w].shape[1]
                n_out  = state_dict[last_w].shape[0]
                h_dims = hidden_dims
            model = MultiLabelModel(
                n_in, n_out,
                hidden_dims=h_dims,
                dropout=dropout,
                activation=activation,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            live.append(model)
        self._live_models = live
        logger.debug(f"Materialized {len(live)} ensemble member(s) on {self.device}.")

    def release(self) -> None:
        """Free all live model instances from memory.

        After this call, the next ``predict_proba`` call will re-materialize.
        On CUDA devices, also flushes the device cache.
        """
        self._live_models = None
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def predict_proba(self, X, *, patient_ids=None, **kwargs):
        """Run calibrated ensemble inference and return averaged probabilities.

        Calls :meth:`materialize` on the first invocation to build live model
        instances; subsequent calls reuse them with no reconstruction overhead.
        Each ensemble member preprocesses ``X`` independently, runs forward
        inference, applies its per-class calibrators, and the results are
        averaged across all members.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``. Must be
                compatible with the fitted preprocessors stored in the bundle.

        Returns:
            np.ndarray: Float32 array of shape ``(n_samples, n_classes)``.
                Column order matches ``self.class_list`` (i.e. ``label_names``
                in the order they were passed at fit time).
        """
        _t0 = time.perf_counter()
        _n_samples = int(np.asarray(X).shape[0])
        logger.debug(
            f"Ensemble inference: {_n_samples} sample(s), "
            f"{len(self.models)} member(s)."
        )
        self.materialize()

        # Warn once if stage2 list is a mix of fitted models and None fallbacks.
        # In that case the final average blends Stage 2 and Stage 1 fold outputs
        # and should not be interpreted as a pure Stage 2 ensemble result.
        if self.stage2 is not None:
            _s2_fitted = sum(1 for m in self.stage2 if m is not None)
            _s2_total  = len(self.stage2)
            if 0 < _s2_fitted < _s2_total:
                logger.warning(
                    "predict_proba: Stage 2 model present for %d of %d fold(s). "
                    "The returned probabilities average Stage 2 outputs (folds "
                    "with a model) and Stage 1 outputs (folds without one). "
                    "This is a degraded hybrid, not a pure Stage 2 estimate.",
                    _s2_fitted, _s2_total,
                )

        all_probs = []
        _cls = self.class_list or []
        for i, model in enumerate(self._live_models):
            X_trans = self.preprocessors[i].transform(X)

            with torch.no_grad():
                # from_numpy shares CPU memory (zero-copy), then .to() does
                # the single necessary copy to device.
                X_t   = torch.from_numpy(np.asarray(X_trans, dtype=np.float32)).to(self.device)
                probs = torch.sigmoid(model(X_t)).cpu().numpy()

            fold_class_list = list(self.calibrators[i].keys())
            calibrated = np.zeros_like(probs)
            for c_idx, cls in enumerate(fold_class_list):
                cal   = self.calibrators[i][cls]
                col_p = probs[:, c_idx].reshape(-1, 1)
                if cal is None:
                    calibrated[:, c_idx] = probs[:, c_idx]
                elif hasattr(cal, "predict_proba"):
                    calibrated[:, c_idx] = cal.predict_proba(col_p)[:, 1]
                else:
                    calibrated[:, c_idx] = np.asarray(
                        cal.predict(col_p.reshape(-1))
                    ).reshape(-1)

            # Apply this fold's Stage 2 before accumulating.
            fold_stage2 = (
                self.stage2[i]
                if self.stage2 is not None and i < len(self.stage2)
                else None
            )
            if fold_stage2 is not None:
                from tracks.representation.training.stage2.fit_stage2_fusion import (
                    apply_stage2_fusion as _apply_fusion,
                    project_stage2_embeddings as _proj_s2,
                )
                if patient_ids is None:
                    raise ValueError(
                        "patient_ids must be provided to predict_proba when "
                        "Stage 2 fusion models are present."
                    )
                from tracks.representation.training.stage2.stage2_context import build_fusion_matrix
                fusion_strategy = fold_stage2["fusion_strategy"]
                r = fold_stage2["r"]
                _X_inf, _ctx_inf = _proj_s2(fold_stage2, X, self.stage2_context_vectors or {})
                _Z = build_fusion_matrix(
                    _X_inf, patient_ids, _ctx_inf,
                    fusion_strategy=fusion_strategy, r=r
                )
                calibrated = _apply_fusion(fold_stage2, _Z, y_cf=calibrated)

            all_probs.append(calibrated)

        result = np.mean(np.stack(all_probs), axis=0)

        logger.debug(
            f"Ensemble inference complete "
            f"({time.perf_counter() - _t0:.2f}s)."
        )
        return result


def load_ensemble_predictor(
    path: str | Path,
    *,
    device: str | None = None,
    use_registry: bool = True,
    cfg: dict[str, Any] | None = None,
) -> EnsemblePredictor:
    """Load an ensemble artifact from disk, handling both legacy and bundle formats.

    Accepts two on-disk representations:
    - A joblib file containing an :class:`EnsemblePredictor` instance directly
      (legacy format written before ``save_bundle`` existed).
    - A joblib file containing a ``pdm_ensemble_v1`` bundle dict
      (written by :meth:`EnsemblePredictor.save_bundle`).

    When ``use_registry=True`` (the default), the module-level
    :class:`~tracks.representation.models.ModelRegistry.ModelRegistry` singleton is consulted before
    touching disk. A cache hit returns the live predictor immediately; a miss
    loads from disk and stores the result for future calls. The cache key
    includes the file's mtime so a re-saved bundle is never served stale.

    Pass ``use_registry=False`` to bypass the registry entirely — useful during
    training where bundles are freshly written and never shared across callers.

    When ``cfg`` is provided, the ``inference`` sub-dict overrides the explicit
    keyword arguments: ``inference.persistent_models`` controls ``use_registry``
    and ``inference.eager_materialize`` controls whether :meth:`materialize` is
    called immediately after loading (pre-warming models before first inference).

    Args:
        path: Path to the joblib artifact file.
        device: Optional device override (e.g. ``"cpu"``, ``"cuda"``). Applied
            in-place for legacy instances; forwarded to
            :meth:`EnsemblePredictor.from_bundle_dict` for bundle dicts.
        use_registry: Whether to consult and populate the module-level registry.
            Overridden by ``cfg["inference"]["persistent_models"]`` when ``cfg``
            is provided.
        cfg: Optional resolved config dict (as returned by ``load_config`` /
            ``load_prerequisites``). When supplied, ``inference.*`` keys take
            precedence over the explicit keyword arguments above.

    Returns:
        EnsemblePredictor: Loaded predictor ready for inference.

    Raises:
        ValueError: If the file contains an unrecognised artifact type.
    """
    # Local import to avoid a circular dependency: ModelRegistry imports
    # EnsemblePredictor under TYPE_CHECKING; this function is only called at
    # runtime, so the deferred import is safe.
    from tracks.representation.models.ModelRegistry import get_registry

    infer_cfg: dict = {}
    if cfg is not None:
        infer_cfg = cfg.get("inference") or {}
        use_registry = bool(infer_cfg.get("persistent_models", use_registry))
    eager_materialize: bool = bool(infer_cfg.get("eager_materialize", False))

    candidate = Path(path)

    if use_registry:
        cached = get_registry().get(candidate, device)
        if cached is not None:
            logger.debug(f"Registry hit: {candidate.name}")
            return cached

    loaded = joblib.load(candidate)
    if not isinstance(loaded, dict) or loaded.get("bundle_format") != "pdm_ensemble_v1":
        raise ValueError(f"Unsupported ensemble artifact at {candidate}")
    predictor = EnsemblePredictor.from_bundle_dict(loaded, device=device)

    if eager_materialize:
        predictor.materialize()

    if use_registry:
        get_registry().put(candidate, device, predictor)

    return predictor


def write_ensemble_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write an ensemble manifest JSON file to disk.

    Creates parent directories as needed. The file is always overwritten.

    Args:
        path: Destination file path for the JSON manifest.
        payload: Dict to serialise. Keys are sorted alphabetically in the
            output for deterministic diffs.

    Returns:
        Path: Resolved path to the written file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
