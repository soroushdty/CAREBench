# cls/Preprocessor.py  — full replacement

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Preprocessor:
    """GPU-accelerated StandardScaler + fixed-k PCA via torch.linalg.svd.

    Uses a fixed number of components (pca_n_components) rather than
    variance-based selection, ensuring every fold produces identically-shaped
    output regardless of which patient is held out.

    Optionally applies whitening (pca_whiten=True) to correct the anisotropy
    present in BERT-family embeddings. Set pca_whiten=False for models already
    trained with contrastive/isotropic objectives (e.g. Sentence Transformers,
    BGE, E5, MiniLM).

    fit() and transform() accept numpy arrays or torch tensors.
    transform() returns a float32 numpy array so all downstream code
    (calibration, threshold tuning, metrics) is unchanged.

    Pickle-safe: GPU tensors are serialised as CPU numpy arrays so that
    joblib.dump / joblib.load works correctly for EnsemblePredictor bundles.
    """

    def __init__(self, n_components: int = 200, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten
        self.n_components_: int | None = None
        # All stored as GPU tensors after fit()
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._components: torch.Tensor | None = None       # (k, n_features)
        self._singular_values: torch.Tensor | None = None  # (k,) — for whitening

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_gpu_f32(self, X) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(dtype=torch.float32, device=DEVICE)
        return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(DEVICE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "Preprocessor":
        Xt = self._to_gpu_f32(X)                           # (N, p)

        if not torch.isfinite(Xt).all():
            raise ValueError("X contains NaN/Inf before preprocessing.")
        if Xt.var() == 0.0:
            raise ValueError("Input X has zero variance before preprocessing.")

        # --- GPU StandardScaler ---
        self._mean = Xt.mean(dim=0)                        # (p,)
        self._std  = Xt.std(dim=0, unbiased=False).clamp(min=1e-8)
        Xs = (Xt - self._mean) / self._std                 # (N, p)

        # --- Truncated SVD (GPU) ---
        N, p = Xs.shape
        k = max(1, min(self.n_components, N, p))
        try:
            _, S, Vh = torch.linalg.svd(Xs, full_matrices=False)
        except RuntimeError:
            # Degenerate input: retry in float64 then downcast
            _, S, Vh = torch.linalg.svd(Xs.double(), full_matrices=False)
            S, Vh = S.float(), Vh.float()

        self._components      = Vh[:k]                     # (k, p)
        self._singular_values = S[:k]                      # (k,)
        self.n_components_    = k
        self._train_n         = N                          # stored for whitening scale

        logger.debug(
            "GPU PCA: n_components=%d, whiten=%s (N=%d, p=%d)",
            k, self.whiten, N, p,
        )
        return self

    def transform(self, X) -> np.ndarray:
        if self._components is None:
            raise RuntimeError("Preprocessor.transform() called before fit().")

        Xt  = self._to_gpu_f32(X)
        Xs  = (Xt - self._mean) / self._std                # (N, p)
        proj = Xs @ self._components.T                     # (N, k)

        if self.whiten:
            # Divide each component by its standard deviation so all output
            # dimensions have unit variance — corrects BERT anisotropy.
            # fit() standardizes with biased std (ddof=0), so S_i² = N·λ_i and
            # the correct whitening scale is S / sqrt(N), not S / sqrt(N-1).
            scale = self._singular_values / self._train_n ** 0.5
            proj  = proj / scale.clamp(min=1e-8)

        return proj.cpu().numpy()                          # CPU numpy for downstream

    # ------------------------------------------------------------------
    # Pickle support for joblib / EnsemblePredictor bundle serialisation
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_mean", "_std", "_components", "_singular_values"):
            t = state.get(key)
            if isinstance(t, torch.Tensor):
                state[key] = t.cpu().numpy()
        return state

    def __setstate__(self, state):
        for key in ("_mean", "_std", "_components", "_singular_values"):
            arr = state.get(key)
            if isinstance(arr, np.ndarray):
                state[key] = torch.from_numpy(arr).to(DEVICE)
        self.__dict__.update(state)
