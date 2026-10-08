from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tracks.representation.models.EnsemblePredictor import EnsemblePredictor

# (resolved_absolute_path, mtime_ns, device) — device is included so that a
# CPU-loaded and CUDA-loaded instance of the same bundle are kept separately.
_CacheKey = tuple[str, int, str]


class ModelRegistry:
    """Thread-safe LRU cache for loaded EnsemblePredictor instances.

    Avoids redundant joblib deserialization when the same bundle is requested
    multiple times. Cache entries are keyed by (resolved path, mtime_ns, device)
    so that a re-saved bundle automatically invalidates its entry.

    Args:
        max_size: Maximum number of predictors to keep in memory simultaneously.
            When the cache is full, the least-recently-used entry is evicted and
            its :meth:`~EnsemblePredictor.release` method is called.
    """

    def __init__(self, max_size: int = 1) -> None:
        self._max_size = max_size
        # OrderedDict preserves insertion/touch order for LRU eviction.
        self._cache: OrderedDict[_CacheKey, Any] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_size(self) -> int:
        return self._max_size

    def _key(self, path: Path, device: str | None) -> _CacheKey:
        return (str(path.resolve()), path.stat().st_mtime_ns, device or "cpu")

    def get(self, path: str | Path, device: str | None) -> EnsemblePredictor | None:
        """Return the cached predictor for (path, mtime, device), or ``None`` on miss.

        A stale entry (mtime changed) produces a miss — the caller is responsible
        for loading from disk and calling :meth:`put` with the fresh predictor.
        """
        p = Path(path)
        if not p.exists():
            return None
        key = self._key(p, device)
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, path: str | Path, device: str | None, predictor: EnsemblePredictor) -> None:
        """Store a predictor; evicts the LRU entry (calling its ``release()``) if full.

        If a matching key already exists, it is replaced in-place and bumped to
        the most-recently-used position.
        """
        p = Path(path)
        key = self._key(p, device)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = predictor
                return
            self._cache[key] = predictor
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                _, evicted = self._cache.popitem(last=False)
                evicted.release()

    def invalidate(self, path: str | Path) -> None:
        """Remove all cached entries for ``path``, regardless of mtime or device.

        Calls ``release()`` on each evicted predictor.
        """
        resolved = str(Path(path).resolve())
        with self._lock:
            stale = [k for k in self._cache if k[0] == resolved]
            for key in stale:
                self._cache.pop(key).release()

    def resize(self, new_max: int) -> None:
        """Update the maximum cache size, evicting LRU entries if the cache is now over-full.

        Args:
            new_max: New maximum number of entries. Must be at least 1.

        Raises:
            ValueError: If ``new_max`` is less than 1.
        """
        if new_max < 1:
            raise ValueError(f"max_size must be at least 1, got {new_max}.")
        with self._lock:
            self._max_size = new_max
            while len(self._cache) > self._max_size:
                _, evicted = self._cache.popitem(last=False)
                evicted.release()

    def clear(self) -> None:
        """Empty the entire cache, calling ``release()`` on every entry."""
        with self._lock:
            for predictor in self._cache.values():
                predictor.release()
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Return the module-level singleton :class:`ModelRegistry`."""
    return _registry


def configure_registry(cfg: dict) -> None:
    """Apply the ``inference`` config block to the module-level registry singleton.

    Reads ``cfg["inference"]["model_cache_size"]`` and resizes the registry
    accordingly.  Safe to call multiple times; idempotent when the value has
    not changed.  Intended to be called once after the full config has been
    loaded and merged (i.e. after ``training_config.yaml`` is merged in).

    Args:
        cfg: Resolved config dict containing an optional ``inference`` sub-dict.
    """
    infer_cfg: dict = cfg.get("inference") or {}
    size = int(infer_cfg.get("model_cache_size", 1))
    _registry.resize(size)
