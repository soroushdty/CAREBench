from .shared.lopo_cv import lopo_splits  # noqa: F401

__all__ = ["lopo_splits"]


def __getattr__(name):
    if name == "train_ensemble_pipeline":
        try:
            from .orchestrator.train_ensemble_pipeline import train_ensemble_pipeline
            return train_ensemble_pipeline
        except ModuleNotFoundError as exc:
            raise AttributeError(
                f"module {__name__!r} attribute 'train_ensemble_pipeline' requires "
                f"torch which is not installed: {exc}"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
