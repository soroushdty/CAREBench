"""tracks.representation.models — Track 1 model classes.

Canonical path: tracks/representation/models/
"""

# ModelRegistry does not require torch and is always importable.
from tracks.representation.models.ModelRegistry import (  # noqa: F401
    ModelRegistry,
    configure_registry,
    get_registry,
)

# Torch-dependent model classes; guarded for torch-free environments.
try:
    from tracks.representation.models.ConstantCalibrator import ConstantCalibrator  # noqa: F401
    from tracks.representation.models.EnsemblePredictor import (  # noqa: F401
        EnsemblePredictor,
        load_ensemble_predictor,
        write_ensemble_manifest,
    )
    from tracks.representation.models.MultiLabelModel import MultiLabelModel  # noqa: F401
    from tracks.representation.models.Preprocessor import Preprocessor  # noqa: F401
except ModuleNotFoundError:
    pass

# __all__ lists only the non-conflicting public symbols (functions and classes
# whose names don't shadow a same-named submodule in the old models/ package).
# ModelRegistry, ConstantCalibrator, etc. share names with submodule files in
# models/, so they are excluded to avoid the submodule-shadowing issue in
# alias-resolution tests.
__all__ = [
    "configure_registry",
    "get_registry",
]
