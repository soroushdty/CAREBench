import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seeds(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Set deterministic seeds and backend flags for reproducible execution.

    ``deterministic_algorithms`` should only be ``True`` once at process
    startup.  Passing ``True`` inside threaded worker calls is not thread-safe
    and can raise a ``RuntimeError`` if another thread is mid-operation.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    logger.debug(
        "Seeds set: seed=%d, deterministic_algorithms=%s.", seed, deterministic_algorithms
    )
