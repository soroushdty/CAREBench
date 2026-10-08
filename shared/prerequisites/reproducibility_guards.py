import logging
import os
import random
import time

import numpy as np
import requests
import torch

logger = logging.getLogger(__name__)

def enforce_reproducibility(seed: int = 42):
    """
    Set all known reproducibility flags and seeds for Python, NumPy, and PyTorch.
    Ensures deterministic algorithms and disables hash randomization.

    Note:
        PYTHONHASHSEED must be set as an environment variable *before* launching Python.
        Setting it at runtime has no effect. If not set, a warning is issued.
    """
    if os.environ.get("PYTHONHASHSEED") is None:
        logger.warning(
            "PYTHONHASHSEED was not set at interpreter launch. "
            "Set PYTHONHASHSEED=<seed> in your environment before starting Python for full reproducibility."
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For PyTorch >= 1.8
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

def fetch_remote_json_with_integrity(url: str, timeout: int = 10, max_retries: int = 3) -> dict:
    """
    Fetch a remote JSON file with timeout, retries, and basic integrity checks.
    Raises on failure or invalid content.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"Remote JSON at {url} is not a dict.")
            return data
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed to fetch {url}: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff (1s, 2s, 4s, …) to handle transient
                # network hiccups without hammering the remote server.
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to fetch remote JSON from {url} after {max_retries} attempts.") from e
