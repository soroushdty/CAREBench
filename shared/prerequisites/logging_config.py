from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path


def _resolve_log_level(log_level: str | int | None) -> int:
    """Resolve a flexible log-level specification to a ``logging`` integer.

    Precedence (first match wins):

    1. ``None`` → ``logging.INFO``.
    2. ``int`` → used as-is.
    3. Numeric string (e.g. ``"20"``) → converted to ``int``.
    4. Level name string (e.g. ``"DEBUG"``, ``"WARNING"``) → resolved via
       ``logging.getLevelName``.

    Args:
        log_level: Level as ``None``, an integer, or a string name/number.

    Returns:
        int: Numeric log level suitable for ``logging.basicConfig``.

    Raises:
        ValueError: If the string does not map to a known log level.
    """
    if log_level is None:
        return logging.INFO
    if isinstance(log_level, int):
        return log_level
    name = str(log_level).strip().upper()
    if name.isdigit():
        return int(name)
    numeric_level = logging.getLevelName(name)
    if isinstance(numeric_level, int):
        return numeric_level
    raise ValueError(f"Invalid log level: {log_level}")


def logging_config(log_file_path: str | Path | None, log_level: str | int | None = None) -> None:
    """Configure the root logger with a consistent format and optional file handler.

    Uses ``logging.basicConfig`` with ``force=True``, which **removes all
    existing handlers** before applying the new configuration. Calling this
    function more than once in a process will therefore discard any handlers
    added between calls — including those set up by third-party libraries.

    When ``log_file_path`` is provided, a file handler writing to that path is
    installed (mode ``"w"`` — file is truncated on each call). When
    ``log_file_path`` is ``None``, output goes to ``sys.stdout``.

    Format string: ``"%(asctime)s  %(levelname)s:  %(message)s"``
    Date format: ``"%H:%M:%S  %m-%d-%y"``

    Args:
        log_file_path: Destination log file. Parent directories are created if
            they do not exist. Pass ``None`` to log to stdout instead.
        log_level: Log level forwarded to :func:`_resolve_log_level`. Accepts
            ``None`` (defaults to ``INFO``), an integer, or a level-name
            string such as ``"DEBUG"``.
    """
    resolved_level = _resolve_log_level(log_level)

    config_kwargs = {
        "level": resolved_level,
        "datefmt": "%H:%M:%S  %m-%d-%y",
        "format": "%(asctime)s  %(levelname)s:  %(message)s",
        "force": True,
    }

    if log_file_path is None:
        config_kwargs["stream"] = sys.stdout
    else:
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        config_kwargs["filename"] = log_file_path
        config_kwargs["filemode"] = "w"

    logging.basicConfig(**config_kwargs)

    logging.captureWarnings(True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    if log_file_path is not None:
        logging.info(f"START LOGGING AT: {log_file_path}")
    logging.info(f"LOG LEVEL: {logging.getLevelName(resolved_level)}")
    logging.info(f"RUNNING ON PYTHON VERSION {sys.version}")
