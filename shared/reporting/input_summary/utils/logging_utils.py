"""Logging utilities for summary module operations."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_summary_logger(
    summary_dir: Path,
    summary_log_path: str | Path | None = None,
) -> logging.Logger:
    """Set up a dedicated logger for summary operations.

    Creates a separate logger that writes detailed summary operations to
    summary_log.txt while high-level events are also propagated to the root logger.

    Args:
        summary_dir: Path to summary directory where log will be stored
        summary_log_path: Optional summary log file path from main config.
            Relative paths are resolved under summary_dir.

    Returns:
        A logger instance configured for summary operations
    """
    logger = logging.getLogger("shared.reporting.input_summary")
    logger.setLevel(logging.DEBUG)

    # Remove and close existing handlers to avoid duplicates and file locks.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Determine log file path.
    log_path_raw = summary_log_path or "summary_log.txt"
    log_path_str = str(log_path_raw).strip()

    if Path(log_path_str).is_absolute():
        log_path = Path(log_path_str)
    else:
        log_path = summary_dir / log_path_str

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler for summary log
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)s:  %(message)s",
        datefmt="%H:%M:%S  %m-%d-%Y",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate console output
    # but high-level events will be logged explicitly to root logger
    logger.propagate = False

    return logger


def log_high_level_event(summary_logger: logging.Logger, root_logger: logging.Logger | None, level: int, message: str) -> None:
    """Log a high-level summary event to both summary and root loggers.

    Args:
        summary_logger: The summary module logger
        root_logger: The root logger (if None, only summary logger is used)
        level: Logging level (e.g., logging.INFO)
        message: Message to log
    """
    summary_logger.log(level, message)
    if root_logger is not None:
        root_logger.log(level, f"[SUMMARY] {message}")
