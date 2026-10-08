"""Path and remote-location helpers shared across the pipeline."""

from __future__ import annotations

from urllib.parse import urlparse


def is_remote_location(value: str) -> bool:
    """Return True if *value* is an HTTP/HTTPS URL."""
    return urlparse(value).scheme in {"http", "https"}
