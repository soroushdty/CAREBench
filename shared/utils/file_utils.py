"""Shared file-system helpers used across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union


JsonPath = Union[str, Path]


def ensure_parent_dir(path: JsonPath) -> Path:
    """Create the parent directory of ``path`` if it does not exist, then return the path.

    Idempotent: calling this function multiple times for the same path is safe.
    Intermediate directories are created as needed (``parents=True``).

    Args:
        path: Target file path whose parent directory should be guaranteed to exist.

    Returns:
        Path: The original ``path`` resolved to a :class:`pathlib.Path`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_csv_file(
    frame: Any,
    output_path: JsonPath,
    *,
    index: bool = False,
) -> None:
    """Write a DataFrame to a CSV file, overwriting any existing content.

    Parent directories are created via :func:`ensure_parent_dir`. The file is
    always overwritten — this function does **not** append.

    Args:
        frame: ``pd.DataFrame`` (or any object with a ``to_csv`` method).
        output_path: Destination file path.
        index: Whether to write the DataFrame row index. Defaults to ``False``.
    """
    save_path = ensure_parent_dir(output_path)
    frame.to_csv(save_path, index=index)


def remove_file_if_exists(path: JsonPath) -> bool:
    """Delete a file if it exists; do nothing if it does not.

    Args:
        path: Path to the file to remove.

    Returns:
        bool: ``True`` if the file existed and was deleted, ``False`` otherwise.
    """
    target = Path(path)
    if target.exists() and target.is_file():
        target.unlink()
        return True
    return False


def write_text_file(
    text: str,
    output_path: JsonPath,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write a string to a file, overwriting any existing content.

    Parent directories are created via :func:`ensure_parent_dir`.

    Args:
        text: String content to write.
        output_path: Destination file path.
        encoding: File encoding. Defaults to ``"utf-8"``.
    """
    save_path = ensure_parent_dir(output_path)
    save_path.write_text(text, encoding=encoding)
