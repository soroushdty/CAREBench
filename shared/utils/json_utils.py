"""JSON I/O helpers shared across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union
from urllib.parse import urlparse
from urllib.request import urlopen

from shared.utils.file_utils import ensure_parent_dir

JsonPath = Union[str, Path]


def load_json_from_source(
    source: JsonPath,
    *,
    allow_json_payload: bool = False,
    base_dir: JsonPath | None = None,
) -> Any:
    """Load JSON from a remote URL, a local file path, or an inline JSON string.

    Resolution order:

    1. If ``source`` looks like an ``http://`` or ``https://`` URL, fetch it
       with :func:`urllib.request.urlopen` and parse the response body.
    2. If ``source`` is a relative path and ``base_dir`` is given, resolve it
       relative to ``base_dir``; then try to open it as a local file.
    3. If ``allow_json_payload=True`` and the string starts with ``{`` or
       ``[``, parse it directly as a JSON literal.
    4. Otherwise raise :exc:`FileNotFoundError`.

    Args:
        source: A URL string, a file path (absolute or relative), or — when
            ``allow_json_payload`` is ``True`` — a raw JSON string.
        allow_json_payload: When ``True``, a string starting with ``{`` or
            ``[`` is accepted as an inline JSON literal after file-path lookup
            fails.
        base_dir: Base directory used to resolve relative paths.
            Ignored for absolute paths and URLs.

    Returns:
        Any: Parsed JSON value (dict, list, str, etc.).

    Raises:
        FileNotFoundError: If the source is a path that does not point to an
            existing file and ``allow_json_payload`` is ``False`` (or the
            string does not look like a JSON literal).
        urllib.error.URLError: If the remote URL cannot be fetched.
        json.JSONDecodeError: If the fetched or read content is not valid JSON.
    """
    text = str(source).strip()
    parsed = urlparse(text.replace("\\", "/"))
    if parsed.scheme in {"http", "https"}:
        with urlopen(text) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    candidate = Path(text)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / candidate
    if candidate.exists() and candidate.is_file():
        with candidate.open("r", encoding="utf-8") as f:
            return json.load(f)
    if allow_json_payload and (text.startswith("{") or text.startswith("[")):
        return json.loads(text)
    raise FileNotFoundError(f"JSON source does not point to an existing file: {candidate}")


def load_json_from_cfg(
    cfg: Mapping[str, Any],
    *,
    key: str = "DIR_JSON_MAP",
    base_dir: JsonPath | None = None,
) -> Any | None:
    """Load JSON from the path stored under a config key.

    Args:
        cfg: Config mapping to read from.
        key: Config key whose value is the JSON source path. Defaults to
            ``"DIR_JSON_MAP"``.
        base_dir: Forwarded to :func:`load_json_from_source` for relative-path
            resolution.

    Returns:
        Any | None: Parsed JSON value, or ``None`` if the key is absent or
            empty.
    """
    mapping_path = cfg.get(key)
    if mapping_path in (None, ""):
        return None
    return load_json_from_source(mapping_path, allow_json_payload=False, base_dir=base_dir)


def save_json_file(
    payload: Mapping[Any, Any] | list[Any] | dict[str, Any],
    output_path: JsonPath,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    stringify_keys: bool = False,
) -> None:
    """Serialise ``payload`` to a JSON file, creating parent directories as needed.

    Args:
        payload: JSON-serialisable object (dict, list, etc.).
        output_path: Destination file path. Parent directories are created via
            :func:`~shared.utils.file_utils.ensure_parent_dir`.
        ensure_ascii: Passed to :func:`json.dump`. Defaults to ``False``
            (non-ASCII characters are written as-is).
        indent: JSON indentation level. Defaults to ``2``.
        stringify_keys: When ``True`` and ``payload`` is a mapping, all keys
            are converted to strings before serialisation.
    """
    save_path = ensure_parent_dir(output_path)
    data_to_save = (
        {str(k): v for k, v in payload.items()}
        if stringify_keys and isinstance(payload, Mapping)
        else payload
    )
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=ensure_ascii, indent=indent)
