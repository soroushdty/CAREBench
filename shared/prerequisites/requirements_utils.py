from __future__ import annotations

import importlib.util
import logging
import re
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


_PACKAGE_TO_MODULE_OVERRIDES = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
}


class ParsedRequirement(NamedTuple):
    name: str
    pinned_version: str | None
    raw: str


def _iter_requirements(requirements_path: Path) -> list[str]:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    reqs: list[str] = []
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
            logger.debug("Package '%s' is required.", line)

    return reqs


def _parse_requirement(requirement: str) -> ParsedRequirement:
    parts = requirement.split("==", maxsplit=1)
    if len(parts) == 2:
        return ParsedRequirement(name=parts[0].strip(), pinned_version=parts[1].strip(), raw=requirement)

    base = re.split(r"[<>=!~]", requirement, maxsplit=1)[0].strip()
    return ParsedRequirement(name=base, pinned_version=None, raw=requirement)


def _requirement_to_module_name(package_name: str) -> str:
    lowered = package_name.lower()
    if lowered in _PACKAGE_TO_MODULE_OVERRIDES:
        return _PACKAGE_TO_MODULE_OVERRIDES[lowered]
    return package_name.replace("-", "_")


def _is_module_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _is_pinned_requirement(requirement: str) -> bool:
    return "==" in requirement


def _installed_version(package_name: str) -> str | None:
    candidates = [package_name]
    lowered = package_name.lower()
    if lowered not in candidates:
        candidates.append(lowered)
    normalized = package_name.replace("_", "-")
    if normalized not in candidates:
        candidates.append(normalized)

    for candidate in candidates:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def _matches_pinned_version(installed: str, required: str) -> bool:
    # Exact match first.
    if installed == required:
        return True

    # Treat local build suffixes as equivalent when requirement pin has no suffix.
    # Example: required 2.10.0 should accept installed 2.10.0+cu128.
    if "+" not in required:
        installed_base = installed.split("+", maxsplit=1)[0]
        return installed_base == required

    return False


def requirements_utils(requirements_path: str | Path, *, enforce_pins: bool = True) -> list[str]:
    requirements_path = Path(requirements_path)
    reqs = _iter_requirements(requirements_path)

    if enforce_pins:
        unpinned = [req for req in reqs if not _is_pinned_requirement(req)]
        if unpinned:
            raise ValueError(
                "Unpinned requirements are not allowed for reproducible runs. "
                f"Found: {', '.join(unpinned)}"
            )

    missing: list[str] = []
    mismatched: list[str] = []

    for raw_req in reqs:
        parsed = _parse_requirement(raw_req)
        module_name = _requirement_to_module_name(parsed.name)

        if not _is_module_importable(module_name):
            logger.info("Module missing for '%s' (import: %s)", raw_req, module_name)
            missing.append(raw_req)
            continue

        if parsed.pinned_version is not None:
            installed = _installed_version(parsed.name)
            if installed is None:
                missing.append(raw_req)
                continue
            if not _matches_pinned_version(installed, parsed.pinned_version):
                mismatched.append(f"{parsed.name}=={installed} (required {parsed.pinned_version})")

    if missing:
        raise ModuleNotFoundError(
            "Missing required dependencies in the active environment: "
            f"{', '.join(missing)}. "
            "Install dependencies before running the pipeline (for example: "
            "`python -m pip install -r <active requirements profile file>`)."
        )

    if mismatched:
        raise RuntimeError(
            "Installed dependency versions do not match pinned requirements: "
            + ", ".join(mismatched)
        )

    return []
