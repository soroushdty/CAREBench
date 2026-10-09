"""
ZIP bundle creation for the analysis bundle pipeline.
"""

from __future__ import annotations

import glob
import logging
import os
import zipfile

logger = logging.getLogger(__name__)


def write_analysis_bundle(output_dir: str) -> str:
    """Create analysis_bundle.zip from all CSVs and the Markdown report.

    Parameters
    ----------
    output_dir:
        Directory containing the analysis output files.

    Returns
    -------
    str
        Absolute path to the created ZIP file.
    """
    output_dir = os.path.abspath(output_dir)
    zip_path = os.path.join(output_dir, "analysis_bundle.zip")

    patterns = [
        os.path.join(output_dir, "*.csv"),
        os.path.join(output_dir, "*.md"),
    ]
    files_to_zip: list[str] = []
    for pattern in patterns:
        files_to_zip.extend(glob.glob(pattern))

    # Exclude the zip itself if it already exists
    files_to_zip = [f for f in files_to_zip if f != zip_path]
    files_to_zip = sorted(set(files_to_zip))

    if not files_to_zip:
        logger.warning("No CSV or MD files found in '%s' to bundle.", output_dir)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_zip:
            arcname = os.path.relpath(fpath, output_dir)
            zf.write(fpath, arcname)
            logger.debug("Added to bundle: %s", arcname)

    logger.info(
        "Analysis bundle created: %s (%d files)", zip_path, len(files_to_zip)
    )
    return zip_path
