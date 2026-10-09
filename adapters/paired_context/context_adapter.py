"""Paired-context context adapter — patient-summary loading.

Isolates patient-summary JSON loading and field structure behind
the ContextAdapter protocol. Track 3 (reasoning) does not need to know
the file format or field names of the patient summaries.

Canonical path: adapters/paired_context/context_adapter.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants: patient-summary field ordering
# ---------------------------------------------------------------------------

CONTEXT_FIELDS: list[str] = [
    "summary",
    "medical_history",
    "allergies",
    "medication_history",
    "social_history",
    "labs",
    "radiology",
    "procedures",
]


class PairedContextContextAdapter:
    """Patient-summary context adapter for the paired-context format.

    Loads ``patient_summaries.json`` and exposes records by a generic
    ``context_entity_id`` key. Preserves the canonical field ordering
    required by context builders without embedding file I/O in Track 3.

    Satisfies: ``shared.adapters.base.ContextAdapter``

    Parameters
    ----------
    summaries_path : str | Path
        Path to the ``patient_summaries.json`` file.

    Raises
    ------
    FileNotFoundError
        If ``summaries_path`` does not exist (raised on ``load_context_records``).
    ValueError
        If the JSON file is empty or contains no patient records.
    """

    def __init__(self, summaries_path: str | Path) -> None:
        self._summaries_path = Path(summaries_path)
        self._records: dict[str, dict[str, Any]] | None = None

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return "paired_context_context"

    def manifest(self) -> dict[str, Any]:
        """Return reproducibility manifest for context loading."""
        return {
            "adapter": self.name,
            "context_source_path": str(self._summaries_path),
            "parsing_policy": "patient_summaries_json",
            "context_fields": list(CONTEXT_FIELDS),
        }

    def load_context_records(self) -> dict[str, dict[str, Any]]:
        """Load and return context records keyed by context_entity_id.

        Returns a mapping of ``{entity_id: {field: value, ...}}`` with all
        keys normalized to strings.

        Raises
        ------
        FileNotFoundError
            If the summaries file does not exist.
        ValueError
            If the JSON file is empty.
        """
        if self._records is not None:
            return self._records

        if not self._summaries_path.exists():
            raise FileNotFoundError(
                f"Patient summaries file not found: {self._summaries_path}"
            )

        with self._summaries_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)

        if not raw:
            raise ValueError(
                f"Patient summaries file is empty: {self._summaries_path}"
            )

        # Normalize all keys to strings for consistent lookup
        self._records = {str(k): v for k, v in raw.items()}
        return self._records

    def get_entity_ids(self) -> list[str]:
        """Return all available context entity IDs.

        Triggers loading if records have not been loaded yet.
        """
        records = self.load_context_records()
        return list(records.keys())

    def get_record(self, context_entity_id: str) -> dict[str, Any]:
        """Return the context record for a specific entity.

        Parameters
        ----------
        context_entity_id : str
            The entity identifier (patient ID as string).

        Returns
        -------
        dict[str, Any]
            The context fields for the requested entity.

        Raises
        ------
        KeyError
            If the entity ID is not found.
        """
        records = self.load_context_records()
        eid = str(context_entity_id)
        if eid not in records:
            raise KeyError(
                f"Context entity ID '{eid}' not found in patient summaries."
            )
        return records[eid]

    @staticmethod
    def format_context(fields: dict[str, Any]) -> str:
        """Format a patient's context fields into a structured text block.

        Fields are rendered in the canonical ``CONTEXT_FIELDS`` order. Each
        field value may be a string or a list of strings. Lists are joined
        with newline-separated bullet points. Missing fields render as empty.

        Parameters
        ----------
        fields : dict[str, Any]
            Mapping of field name → value (str or list[str]).

        Returns
        -------
        str
            Structured text with ``[FIELD_NAME]`` headers separated by blank lines.
        """
        lines: list[str] = []
        for field in CONTEXT_FIELDS:
            raw_value = fields.get(field, "")
            if isinstance(raw_value, list):
                value = "\n".join(f"- {item}" for item in raw_value) if raw_value else ""
            else:
                value = str(raw_value) if raw_value else ""
            value = value.strip()
            lines.append(f"[{field.upper()}]\n{value}")
        return "\n\n".join(lines)
