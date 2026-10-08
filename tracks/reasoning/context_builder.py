"""
Context builder for the LLM Context-Shift Assay.

Builds correct and shuffled patient context strings for each Patient_Item_Pair.
Context records are loaded through the adapter boundary (PairedContextContextAdapter)
or provided directly as pre-loaded records. This module does NOT directly read
patient-summary files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from adapters.paired_context.context_adapter import PairedContextContextAdapter, CONTEXT_FIELDS


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class LabelLeakageError(Exception):
    """Raised when item text appears (case-insensitive) in a context field.

    Attributes
    ----------
    field_name : str
        The context field where the item text was found.
    matched_text : str
        The item text that was found in the field.
    patient_id : str
        The patient ID whose context triggered the error.
    """

    def __init__(self, field_name: str, matched_text: str, patient_id: str) -> None:
        self.field_name = field_name
        self.matched_text = matched_text
        self.patient_id = patient_id
        super().__init__(
            f"Label leakage detected in patient '{patient_id}': "
            f"item text '{matched_text}' found in field '{field_name}'."
        )


# ---------------------------------------------------------------------------
# Helper: format context fields into structured text
# ---------------------------------------------------------------------------


def format_context(fields: dict[str, Any]) -> str:
    """Format a patient's context fields into a structured text block.

    Fields are rendered in the canonical ``CONTEXT_FIELDS`` order.  Each
    field value may be a string or a list of strings (as stored in
    ``patient_summaries.json``).  Lists are joined with newline-separated
    bullet points.  Missing fields are rendered as empty strings.

    Parameters
    ----------
    fields : dict[str, Any]
        Mapping of field name → value (str or list[str]).

    Returns
    -------
    str
        Structured text with ``[FIELD_NAME]`` headers separated by blank lines.
    """
    return PairedContextContextAdapter.format_context(fields)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Builds correct and shuffled patient context strings.

    Context records are loaded through the adapter boundary. The builder
    accepts either:
    - A ``summaries_path`` (loads via ``PairedContextContextAdapter`` internally)
    - Pre-loaded ``context_records`` (dict of entity_id → fields)
    - A ``context_adapter`` instance directly

    Parameters
    ----------
    summaries_path : str | Path | None
        Path to patient summaries file (loaded via adapter).
    patient_ids : list[str] | list[int] | None
        Explicit list of patient IDs to use as the shuffled-context pool.
        If ``None``, all IDs present in the records are used.
    skip_label_leakage_check : bool
        If ``True``, skip label leakage validation. Default is ``False``.
    context_records : dict[str, dict[str, Any]] | None
        Pre-loaded context records. If provided, ``summaries_path`` is not
        required.
    context_adapter : PairedContextContextAdapter | None
        A context adapter instance. If provided, records are loaded from it.

    Raises
    ------
    FileNotFoundError
        If ``summaries_path`` does not exist.
    ValueError
        If no context source is provided or records are empty.
    """

    def __init__(
        self,
        summaries_path: str | Path | None = None,
        patient_ids: list[str] | list[int] | None = None,
        skip_label_leakage_check: bool = False,
        context_records: dict[str, dict[str, Any]] | None = None,
        context_adapter: PairedContextContextAdapter | None = None,
    ) -> None:
        # Resolve context records from one of the three sources
        if context_records is not None:
            raw = context_records
        elif context_adapter is not None:
            raw = context_adapter.load_context_records()
        elif summaries_path is not None:
            adapter = PairedContextContextAdapter(summaries_path=summaries_path)
            raw = adapter.load_context_records()
        else:
            raise ValueError(
                "ContextBuilder requires one of: summaries_path, "
                "context_records, or context_adapter."
            )

        if not raw:
            raise ValueError("Context records are empty.")

        # Normalise all keys to strings for consistent lookup
        self._summaries: dict[str, dict[str, Any]] = {
            str(k): v for k, v in raw.items()
        }

        # Build the pool of patient IDs available for shuffling
        if patient_ids is not None:
            self._all_patient_ids: list[str] = [str(pid) for pid in patient_ids]
        else:
            self._all_patient_ids = list(self._summaries.keys())

        # Internal mapping: {(patient_id, item_text): shuffled_patient_id}
        self._shuffled_id_map: dict[tuple[str, str], str] = {}

        # Label leakage checking
        self._skip_label_leakage_check = skip_label_leakage_check

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_correct_context(self, patient_id: str | int, item_text: str) -> str:
        """Build the correct patient context for a Patient_Item_Pair.

        Looks up the patient in ``patient_summaries.json``, extracts all eight
        context fields, checks for label leakage, and returns a formatted text
        block.

        Parameters
        ----------
        patient_id : str | int
            The patient identifier.
        item_text : str
            The EHR item text being classified.

        Returns
        -------
        str
            Structured context text.  Deterministic: same ``patient_id``
            always produces the same output.

        Raises
        ------
        KeyError
            If ``patient_id`` is not found in ``patient_summaries.json``.
        LabelLeakageError
            If ``item_text`` appears (case-insensitive) in any context field.
        """
        pid = str(patient_id)
        if pid not in self._summaries:
            raise KeyError(
                f"Patient ID '{pid}' not found in patient_summaries.json."
            )

        fields = self._summaries[pid]
        self._check_label_leakage(pid, item_text, fields)
        return format_context(fields)

    def build_shuffled_context(
        self,
        patient_id: str | int,
        item_text: str,
        rng: np.random.Generator,
    ) -> str:
        """Build a shuffled patient context for a Patient_Item_Pair.

        Selects a different patient ID at random (excluding ``patient_id``),
        records the mapping, and returns the formatted context for the
        shuffled patient.

        Parameters
        ----------
        patient_id : str | int
            The correct patient identifier (excluded from the pool).
        item_text : str
            The EHR item text being classified.
        rng : numpy.random.Generator
            A seeded random generator for deterministic shuffling.  The caller
            is responsible for seeding (e.g. via
            ``numpy.random.default_rng(seed)``).

        Returns
        -------
        str
            Structured context text for the shuffled patient.

        Raises
        ------
        ValueError
            If the pool has fewer than 2 patients (cannot exclude the correct
            patient and still have a candidate).
        LabelLeakageError
            If ``item_text`` appears in any field of the shuffled patient's
            context.
        """
        pid = str(patient_id)
        pool = [p for p in self._all_patient_ids if p != pid]

        if not pool:
            raise ValueError(
                f"Cannot build shuffled context for patient '{pid}': "
                "the patient pool has fewer than 2 entries."
            )

        # Draw a shuffled patient ID
        shuffled_pid: str = pool[int(rng.integers(0, len(pool)))]

        # Record the mapping for audit
        self._shuffled_id_map[(pid, item_text)] = shuffled_pid

        # Build context from the shuffled patient (same pipeline + leakage check)
        if shuffled_pid not in self._summaries:
            raise KeyError(
                f"Shuffled patient ID '{shuffled_pid}' not found in "
                "patient_summaries.json."
            )

        fields = self._summaries[shuffled_pid]
        self._check_label_leakage(shuffled_pid, item_text, fields)
        return format_context(fields)

    def get_shuffled_id_map(self) -> dict[tuple[str, str], str]:
        """Return a copy of the shuffled patient ID mapping for audit.

        Returns
        -------
        dict
            ``{(patient_id, item_text): shuffled_patient_id}`` for every
            :meth:`build_shuffled_context` call made so far.
        """
        return dict(self._shuffled_id_map)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_label_leakage(
        self,
        patient_id: str,
        item_text: str,
        fields: dict[str, Any],
    ) -> None:
        """Raise :class:`LabelLeakageError` if item_text appears in any field.

        The check is case-insensitive and treats list-valued fields by joining
        them into a single string before searching.

        Parameters
        ----------
        patient_id : str
            Patient ID (used in the error message).
        item_text : str
            The item text to search for.
        fields : dict[str, Any]
            The patient's context fields.

        Raises
        ------
        LabelLeakageError
            On the first field where a match is found (if checking is enabled).
        """
        if self._skip_label_leakage_check:
            return

        needle = item_text.lower()
        for field in CONTEXT_FIELDS:
            raw_value = fields.get(field, "")
            if isinstance(raw_value, list):
                haystack = " ".join(str(v) for v in raw_value).lower()
            else:
                haystack = str(raw_value).lower()

            if needle and needle in haystack:
                raise LabelLeakageError(
                    field_name=field,
                    matched_text=item_text,
                    patient_id=patient_id,
                )
