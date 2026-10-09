"""
Prompt template for the LLM Context-Shift Assay.

Generates deterministic prompts for each of the three conditions:
  - context_free
  - correct_context
  - shuffled_context

The category list and JSON format come from a
:class:`~shared.label_space.LabelSpace` (by default the ten SHARES
categories).
"""

from __future__ import annotations

import hashlib

from shared.label_space import DEFAULT_LABEL_SPACE, LabelSpace

# ---------------------------------------------------------------------------
# Default category definitions (kept as module constants for backward
# compatibility; the source of truth is shared.label_space)
# ---------------------------------------------------------------------------

PRIVACY_CATEGORIES: dict[str, str] = {
    d.key: d.definition for d in DEFAULT_LABEL_SPACE
}

# Canonical ordered list of category names (matches glossary order)
CATEGORY_NAMES: list[str] = DEFAULT_LABEL_SPACE.keys()

# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_INTRO = (
    "You are a clinical privacy expert. "
    "Classify the following EHR item into privacy categories."
)

_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)


def _category_header(n_categories: int) -> str:
    count = (
        _NUMBER_WORDS[n_categories]
        if n_categories < len(_NUMBER_WORDS)
        else str(n_categories)
    )
    noun = "category" if n_categories == 1 else "categories"
    return (
        f"For each of the following {count} privacy {noun}, provide a score between 0.0 and 1.0\n"
        "representing the probability that this item belongs to that category."
    )


def _category_block(label_space: LabelSpace) -> str:
    return "Categories:\n" + "\n".join(
        f"- {d.key}: {d.definition}" if d.definition else f"- {d.key}"
        for d in label_space
    )


def _json_format(label_space: LabelSpace) -> str:
    return (
        "Respond with ONLY a JSON object in this exact format:\n"
        "{\n"
        + ",\n".join(f'  "{d.key}": <score>' for d in label_space)
        + "\n}"
    )


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class PromptTemplate:
    """Generates deterministic prompts for the three assay conditions.

    All methods are pure functions of their arguments — no timestamps,
    no random elements, no external I/O.  The same inputs always produce
    the same prompt string.

    Physician labels, ground-truth classifications, and any reference to
    ``Physician_Survey_Consensus``, ``Physician_Interview_Consensus``, or
    ``Delta_Physician`` are intentionally absent from every prompt variant.

    Parameters
    ----------
    label_space : LabelSpace, optional
        The categories listed in the prompt and required in the JSON answer.
        Defaults to the ten SHARES categories. Categories without a
        definition are listed by key only.
    """

    def __init__(self, label_space: LabelSpace | None = None) -> None:
        self._label_space = label_space or DEFAULT_LABEL_SPACE
        self._header = _category_header(len(self._label_space))
        self._category_block = _category_block(self._label_space)
        self._json_format = _json_format(self._label_space)

    @property
    def label_space(self) -> LabelSpace:
        return self._label_space

    # ------------------------------------------------------------------
    # Public prompt builders
    # ------------------------------------------------------------------

    def build_context_free_prompt(self, item_text: str) -> str:
        """Build a context-free prompt containing only the item and category definitions.

        Parameters
        ----------
        item_text : str
            The EHR item string to classify.

        Returns
        -------
        str
            Deterministic prompt for the ``context_free`` condition.
        """
        parts = [
            _INTRO,
            "",
            f"ITEM: {item_text}",
            "",
            self._header,
            "",
            self._category_block,
            "",
            self._json_format,
        ]
        return "\n".join(parts)

    def build_correct_context_prompt(
        self, item_text: str, context_text: str
    ) -> str:
        """Build a correct-context prompt with the patient's clinical snapshot.

        The ``PATIENT CONTEXT:`` section is inserted between the item and the
        category definitions.

        Parameters
        ----------
        item_text : str
            The EHR item string to classify.
        context_text : str
            The formatted patient context produced by
            :class:`~assay.context_builder.ContextBuilder`.

        Returns
        -------
        str
            Deterministic prompt for the ``correct_context`` condition.
        """
        return self._build_context_prompt(item_text, context_text)

    def build_shuffled_context_prompt(
        self, item_text: str, shuffled_context_text: str
    ) -> str:
        """Build a shuffled-context prompt with a different patient's clinical snapshot.

        Structurally identical to :meth:`build_correct_context_prompt`; the
        caller is responsible for supplying the shuffled context text.

        Parameters
        ----------
        item_text : str
            The EHR item string to classify.
        shuffled_context_text : str
            The formatted context for the shuffled (control) patient.

        Returns
        -------
        str
            Deterministic prompt for the ``shuffled_context`` condition.
        """
        return self._build_context_prompt(item_text, shuffled_context_text)

    # ------------------------------------------------------------------
    # Hash utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_prompt_hash(prompt: str) -> str:
        """Return a 16-character SHA-256 hex digest of the prompt.

        Used as part of the cache key to detect prompt changes.

        Parameters
        ----------
        prompt : str
            The full prompt string.

        Returns
        -------
        str
            First 16 hex characters of the SHA-256 digest.
        """
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------

    @staticmethod
    def verify_no_physician_labels(prompt: str) -> bool:
        """Return True if the prompt contains no physician label references.

        Checks for the presence of any of the following strings
        (case-insensitive):
          - ``Physician_Survey_Consensus``
          - ``Physician_Interview_Consensus``
          - ``Delta_Physician``
          - ``Delta_LLM``
          - ``ground truth``
          - ``ground_truth``

        Parameters
        ----------
        prompt : str
            The prompt string to inspect.

        Returns
        -------
        bool
            ``True`` if none of the forbidden strings are found; ``False``
            otherwise.
        """
        forbidden = [
            "physician_survey_consensus",
            "physician_interview_consensus",
            "delta_physician",
            "delta_llm",
            "ground truth",
            "ground_truth",
        ]
        lower_prompt = prompt.lower()
        return not any(term in lower_prompt for term in forbidden)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context_prompt(self, item_text: str, context_text: str) -> str:
        """Shared builder for correct-context and shuffled-context prompts."""
        parts = [
            _INTRO,
            "",
            f"ITEM: {item_text}",
            "",
            "PATIENT CONTEXT:",
            context_text,
            "",
            self._header,
            "",
            self._category_block,
            "",
            self._json_format,
        ]
        return "\n".join(parts)
