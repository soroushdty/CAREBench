"""Output-dimension label space (the category taxonomy).

A :class:`LabelSpace` is the ordered list of output dimensions a dataset is
labelled with. Each dimension has

- a **key**: the snake_case machine identifier used in score files, JSON
  responses, manifests and result tables;
- a **display name**: the column name in the dataset workbook and the label
  shown in reports;
- an optional **definition**: the text shown to an LLM in the Track 3 prompt.

The default label space is the ten sensitive-data categories used in the
SHARES project (:data:`DEFAULT_LABEL_SPACE`). Other taxonomies are built from
config with :meth:`LabelSpace.from_config`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class OutputDimension:
    """One output dimension of a label space."""

    key: str
    display_name: str
    definition: str | None = None


class UnknownOutputDimensionError(ValueError):
    """Raised when a value is neither a key nor a display name of the label space."""

    def __init__(self, value: str, label_space: LabelSpace) -> None:
        super().__init__(
            f"Unknown output dimension: {value!r}. "
            f"Valid keys: [{', '.join(label_space.keys())}]. "
            f"Valid display names: [{', '.join(label_space.display_names())}]."
        )


def slugify_key(name: str) -> str:
    """Derive a snake_case key from a display name (``"Mood & anxiety"`` → ``"mood_anxiety"``)."""
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if key and key[0].isdigit():
        key = f"c_{key}"
    return key


class LabelSpace:
    """An ordered, validated set of output dimensions.

    Raises
    ------
    ValueError
        If the label space is empty, a key is not snake_case, or keys or
        display names are duplicated.
    """

    def __init__(self, dimensions: Sequence[OutputDimension]) -> None:
        dims = tuple(dimensions)
        if not dims:
            raise ValueError("A label space needs at least one output dimension.")
        for dim in dims:
            if not _KEY_PATTERN.match(dim.key):
                raise ValueError(
                    f"Output-dimension key {dim.key!r} (for {dim.display_name!r}) must be "
                    "snake_case: lowercase letters, digits and underscores, starting with a letter."
                )
        for attr in ("key", "display_name"):
            values = [getattr(d, attr) for d in dims]
            dupes = sorted({v for v in values if values.count(v) > 1})
            if dupes:
                raise ValueError(f"Duplicate output-dimension {attr}(s): {dupes}")
        lowered = [d.display_name.lower() for d in dims]
        if len(set(lowered)) != len(lowered):
            raise ValueError(
                "Output-dimension display names must be unique ignoring case: "
                f"{[d.display_name for d in dims]}"
            )

        self._dims = dims
        self._by_key = {d.key: d for d in dims}
        self._by_display = {d.display_name: d for d in dims}
        self._by_display_lower = {d.display_name.lower(): d for d in dims}

    # ------------------------------------------------------------------
    # Construction from config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        classes: Sequence[str],
        class_definitions: Mapping[str, Any] | None = None,
    ) -> LabelSpace:
        """Build a label space from the ``classes`` and ``class_definitions`` config keys.

        Parameters
        ----------
        classes:
            Display names in column order (the workbook column names).
        class_definitions:
            Optional mapping from display name to either a definition string
            or a mapping with optional ``key`` and ``definition`` entries.

        A class whose name matches a default dimension (by display name,
        ignoring case, or by key) takes the default key and definition unless
        ``class_definitions`` overrides them. Any other class gets a key
        derived from its name and no definition.
        """
        if isinstance(classes, str) or not isinstance(classes, Sequence):
            raise ValueError("'classes' must be a list of class (column) names.")
        definitions = dict(class_definitions or {})
        unknown = sorted(set(definitions) - {str(c) for c in classes})
        if unknown:
            raise ValueError(
                f"'class_definitions' has entries for names not in 'classes': {unknown}"
            )

        dims: list[OutputDimension] = []
        for raw_name in classes:
            name = str(raw_name)
            default = DEFAULT_LABEL_SPACE.get(name)
            key = default.key if default else slugify_key(name)
            definition = default.definition if default else None

            spec = definitions.get(name)
            if isinstance(spec, str):
                definition = spec
            elif isinstance(spec, Mapping):
                extra = sorted(set(spec) - {"key", "definition"})
                if extra:
                    raise ValueError(
                        f"class_definitions[{name!r}] has unknown field(s) {extra}; "
                        "allowed fields are 'key' and 'definition'."
                    )
                key = str(spec.get("key", key))
                if "definition" in spec:
                    definition = spec["definition"]
            elif spec is not None:
                raise ValueError(
                    f"class_definitions[{name!r}] must be a string or a mapping, "
                    f"got {type(spec).__name__}."
                )
            if definition is not None:
                definition = str(definition).strip() or None

            dims.append(OutputDimension(key=key, display_name=name, definition=definition))
        return cls(dims)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dims)

    def __iter__(self) -> Iterator[OutputDimension]:
        return iter(self._dims)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LabelSpace) and self._dims == other._dims

    def __hash__(self) -> int:
        return hash(self._dims)

    def __repr__(self) -> str:
        return f"LabelSpace({list(self.keys())!r})"

    @property
    def dimensions(self) -> tuple[OutputDimension, ...]:
        return self._dims

    def keys(self) -> list[str]:
        """Machine keys in column order."""
        return [d.key for d in self._dims]

    def display_names(self) -> list[str]:
        """Display names in column order."""
        return [d.display_name for d in self._dims]

    def definitions(self) -> dict[str, str | None]:
        """Key → definition (``None`` where no definition was given)."""
        return {d.key: d.definition for d in self._dims}

    def get(self, value: str) -> OutputDimension | None:
        """Look up a dimension by key, display name, or display name ignoring case."""
        return (
            self._by_key.get(value)
            or self._by_display.get(value)
            or self._by_display_lower.get(str(value).lower())
        )

    def normalize(self, value: str) -> str:
        """Return the key for a key or display name (display names match ignoring case)."""
        dim = self.get(value)
        if dim is None:
            raise UnknownOutputDimensionError(value, self)
        return dim.key

    def key_for_display(self, display_name: str) -> str:
        """Return the key for an exact (case-sensitive) display name."""
        dim = self._by_display.get(display_name)
        if dim is None:
            raise UnknownOutputDimensionError(display_name, self)
        return dim.key

    def display_for_key(self, key: str) -> str:
        """Return the display name for a key."""
        dim = self._by_key.get(key)
        if dim is None:
            raise UnknownOutputDimensionError(key, self)
        return dim.display_name

    def manifest(self) -> dict[str, str]:
        """Key → display name, for run manifests."""
        return {d.key: d.display_name for d in self._dims}


# ---------------------------------------------------------------------------
# Default label space: the SHARES sensitive-data categories
# ---------------------------------------------------------------------------

DEFAULT_LABEL_SPACE: LabelSpace = LabelSpace([
    OutputDimension(
        "behavioral_health", "Behavioral health",
        "Mental health conditions, substance use disorders, psychiatric diagnoses, "
        "and related treatments",
    ),
    OutputDimension(
        "diagnoses", "Diagnoses",
        "Medical diagnoses, conditions, and clinical findings",
    ),
    OutputDimension(
        "disabilities", "Disabilities",
        "Physical, cognitive, or developmental disabilities and functional limitations",
    ),
    OutputDimension(
        "infectious_diseases", "Infectious diseases",
        "Communicable diseases including HIV/AIDS, STIs, tuberculosis, and hepatitis",
    ),
    OutputDimension(
        "genetics", "Genetics",
        "Genetic test results, hereditary conditions, and family genetic history",
    ),
    OutputDimension(
        "medications", "Medications",
        "Prescription drugs, dosages, medication history, and pharmacological treatments",
    ),
    OutputDimension(
        "sexual_reproductive_health", "Sexual and reproductive health",
        "Sexual health, reproductive conditions, contraception, pregnancy, and fertility",
    ),
    OutputDimension(
        "social_determinants_of_health", "Social determinants of health",
        "Housing, employment, food security, transportation, and social support factors",
    ),
    OutputDimension(
        "violence", "Violence",
        "Domestic violence, abuse, trauma history, and safety concerns",
    ),
    OutputDimension(
        "other", "Other",
        "Sensitive health information not captured by the above categories",
    ),
])
