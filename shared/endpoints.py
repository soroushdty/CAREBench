"""Named endpoints shared by all tracks.

Each endpoint answers one question about how a model's judgments respond to
patient context. Tracks declare which endpoints they compute and write their
results under these names, so that "directional alignment" means the same
question in every track and every report. The statistic and inference
procedure each track uses are described in ``docs/methodology.md``.

Earlier versions numbered hypotheses per track (Track 3's H2 was Track 1's
H1). Tracks keep their old numbers only as display aliases; see
``legacy_alias`` in each track's endpoint declaration.

``PROTOCOL_VERSION`` versions the evaluation protocol separately from the
code: the endpoints, their statistics and their tests. It is written to
every run manifest and report, so results can cite the protocol they were
computed under, and a refactor that leaves the numbers unchanged does not
invalidate them. Bump it whenever an endpoint definition, statistic, test
or default that changes reported values is changed, and add a row to the
protocol history in ``docs/methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = "1"


@dataclass(frozen=True)
class Endpoint:
    """A named endpoint: what it is called and which question it answers."""

    name: str
    title: str
    question: str


CONTEXT_SENSITIVITY = Endpoint(
    "context_sensitivity",
    "Context sensitivity",
    "Does the correct patient's context change the model's scores more than "
    "another patient's context does?",
)
DIRECTIONAL_ALIGNMENT = Endpoint(
    "directional_alignment",
    "Directional alignment",
    "Does the model's score change in the same direction as the reference "
    "observers' judgment change?",
)
CLASS_CORRESPONDENCE = Endpoint(
    "class_correspondence",
    "Class-level correspondence",
    "Do the categories that shift most for the reference observers also shift "
    "most for the model?",
)
CONTEXT_SPECIFICITY = Endpoint(
    "context_specificity",
    "Context specificity",
    "Is the alignment specific to the correct patient's context, compared "
    "with another patient's (shuffled) context?",
)
BRIER_IMPROVEMENT = Endpoint(
    "brier_improvement",
    "Brier improvement",
    "Does context bring the model's scores closer to the correct-context "
    "reference labels (lower Brier score)?",
)

ENDPOINTS: dict[str, Endpoint] = {
    e.name: e
    for e in (
        CONTEXT_SENSITIVITY,
        DIRECTIONAL_ALIGNMENT,
        CLASS_CORRESPONDENCE,
        CONTEXT_SPECIFICITY,
        BRIER_IMPROVEMENT,
    )
}


@dataclass(frozen=True)
class TrackEndpoint:
    """An endpoint as computed by one track."""

    endpoint: Endpoint
    legacy_alias: str | None = None

    @property
    def name(self) -> str:
        return self.endpoint.name

    @property
    def heading(self) -> str:
        """Report heading, e.g. ``"Context sensitivity (formerly H1)"``."""
        if self.legacy_alias:
            return f"{self.endpoint.title} (formerly {self.legacy_alias})"
        return self.endpoint.title


def get_endpoint(name: str) -> Endpoint:
    """Return the endpoint called *name*."""
    try:
        return ENDPOINTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown endpoint {name!r}. Known endpoints: {', '.join(ENDPOINTS)}"
        ) from None
