"""Endpoints computed by Track 3 (reasoning).

Track 3 used to number these H1-H4; the numbers are kept as display aliases.
"""

from __future__ import annotations

from shared.endpoints import (
    CLASS_CORRESPONDENCE,
    CONTEXT_SENSITIVITY,
    CONTEXT_SPECIFICITY,
    DIRECTIONAL_ALIGNMENT,
    TrackEndpoint,
)

CONTEXT_SENSITIVITY_T3 = TrackEndpoint(CONTEXT_SENSITIVITY, legacy_alias="H1")
DIRECTIONAL_ALIGNMENT_T3 = TrackEndpoint(DIRECTIONAL_ALIGNMENT, legacy_alias="H2")
CLASS_CORRESPONDENCE_T3 = TrackEndpoint(CLASS_CORRESPONDENCE, legacy_alias="H3")
CONTEXT_SPECIFICITY_T3 = TrackEndpoint(CONTEXT_SPECIFICITY, legacy_alias="H4")

TRACK_ENDPOINTS: tuple[TrackEndpoint, ...] = (
    CONTEXT_SENSITIVITY_T3,
    DIRECTIONAL_ALIGNMENT_T3,
    CLASS_CORRESPONDENCE_T3,
    CONTEXT_SPECIFICITY_T3,
)
