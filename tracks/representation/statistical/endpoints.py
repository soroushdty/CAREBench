"""Endpoints computed by Track 1 (representation).

Track 1 used to number these H1 and H2; the numbers are kept as display
aliases. Track 1's H1 is the same question as Track 3's H2.
"""

from __future__ import annotations

from shared.endpoints import BRIER_IMPROVEMENT, DIRECTIONAL_ALIGNMENT, TrackEndpoint

DIRECTIONAL_ALIGNMENT_T1 = TrackEndpoint(DIRECTIONAL_ALIGNMENT, legacy_alias="H1")
BRIER_IMPROVEMENT_T1 = TrackEndpoint(BRIER_IMPROVEMENT, legacy_alias="H2")

TRACK_ENDPOINTS: tuple[TrackEndpoint, ...] = (
    DIRECTIONAL_ALIGNMENT_T1,
    BRIER_IMPROVEMENT_T1,
)
