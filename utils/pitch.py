from __future__ import annotations

from components.types import PitchDimensions


# Conservative "standard" sizes used as defaults.
# These vary by league; adjust if your league publishes exact dimensions.
PITCH_9V9 = PitchDimensions(length_m=70.0, width_m=45.0)
PITCH_11V11 = PitchDimensions(length_m=100.0, width_m=64.0)


def select_pitch_dimensions(players_on_field: int | None) -> PitchDimensions:
    if players_on_field is None:
        return PITCH_11V11
    if players_on_field <= 18:
        return PITCH_9V9
    return PITCH_11V11
