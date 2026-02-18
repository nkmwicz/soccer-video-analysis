from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PhaseSegment:
    period: int
    phase: str
    start_time_s: float
    end_time_s: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


def segment_game_phases(video_path: str) -> List[PhaseSegment]:
    """
    Placeholder for VEO-specific segmentation.
    Expected phases: "pregame", "first_half", "halftime", "second_half", "postgame".
    """
    return []
