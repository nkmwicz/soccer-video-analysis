from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PossessionSegment:
    possession_id: str
    team_color: Optional[str]
    player_track_id: Optional[str]
    start_time_s: float
    end_time_s: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


def infer_possessions(
    video_path: str,
    track_data: object,
) -> List[PossessionSegment]:
    """
    Placeholder for possession inference (ball-owner assignment).
    """
    return []
