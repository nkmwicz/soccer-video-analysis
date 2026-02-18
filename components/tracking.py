from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TrackFrame:
    frame_index: int
    time_s: float
    x: float
    y: float


@dataclass(frozen=True)
class Track:
    track_id: str
    label: str
    frames: List[TrackFrame]
    team_color: Optional[str] = None
    jersey_number: Optional[str] = None


@dataclass(frozen=True)
class TrackingResult:
    player_tracks: List[Track]
    ball_tracks: List[Track]


def run_tracking(video_path: str) -> TrackingResult:
    """
    Placeholder for detection + tracking (players + ball).
    """
    return TrackingResult(player_tracks=[], ball_tracks=[])
