from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PitchDimensions:
    length_m: float
    width_m: float


@dataclass(frozen=True)
class VideoMetadata:
    game_id: str
    video_path: str
    players_on_field: Optional[int] = None


@dataclass(frozen=True)
class ActionEvent:
    game_id: str
    event_id: Optional[str]
    period: Optional[int]
    phase: Optional[str]
    possession_id: Optional[str]
    team_color: Optional[str]
    team_name: Optional[str]
    player_number: Optional[str]
    player_track_id: Optional[str]
    ball_owner_track_id: Optional[str]
    action: str
    subaction: Optional[str]
    start_frame: Optional[int]
    end_frame: Optional[int]
    start_time_s: float
    end_time_s: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    confidence: float
