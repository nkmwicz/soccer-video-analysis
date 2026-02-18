from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    track_data: "TrackingResult",
    max_ball_distance: float = 60.0,
    min_segment_duration: float = 0.1,
) -> List[PossessionSegment]:
    """
    Infers ball possession by finding nearest player to ball over time.
    Returns list of possession segments (player/team owns ball).
    """
    from components.tracking import TrackingResult

    if not track_data.ball_tracks or not track_data.player_tracks:
        return []

    ball_track = max(track_data.ball_tracks, key=lambda t: len(t.frames))
    if not ball_track.frames:
        return []

    track_map = {t.track_id: t for t in track_data.player_tracks}
    possessions: List[PossessionSegment] = []
    current_owner_id: Optional[str] = None
    segment_start_frame: Optional[int] = None
    segment_start_time: Optional[float] = None

    for ball_frame in ball_track.frames:
        frame_idx = ball_frame.frame_index
        nearest_player = _find_nearest_player_at_frame(
            ball_frame, track_data.player_tracks, max_dist=max_ball_distance
        )

        owner_id = nearest_player.track_id if nearest_player else None

        if owner_id != current_owner_id:
            if current_owner_id is not None and segment_start_frame is not None and segment_start_time is not None:
                duration = ball_frame.time_s - segment_start_time
                if duration >= min_segment_duration:
                    old_owner = track_map.get(current_owner_id)
                    segment = PossessionSegment(
                        possession_id=str(uuid.uuid4())[:8],
                        team_color=old_owner.team_color if old_owner else None,
                        player_track_id=current_owner_id,
                        start_time_s=segment_start_time,
                        end_time_s=ball_frame.time_s,
                        start_frame=segment_start_frame,
                        end_frame=frame_idx,
                    )
                    possessions.append(segment)

            current_owner_id = owner_id
            segment_start_frame = frame_idx
            segment_start_time = ball_frame.time_s

    if current_owner_id is not None and segment_start_frame is not None and segment_start_time is not None:
        old_owner = track_map.get(current_owner_id)
        final_frame = ball_track.frames[-1]
        duration = final_frame.time_s - segment_start_time
        if duration >= min_segment_duration:
            segment = PossessionSegment(
                possession_id=str(uuid.uuid4())[:8],
                team_color=old_owner.team_color if old_owner else None,
                player_track_id=current_owner_id,
                start_time_s=segment_start_time,
                end_time_s=final_frame.time_s,
                start_frame=segment_start_frame,
                end_frame=final_frame.frame_index,
            )
            possessions.append(segment)

    return possessions


def _find_nearest_player_at_frame(
    ball_frame: "TrackFrame",
    player_tracks: list,
    max_dist: float = 100.0,
) -> Optional["Track"]:
    from components.tracking import Track, TrackFrame

    min_dist = max_dist
    nearest = None
    for track in player_tracks:
        for f in track.frames:
            if f.frame_index == ball_frame.frame_index:
                dist = ((f.x - ball_frame.x) ** 2 + (f.y - ball_frame.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = track
                break
    return nearest

