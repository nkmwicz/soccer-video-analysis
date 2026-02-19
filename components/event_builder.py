from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from components.action_recognition import ActionCandidate
from components.pitch_mapper import PitchMapper
from components.possession import PossessionSegment
from components.tracking import TrackingResult
from components.types import ActionEvent


def build_action_events(
    game_id: str,
    action_candidates: List[ActionCandidate],
    tracking: TrackingResult,
    pitch_mapper: PitchMapper,
    possession_segments: List[PossessionSegment],
    phase_segments: List,
    team_names: Dict[str, str],
) -> List[ActionEvent]:
    """
    Converts action candidates to ActionEvent objects with pitch coords, team, and possession info.
    """
    events: List[ActionEvent] = []
    track_map = {t.track_id: t for t in tracking.player_tracks}
    possession_map = _build_possession_map(possession_segments)
    phase_map = _build_phase_map(phase_segments)

    for candidate in action_candidates:
        event_id = str(uuid.uuid4())[:8]
        player_track = track_map.get(candidate.player_track_id)
        if player_track is None:
            continue

        start_pitch = pitch_mapper.transform_point(candidate.start_x, candidate.start_y, candidate.start_frame)
        end_pitch = pitch_mapper.transform_point(candidate.end_x, candidate.end_y, candidate.end_frame)

        if start_pitch is None or end_pitch is None:
            start_pitch = (candidate.start_x, candidate.start_y)
            end_pitch = (candidate.end_x, candidate.end_y)

        possession_id = possession_map.get(candidate.start_frame)
        period, phase = phase_map.get(candidate.start_frame, (None, None))
        team_name = team_names.get(player_track.team_color.lower() if player_track.team_color else "", None)

        event = ActionEvent(
            game_id=game_id,
            event_id=event_id,
            period=period,
            phase=phase,
            possession_id=possession_id,
            team_color=player_track.team_color,
            team_name=team_name,
            player_number=player_track.jersey_number,
            player_track_id=candidate.player_track_id,
            ball_owner_track_id=candidate.ball_track_id,
            action=candidate.action,
            subaction=candidate.subaction,
            start_frame=candidate.start_frame,
            end_frame=candidate.end_frame,
            start_time_s=candidate.start_time_s,
            end_time_s=candidate.end_time_s,
            start_x=start_pitch[0],
            start_y=start_pitch[1],
            end_x=end_pitch[0],
            end_y=end_pitch[1],
            confidence=candidate.confidence,
        )
        events.append(event)

    return events


def _build_possession_map(segments: List[PossessionSegment]) -> Dict[int, str]:
    """Maps frame_index to possession_id."""
    possession_map: Dict[int, str] = {}
    for segment in segments:
        if segment.start_frame is None or segment.end_frame is None:
            continue
        for frame in range(segment.start_frame, segment.end_frame + 1):
            possession_map[frame] = segment.possession_id
    return possession_map


def _build_phase_map(segments: List) -> Dict[int, tuple[Optional[int], Optional[str]]]:
    """Maps frame_index to (period, phase)."""
    phase_map: Dict[int, tuple[Optional[int], Optional[str]]] = {}
    for segment in segments:
        if segment.start_frame is None or segment.end_frame is None:
            continue
        for frame in range(segment.start_frame, segment.end_frame + 1):
            phase_map[frame] = (segment.period, segment.phase)
    return phase_map
