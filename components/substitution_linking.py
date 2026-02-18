from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from components.tracking import Track, TrackingResult


def link_substitutions(
    tracking: TrackingResult,
    max_time_gap_s: float = 5.0,
    max_position_dist_px: float = 100.0,
) -> TrackingResult:
    """
    Links player tracks across substitutions.
    If a track ends and a new one starts nearby in time/space with a matching jersey,
    merge them into a single track.
    """
    if len(tracking.player_tracks) < 2:
        return tracking

    player_tracks = list(tracking.player_tracks)
    merged_ids: Set[str] = set()
    track_map: Dict[str, Track] = {t.track_id: t for t in player_tracks}
    jersey_map: Dict[Optional[str], List[str]] = {}

    for track in player_tracks:
        jersey_map.setdefault(track.jersey_number, []).append(track.track_id)

    replacements: Dict[str, str] = {}
    for jersey, track_ids in jersey_map.items():
        if jersey is None or len(track_ids) <= 1:
            continue
        sorted_ids = sorted(track_ids, key=lambda tid: track_map[tid].frames[0].time_s)
        for i in range(len(sorted_ids) - 1):
            old_id = sorted_ids[i]
            new_id = sorted_ids[i + 1]
            old_track = track_map[old_id]
            new_track = track_map[new_id]

            if not old_track.frames or not new_track.frames:
                continue

            old_end_time = old_track.frames[-1].time_s
            new_start_time = new_track.frames[0].time_s
            time_gap = new_start_time - old_end_time

            if time_gap > max_time_gap_s or time_gap < 0:
                continue

            old_end_pos = (old_track.frames[-1].x, old_track.frames[-1].y)
            new_start_pos = (new_track.frames[0].x, new_track.frames[0].y)
            dist = ((old_end_pos[0] - new_start_pos[0]) ** 2 + (old_end_pos[1] - new_start_pos[1]) ** 2) ** 0.5

            if dist > max_position_dist_px:
                continue

            replacements[new_id] = old_id
            merged_ids.add(new_id)

    updated_tracks: List[Track] = []
    for track in player_tracks:
        if track.track_id in merged_ids:
            continue
        updated_tracks.append(track)

    for i, track in enumerate(updated_tracks):
        if track.track_id in replacements:
            target_id = replacements[track.track_id]
            for j, t in enumerate(updated_tracks):
                if t.track_id == target_id:
                    merged = _merge_tracks(t, track)
                    updated_tracks[j] = merged
                    break

    return TrackingResult(player_tracks=updated_tracks, ball_tracks=tracking.ball_tracks)


def _merge_tracks(track1: Track, track2: Track) -> Track:
    frames = track1.frames + track2.frames
    frames = sorted(frames, key=lambda f: f.time_s)
    jersey = track1.jersey_number or track2.jersey_number
    return replace(track1, frames=frames, jersey_number=jersey)
