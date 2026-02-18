from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from components.tracking import Track, TrackingResult


@dataclass
class ActionCandidate:
    action: str
    subaction: Optional[str]
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    player_track_id: Optional[str]
    ball_track_id: Optional[str]
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    confidence: float


def recognize_actions(
    tracking: TrackingResult,
    pitch_dims: Tuple[float, float],
) -> List[ActionCandidate]:
    """
    Recognizes soccer actions from tracking data using heuristics.
    """
    actions: List[ActionCandidate] = []

    if not tracking.ball_tracks or not tracking.player_tracks:
        return actions

    ball_track = max(tracking.ball_tracks, key=lambda t: len(t.frames))
    if len(ball_track.frames) < 2:
        return actions

    actions.extend(_detect_passes(ball_track, tracking.player_tracks))
    actions.extend(_detect_shots(ball_track, tracking.player_tracks, pitch_dims))
    actions.extend(_detect_dribbles(ball_track, tracking.player_tracks))
    actions.extend(_detect_challenges(ball_track, tracking.player_tracks))
    actions.extend(_detect_intercepts(ball_track, tracking.player_tracks))

    return sorted(actions, key=lambda a: a.start_frame)


def _detect_passes(ball_track: Track, player_tracks: List[Track]) -> List[ActionCandidate]:
    """
    Pass: ball near player A, then moves to player B (same team, within ~3 seconds).
    """
    passes: List[ActionCandidate] = []
    frames = ball_track.frames
    if len(frames) < 10:
        return passes

    for i in range(len(frames) - 1):
        frame = frames[i]
        nearby_player = _find_nearest_player(frame, player_tracks, max_dist=50)
        if nearby_player is None:
            continue

        future_frames = frames[i:min(i + 90, len(frames))]
        for j, future_frame in enumerate(future_frames):
            if j == 0:
                continue
            other_player = _find_nearest_player(future_frame, player_tracks, max_dist=50)
            if other_player is None or other_player.track_id == nearby_player.track_id:
                continue
            if not _same_team(nearby_player, other_player):
                continue

            dist_moved = ((future_frame.x - frame.x) ** 2 + (future_frame.y - frame.y) ** 2) ** 0.5
            if dist_moved < 30:
                continue

            subaction = "accurate" if _is_successful_pass(nearby_player, other_player, future_frame) else "inaccurate"
            passes.append(
                ActionCandidate(
                    action="pass",
                    subaction=subaction,
                    start_frame=frame.frame_index,
                    end_frame=future_frame.frame_index,
                    start_time_s=frame.time_s,
                    end_time_s=future_frame.time_s,
                    player_track_id=nearby_player.track_id,
                    ball_track_id=ball_track.track_id,
                    start_x=frame.x,
                    start_y=frame.y,
                    end_x=future_frame.x,
                    end_y=future_frame.y,
                    confidence=0.7,
                )
            )
            break

    return passes


def _detect_shots(
    ball_track: Track,
    player_tracks: List[Track],
    pitch_dims: Tuple[float, float],
) -> List[ActionCandidate]:
    """
    Shot: ball moves with high velocity towards goal line.
    """
    shots: List[ActionCandidate] = []
    frames = ball_track.frames
    if len(frames) < 5:
        return shots

    length, width = pitch_dims
    goal_threshold = length * 0.1

    for i in range(1, len(frames) - 1):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        next_frame = frames[i + 1]

        dt = max(curr_frame.time_s - prev_frame.time_s, 1e-3)
        vx = (next_frame.x - prev_frame.x) / (2 * dt)
        vy = (next_frame.y - prev_frame.y) / (2 * dt)
        speed = (vx * vx + vy * vy) ** 0.5

        if speed < 100:
            continue

        if curr_frame.x > length - goal_threshold or curr_frame.x < goal_threshold:
            nearby_player = _find_nearest_player(curr_frame, player_tracks, max_dist=80)
            subaction = "goal" if _near_goal(curr_frame, length) else "inaccurate"
            shots.append(
                ActionCandidate(
                    action="shoot",
                    subaction=subaction,
                    start_frame=curr_frame.frame_index,
                    end_frame=next_frame.frame_index,
                    start_time_s=curr_frame.time_s,
                    end_time_s=next_frame.time_s,
                    player_track_id=nearby_player.track_id if nearby_player else None,
                    ball_track_id=ball_track.track_id,
                    start_x=curr_frame.x,
                    start_y=curr_frame.y,
                    end_x=next_frame.x,
                    end_y=next_frame.y,
                    confidence=0.6,
                )
            )

    return shots


def _detect_dribbles(ball_track: Track, player_tracks: List[Track]) -> List[ActionCandidate]:
    """
    Dribble: ball stays close to one player for sustained time while moving.
    """
    dribbles: List[ActionCandidate] = []
    frames = ball_track.frames
    if len(frames) < 5:
        return dribbles

    i = 0
    while i < len(frames):
        start_frame = frames[i]
        player = _find_nearest_player(start_frame, player_tracks, max_dist=40)
        if player is None:
            i += 1
            continue

        j = i + 1
        while j < len(frames):
            frame = frames[j]
            nearest = _find_nearest_player(frame, player_tracks, max_dist=40)
            if nearest is None or nearest.track_id != player.track_id:
                break
            j += 1

        if j - i >= 5:
            end_frame = frames[j - 1]
            dist_moved = ((end_frame.x - start_frame.x) ** 2 + (end_frame.y - start_frame.y) ** 2) ** 0.5
            if dist_moved > 30:
                subaction = "keep" if _dribble_kept(player_tracks, player) else "lose"
                dribbles.append(
                    ActionCandidate(
                        action="dribble",
                        subaction=subaction,
                        start_frame=start_frame.frame_index,
                        end_frame=end_frame.frame_index,
                        start_time_s=start_frame.time_s,
                        end_time_s=end_frame.time_s,
                        player_track_id=player.track_id,
                        ball_track_id=ball_track.track_id,
                        start_x=start_frame.x,
                        start_y=start_frame.y,
                        end_x=end_frame.x,
                        end_y=end_frame.y,
                        confidence=0.65,
                    )
                )
        i = j if j > i + 1 else i + 1

    return dribbles


def _detect_challenges(ball_track: Track, player_tracks: List[Track]) -> List[ActionCandidate]:
    """
    Challenge: two opposing players close together competing for the ball.
    """
    challenges: List[ActionCandidate] = []
    frames = ball_track.frames
    if len(frames) < 2:
        return challenges

    for i in range(len(frames)):
        frame = frames[i]
        nearby = _find_all_nearby_players(frame, player_tracks, max_dist=60)
        if len(nearby) < 2:
            continue

        teams = set()
        for p in nearby:
            teams.add((p.team_color, p.jersey_number))

        if len(teams) >= 2:
            player1 = nearby[0]
            player2 = next((p for p in nearby if p.track_id != player1.track_id and p.team_color != player1.team_color), None)
            if player2:
                challenges.append(
                    ActionCandidate(
                        action="challenge",
                        subaction="win" if player1.team_color == player2.team_color else "lose",
                        start_frame=frame.frame_index,
                        end_frame=frame.frame_index,
                        start_time_s=frame.time_s,
                        end_time_s=frame.time_s,
                        player_track_id=player1.track_id,
                        ball_track_id=ball_track.track_id,
                        start_x=frame.x,
                        start_y=frame.y,
                        end_x=frame.x,
                        end_y=frame.y,
                        confidence=0.5,
                    )
                )

    return challenges


def _detect_intercepts(ball_track: Track, player_tracks: List[Track]) -> List[ActionCandidate]:
    """
    Intercept: defender gets to ball before attacker passes it.
    """
    intercepts: List[ActionCandidate] = []
    frames = ball_track.frames
    if len(frames) < 10:
        return intercepts

    for i in range(5, len(frames) - 5):
        frame = frames[i]
        nearby = _find_all_nearby_players(frame, player_tracks, max_dist=50)
        if len(nearby) < 2:
            continue

        attacker = nearby[0]
        defender = next((p for p in nearby if p.team_color != attacker.team_color), None)
        if defender is None:
            continue

        prev_frames = frames[max(0, i - 10):i]
        was_attacker_possession = any(
            _find_nearest_player(f, player_tracks, max_dist=40) == attacker for f in prev_frames
        )

        if was_attacker_possession:
            intercepts.append(
                ActionCandidate(
                    action="intercept",
                    subaction="success",
                    start_frame=frame.frame_index,
                    end_frame=frame.frame_index,
                    start_time_s=frame.time_s,
                    end_time_s=frame.time_s,
                    player_track_id=defender.track_id,
                    ball_track_id=ball_track.track_id,
                    start_x=frame.x,
                    start_y=frame.y,
                    end_x=frame.x,
                    end_y=frame.y,
                    confidence=0.55,
                )
            )

    return intercepts


def _find_nearest_player(
    frame_pos: "TrackFrame",
    player_tracks: List[Track],
    max_dist: float = 100,
) -> Optional[Track]:
    min_dist = max_dist
    nearest = None
    for track in player_tracks:
        for f in track.frames:
            if f.frame_index == frame_pos.frame_index:
                dist = ((f.x - frame_pos.x) ** 2 + (f.y - frame_pos.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = track
                break
    return nearest


def _find_all_nearby_players(
    frame_pos: "TrackFrame",
    player_tracks: List[Track],
    max_dist: float = 100,
) -> List[Track]:
    nearby = []
    for track in player_tracks:
        for f in track.frames:
            if f.frame_index == frame_pos.frame_index:
                dist = ((f.x - frame_pos.x) ** 2 + (f.y - frame_pos.y) ** 2) ** 0.5
                if dist < max_dist:
                    nearby.append(track)
                break
    return nearby


def _same_team(player1: Track, player2: Track) -> bool:
    return player1.team_color is not None and player1.team_color == player2.team_color


def _is_successful_pass(passer: Track, receiver: Track, ball_frame: "TrackFrame") -> bool:
    for f in receiver.frames:
        if f.frame_index == ball_frame.frame_index:
            dist = ((f.x - ball_frame.x) ** 2 + (f.y - ball_frame.y) ** 2) ** 0.5
            return dist < 50
    return False


def _near_goal(frame: "TrackFrame", length: float) -> bool:
    return frame.x < length * 0.05 or frame.x > length * 0.95


def _dribble_kept(player_tracks: List[Track], player: Track) -> bool:
    return player.jersey_number is not None


# Type hints for TrackFrame (avoid circular import)
from components.tracking import TrackFrame
