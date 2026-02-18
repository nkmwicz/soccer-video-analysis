from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from components.tracking import TrackingResult, run_tracking


@dataclass(frozen=True)
class PhaseSegment:
    period: int
    phase: str
    start_time_s: float
    end_time_s: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


def segment_game_phases(
    video_path: str,
    tracking: Optional[TrackingResult] = None,
    line_color: Optional[str] = None,
) -> List[PhaseSegment]:
    """
    VEO-specific segmentation (kickoff + halftime) placeholder.
    Expected phases: "pregame", "first_half", "halftime", "second_half", "postgame".
    """
    _ = line_color
    if tracking is None:
        tracking = run_tracking(video_path, sample_fps=2, max_frames=2000)

    kickoff_time_s = _detect_kickoff_time(tracking)
    if kickoff_time_s is None:
        return []

    return [
        PhaseSegment(period=1, phase="first_half", start_time_s=kickoff_time_s, end_time_s=kickoff_time_s),
    ]


def _detect_kickoff_time(tracking: TrackingResult) -> Optional[float]:
    """
    Heuristic kickoff detector:
    - find first ball track
    - look for a dwell period (ball nearly stationary)
    - trigger when ball speed spikes after dwell

    This is a placeholder until field mapping and team formation detection
    are implemented.
    """
    if not tracking.ball_tracks:
        return None

    ball_track = max(tracking.ball_tracks, key=lambda t: len(t.frames))
    frames = ball_track.frames
    if len(frames) < 3:
        return None

    dwell_window = 5
    speed_threshold_px_s = 80.0
    max_drift_px = 10.0

    for idx in range(dwell_window, len(frames) - 1):
        window = frames[idx - dwell_window : idx]
        xs = [f.x for f in window]
        ys = [f.y for f in window]
        if max(xs) - min(xs) > max_drift_px or max(ys) - min(ys) > max_drift_px:
            continue
        prev = frames[idx - 1]
        curr = frames[idx]
        next_frame = frames[idx + 1]
        dt = max(next_frame.time_s - curr.time_s, 1e-3)
        vx = (next_frame.x - curr.x) / dt
        vy = (next_frame.y - curr.y) / dt
        speed = (vx * vx + vy * vy) ** 0.5
        if speed >= speed_threshold_px_s:
            return curr.time_s

    return None
