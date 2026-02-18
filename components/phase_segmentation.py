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
    VEO-specific segmentation (kickoff, halftime, second-half).
    Expected phases: "pregame", "first_half", "halftime", "second_half", "postgame".
    """
    if tracking is None:
        tracking = run_tracking(video_path, sample_fps=2, max_frames=2000)

    _ = line_color
    kickoff_time_s = _detect_kickoff_time(tracking)
    if kickoff_time_s is None:
        return []

    halftime_info = _detect_halftime(tracking, kickoff_time_s)
    if halftime_info is None:
        return [
            PhaseSegment(period=1, phase="first_half", start_time_s=kickoff_time_s, end_time_s=kickoff_time_s),
        ]

    halftime_start_s, halftime_end_s = halftime_info
    second_kickoff_s = _detect_second_half_kickoff(tracking, halftime_end_s)

    segments: List[PhaseSegment] = []
    segments.append(
        PhaseSegment(period=1, phase="first_half", start_time_s=kickoff_time_s, end_time_s=halftime_start_s)
    )
    segments.append(
        PhaseSegment(period=1, phase="halftime", start_time_s=halftime_start_s, end_time_s=halftime_end_s)
    )

    if second_kickoff_s is not None:
        segments.append(
            PhaseSegment(period=2, phase="second_half", start_time_s=second_kickoff_s, end_time_s=second_kickoff_s)
        )
    else:
        final_frame = max(tracking.ball_tracks, key=lambda t: len(t.frames)).frames[-1]
        segments.append(
            PhaseSegment(
                period=2, phase="second_half", start_time_s=halftime_end_s, end_time_s=final_frame.time_s
            )
        )

    return segments


def _detect_kickoff_time(tracking: TrackingResult) -> Optional[float]:
    """
    Heuristic kickoff detector:
    - find first ball track
    - look for a dwell period (ball nearly stationary)
    - trigger when ball speed spikes after dwell
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


def _detect_halftime(tracking: TrackingResult, kickoff_time_s: float) -> Optional[tuple[float, float]]:
    """
    Halftime detector:
    - looks for extended period (>30s) with no ball movement or very low activity
    - returns (halftime_start_s, halftime_end_s)
    """
    if not tracking.ball_tracks:
        return None

    ball_track = max(tracking.ball_tracks, key=lambda t: len(t.frames))
    frames = ball_track.frames
    if len(frames) < 20:
        return None

    min_halftime_duration_s = 300.0
    activity_window = 30
    activity_threshold = 50.0

    for idx in range(len(frames) - activity_window):
        window = frames[idx : idx + activity_window]
        if not window:
            continue

        xs = [f.x for f in window]
        ys = [f.y for f in window]
        spatial_range = max(max(xs) - min(xs), max(ys) - min(ys))

        if spatial_range < activity_threshold:
            halftime_start_s = window[0].time_s
            for end_idx in range(idx + activity_window, len(frames)):
                future_window = frames[end_idx : end_idx + activity_window]
                if not future_window:
                    break

                future_xs = [f.x for f in future_window]
                future_ys = [f.y for f in future_window]
                future_range = max(max(future_xs) - min(future_xs), max(future_ys) - min(future_ys))

                if future_range > activity_threshold:
                    halftime_end_s = frames[end_idx].time_s
                    if halftime_end_s - halftime_start_s >= min_halftime_duration_s:
                        return (halftime_start_s, halftime_end_s)
                    break

    return None


def _detect_second_half_kickoff(
    tracking: TrackingResult,
    halftime_end_s: float,
) -> Optional[float]:
    """
    Detects kickoff of second half after halftime.
    Similar to first-half kickoff detection, but starts search after halftime.
    """
    if not tracking.ball_tracks:
        return None

    ball_track = max(tracking.ball_tracks, key=lambda t: len(t.frames))
    frames = ball_track.frames
    if len(frames) < 3:
        return None

    start_idx = next((i for i, f in enumerate(frames) if f.time_s >= halftime_end_s), 0)
    if start_idx + 3 >= len(frames):
        return None

    dwell_window = 5
    speed_threshold_px_s = 80.0
    max_drift_px = 10.0

    for idx in range(start_idx + dwell_window, len(frames) - 1):
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

