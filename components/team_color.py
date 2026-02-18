from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from components.tracking import Track, TrackingResult


def assign_team_colors(
    video_path: str,
    tracking: TrackingResult,
    num_teams: int = 2,
    samples_per_track: int = 5,
) -> TrackingResult:
    """
    Assigns team colors to player tracks by clustering jersey colors.
    Returns a new TrackingResult with team_color filled.
    """
    if not tracking.player_tracks:
        return tracking

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and numpy are required for team color assignment.") from exc

    sample_map = _collect_samples(tracking.player_tracks, samples_per_track)
    if not sample_map:
        return tracking

    frame_colors = _extract_colors(video_path, sample_map)
    if not frame_colors:
        return tracking

    feature_list: List[Tuple[str, Tuple[float, float, float]]] = []
    for track_id, colors in frame_colors.items():
        if not colors:
            continue
        mean_color = tuple(float(sum(c[i] for c in colors)) / len(colors) for i in range(3))
        feature_list.append((track_id, mean_color))

    if len(feature_list) < num_teams:
        return tracking

    data = np.array([color for _, color in feature_list], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(data, num_teams, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    cluster_names = [_name_color(center) for center in centers]
    track_to_color: Dict[str, str] = {}
    for (track_id, _), label in zip(feature_list, labels.flatten().tolist()):
        track_to_color[track_id] = cluster_names[int(label)]

    updated_players: List[Track] = []
    for track in tracking.player_tracks:
        color = track_to_color.get(track.track_id)
        updated_players.append(replace(track, team_color=color))

    return TrackingResult(player_tracks=updated_players, ball_tracks=tracking.ball_tracks)


def _collect_samples(player_tracks: List[Track], samples_per_track: int) -> Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]:
    sample_map: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
    for track in player_tracks:
        if not track.frames:
            continue
        step = max(len(track.frames) // samples_per_track, 1)
        for frame in track.frames[::step][:samples_per_track]:
            bbox = (frame.x1, frame.y1, frame.x2, frame.y2)
            sample_map.setdefault(frame.frame_index, []).append((track.track_id, bbox))
    return sample_map


def _extract_colors(
    video_path: str,
    sample_map: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
) -> Dict[str, List[Tuple[float, float, float]]]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    target_frames = sorted(sample_map.keys())
    frame_colors: Dict[str, List[Tuple[float, float, float]]] = {}
    frame_index = 0
    target_iter = iter(target_frames)
    current_target = next(target_iter, None)

    while current_target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index < current_target:
            frame_index += 1
            continue
        if frame_index > current_target:
            current_target = next(target_iter, None)
            continue

        for track_id, (x1, y1, x2, y2) in sample_map[current_target]:
            h, w = frame.shape[:2]
            x1i = max(int(x1), 0)
            y1i = max(int(y1), 0)
            x2i = min(int(x2), w - 1)
            y2i = min(int(y2), h - 1)
            if x2i <= x1i or y2i <= y1i:
                continue

            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue

            jersey = _crop_jersey_region(crop)
            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
            mask = _non_green_mask(hsv)
            pixels = hsv[mask]
            if pixels.size == 0:
                pixels = hsv.reshape(-1, 3)
            mean = pixels.mean(axis=0)
            frame_colors.setdefault(track_id, []).append((float(mean[0]), float(mean[1]), float(mean[2])))

        frame_index += 1
        current_target = next(target_iter, None)

    cap.release()
    return frame_colors


def _crop_jersey_region(crop: "np.ndarray") -> "np.ndarray":
    h, w = crop.shape[:2]
    y1 = int(h * 0.2)
    y2 = int(h * 0.7)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    return crop[y1:y2, x1:x2]


def _non_green_mask(hsv: "np.ndarray") -> "np.ndarray":
    import numpy as np

    # Rough green range in HSV
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    green = ((hsv[:, :, 0] >= lower[0]) & (hsv[:, :, 0] <= upper[0]) &
             (hsv[:, :, 1] >= lower[1]) & (hsv[:, :, 1] <= upper[1]) &
             (hsv[:, :, 2] >= lower[2]) & (hsv[:, :, 2] <= upper[2]))
    return ~green


def _name_color(center: "np.ndarray") -> str:
    h, s, v = center
    if v < 40:
        return "black"
    if s < 30 and v > 180:
        return "white"
    if 0 <= h < 10 or h >= 170:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 110:
        return "cyan"
    if 110 <= h < 140:
        return "blue"
    return "purple"
