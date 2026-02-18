from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TrackFrame:
    frame_index: int
    time_s: float
    x: float
    y: float
    x1: float
    y1: float
    x2: float
    y2: float


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


def run_tracking(
    video_path: str,
    sample_fps: int = 5,
    max_frames: Optional[int] = None,
) -> TrackingResult:
    """
    Detection + tracking (players + ball) using YOLO + ByteTrack.
    Uses COCO labels: person -> player, sports ball -> ball.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video tracking.") from exc

    try:
        import supervision as sv
    except ImportError as exc:
        raise RuntimeError("supervision is required for ByteTrack.") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for YOLO detection.") from exc

    model = YOLO("yolov8n.pt")
    model.to("cpu")  # Force CPU to avoid GPU compatibility issues
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return TrackingResult(player_tracks=[], ball_tracks=[])

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    step = max(int(round(fps / max(sample_fps, 1))), 1)
    tracker = sv.ByteTrack(frame_rate=fps)

    player_builders: Dict[str, List[TrackFrame]] = {}
    ball_builders: Dict[str, List[TrackFrame]] = {}
    frame_index = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue

        results = model.predict(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        names = results.names

        if detections.class_id is None or len(detections) == 0:
            frame_index += 1
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
            continue

        tracked = tracker.update_with_detections(detections)
        if tracked.class_id is None or tracked.tracker_id is None:
            frame_index += 1
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
            continue

        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i])
            class_name = names.get(class_id, str(class_id))
            if class_name not in {"person", "sports ball"}:
                continue
            track_id = str(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i]
            x = float((x1 + x2) / 2.0)
            y = float((y1 + y2) / 2.0)
            frame = TrackFrame(
                frame_index=frame_index,
                time_s=frame_index / fps,
                x=x,
                y=y,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
            if class_name == "person":
                player_builders.setdefault(track_id, []).append(frame)
            else:
                ball_builders.setdefault(track_id, []).append(frame)

        frame_index += 1
        processed += 1
        if max_frames is not None and processed >= max_frames:
            break

    cap.release()

    player_tracks = [
        Track(track_id=track_id, label="player", frames=frames)
        for track_id, frames in player_builders.items()
    ]
    ball_tracks = [
        Track(track_id=track_id, label="ball", frames=frames)
        for track_id, frames in ball_builders.items()
    ]
    return TrackingResult(player_tracks=player_tracks, ball_tracks=ball_tracks)
