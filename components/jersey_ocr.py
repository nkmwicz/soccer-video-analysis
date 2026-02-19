from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from components.tracking import Track, TrackingResult

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def assign_jersey_numbers(
    video_path: str,
    tracking: TrackingResult,
    samples_per_track: int = 5,
    min_confidence: float = 0.4,
) -> TrackingResult:
    """
    Assigns jersey numbers to player tracks using OCR on jersey crops.
    Returns a new TrackingResult with jersey_number filled.
    """
    if not tracking.player_tracks:
        return tracking

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for jersey OCR.") from exc

    try:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("easyocr is required for jersey OCR.") from exc

    sample_map = _collect_samples(tracking.player_tracks, samples_per_track)
    if not sample_map:
        return tracking

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    frame_numbers = _extract_numbers(video_path, sample_map, reader, min_confidence)

    updated_players: List[Track] = []
    for track in tracking.player_tracks:
        numbers = frame_numbers.get(track.track_id, [])
        jersey_number = _choose_number(numbers)
        updated_players.append(replace(track, jersey_number=jersey_number))

    return TrackingResult(player_tracks=updated_players, ball_tracks=tracking.ball_tracks)


def _collect_samples(
    player_tracks: List[Track],
    samples_per_track: int,
) -> Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]:
    sample_map: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
    for track in player_tracks:
        if not track.frames:
            continue
        step = max(len(track.frames) // samples_per_track, 1)
        for frame in track.frames[::step][:samples_per_track]:
            bbox = (frame.x1, frame.y1, frame.x2, frame.y2)
            sample_map.setdefault(frame.frame_index, []).append((track.track_id, bbox))
    return sample_map


def _extract_numbers(
    video_path: str,
    sample_map: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]],
    reader: "easyocr.Reader",
    min_confidence: float,
) -> Dict[str, List[str]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    target_frames = sorted(sample_map.keys())
    frame_numbers: Dict[str, List[str]] = {}
    frame_index = 0
    target_iter = iter(target_frames)
    current_target = next(target_iter, None)
    pbar = tqdm(total=len(target_frames), desc="Jersey OCR", unit="frame") if tqdm else None

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
            gray = cv2.cvtColor(jersey, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray)
            for _, text, conf in results:
                if conf < min_confidence:
                    continue
                digits = "".join(ch for ch in text if ch.isdigit())
                if digits:
                    frame_numbers.setdefault(track_id, []).append(digits)

        frame_index += 1
        if pbar:
            pbar.update(1)
        current_target = next(target_iter, None)

    if pbar:
        pbar.close()
    cap.release()
    return frame_numbers


def _choose_number(numbers: List[str]) -> Optional[str]:
    if not numbers:
        return None
    counts = Counter(numbers)
    return counts.most_common(1)[0][0]


def _crop_jersey_region(crop: "cv2.Mat") -> "cv2.Mat":
    h, w = crop.shape[:2]
    y1 = int(h * 0.2)
    y2 = int(h * 0.8)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    return crop[y1:y2, x1:x2]
