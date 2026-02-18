from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from components.homography import Homography, apply_homography, compute_homography
from components.tracking import TrackingResult


class PitchMapper:
    """Maps frame coordinates to pitch coordinates using per-frame homographies."""

    def __init__(
        self,
        video_path: str,
        tracking: TrackingResult,
        pitch_dims: tuple[float, float],
        line_color: Optional[str] = None,
        sample_interval: int = 30,
    ) -> None:
        self.video_path = video_path
        self.tracking = tracking
        self.pitch_dims = pitch_dims
        self.line_color = line_color
        self.sample_interval = sample_interval
        self._H_cache: Dict[int, Optional[Homography]] = {}

    def get_homography(self, frame_index: int) -> Optional[Homography]:
        """Caches homographies; recomputes if needed."""
        cache_key = (frame_index // self.sample_interval) * self.sample_interval
        if cache_key in self._H_cache:
            return self._H_cache[cache_key]

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cache_key)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            self._H_cache[cache_key] = None
            return None

        H = compute_homography(frame, self.pitch_dims, self.line_color)
        self._H_cache[cache_key] = H
        return H

    def transform_track(self, track_id: str) -> List[tuple[float, float]]:
        """Returns pitch coordinates for all frames in a track."""
        for track in self.tracking.player_tracks:
            if track.track_id != track_id:
                continue

            px_pts = [(f.x, f.y) for f in track.frames]
            pitch_coords = []
            for i, (x, y) in enumerate(px_pts):
                frame_idx = track.frames[i].frame_index
                H = self.get_homography(frame_idx)
                if H is not None:
                    transformed = apply_homography([(x, y)], H)
                    pitch_coords.append(transformed[0] if transformed else (x, y))
                else:
                    pitch_coords.append((x, y))
            return pitch_coords

        return []
