from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


Homography = np.ndarray  # 3x3 matrix


def detect_field_lines(
    frame: np.ndarray,
    line_color: Optional[str] = None,
) -> List[Tuple[float, float, float, float]]:
    """
    Detects field lines in a frame using edge detection + Hough lines.
    line_color: preferred color ("white", "yellow", etc.) to filter candidate lines.
    Returns list of (x1, y1, x2, y2) line segments.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return []

    candidates: List[Tuple[float, float, float, float]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        candidates.append((float(x1), float(y1), float(x2), float(y2)))

    if line_color:
        candidates = _filter_by_color(frame, candidates, line_color)

    return candidates


def compute_homography(
    frame: np.ndarray,
    pitch_dims: Tuple[float, float],
    line_color: Optional[str] = None,
) -> Optional[Homography]:
    """
    Computes homography from frame coords to pitch coords.
    Detects field corners from lines and fits homography.
    pitch_dims: (length_m, width_m)
    Returns 3x3 homography matrix or None if detection fails.
    """
    lines = detect_field_lines(frame, line_color)
    if len(lines) < 4:
        return None

    corners = _fit_field_corners(frame.shape, lines)
    if corners is None or len(corners) < 4:
        return None

    h, w = frame.shape[:2]
    src_pts = np.array(corners[:4], dtype=np.float32)
    dst_pts = np.array(
        [
            [0, 0],
            [pitch_dims[0], 0],
            [pitch_dims[0], pitch_dims[1]],
            [0, pitch_dims[1]],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def apply_homography(
    points_px: List[Tuple[float, float]],
    H: Homography,
) -> List[Tuple[float, float]]:
    """
    Transforms pixel coords to pitch coords using homography.
    points_px: list of (x, y) in pixel space
    Returns list of (x, y) in pitch space.
    """
    if H is None:
        return points_px

    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return [(float(p[0][0]), float(p[0][1])) for p in transformed]


def _filter_by_color(
    frame: np.ndarray,
    candidates: List[Tuple[float, float, float, float]],
    line_color: str,
) -> List[Tuple[float, float, float, float]]:
    """
    Filters line candidates by checking if they match the preferred color.
    """
    h, w = frame.shape[:2]
    filtered = []

    for x1, y1, x2, y2 in candidates:
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(int(x2), w - 1), min(int(y2), h - 1)
        if x2i <= x1i or y2i <= y1i:
            continue

        crop = frame[y1i:y2i, x1i:x2i]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = float(hsv[:, :, 0].mean())
        mean_s = float(hsv[:, :, 1].mean())
        mean_v = float(hsv[:, :, 2].mean())

        if _color_matches(mean_h, mean_s, mean_v, line_color):
            filtered.append((x1, y1, x2, y2))

    return filtered if filtered else candidates


def _color_matches(h: float, s: float, v: float, target: str) -> bool:
    if target.lower() == "white":
        return s < 30 and v > 180
    if target.lower() == "yellow":
        return 25 <= h < 35 and s > 100 and v > 100
    if target.lower() == "green":
        return 35 <= h < 85 and s > 100
    return True


def _fit_field_corners(
    shape: Tuple[int, int, int],
    lines: List[Tuple[float, float, float, float]],
) -> Optional[List[Tuple[float, float]]]:
    """
    Fits field corners (four corners) from detected line segments.
    Placeholder: returns frame corners if fitting fails.
    """
    if len(lines) < 4:
        h, w = shape[:2]
        return [(0, 0), (w, 0), (w, h), (0, h)]

    all_pts = []
    for x1, y1, x2, y2 in lines:
        all_pts.extend([(x1, y1), (x2, y2)])

    if not all_pts:
        h, w = shape[:2]
        return [(0, 0), (w, 0), (w, h), (0, h)]

    pts = np.array(all_pts, dtype=np.float32)
    hull = cv2.convexHull(pts)
    if len(hull) >= 4:
        corners = hull[:4].reshape(-1, 2).tolist()
        return [(float(c[0]), float(c[1])) for c in corners]

    h, w = shape[:2]
    return [(0, 0), (w, 0), (w, h), (0, h)]
