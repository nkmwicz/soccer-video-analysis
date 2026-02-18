from __future__ import annotations

from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def discover_videos(videos_dir: Path) -> Iterable[Path]:
    if not videos_dir.exists():
        return []
    return sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
