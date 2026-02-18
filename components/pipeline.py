from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from components.csv_writer import write_events_csv
from components.phase_segmentation import segment_game_phases
from components.possession import infer_possessions
from components.tracking import run_tracking
from components.types import ActionEvent, VideoMetadata
from components.video_discovery import discover_videos
from utils.pitch import select_pitch_dimensions


@dataclass
class PipelineConfig:
    videos_dir: Path
    data_dir: Path
    players_on_field: Optional[int] = None


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self) -> None:
        videos = discover_videos(self._config.videos_dir)
        for video_path in videos:
            game_id = video_path.stem
            output_path = self._config.data_dir / f"{game_id}.csv"
            if output_path.exists():
                continue
            metadata = VideoMetadata(
                game_id=game_id,
                video_path=str(video_path),
                players_on_field=self._config.players_on_field,
            )
            events = self._process_video(metadata)
            write_events_csv(output_path, events)

    def _process_video(self, metadata: VideoMetadata) -> List[ActionEvent]:
        _pitch = select_pitch_dimensions(metadata.players_on_field)
        _segments = segment_game_phases(metadata.video_path)
        _tracking = run_tracking(metadata.video_path)
        _possessions = infer_possessions(metadata.video_path, _tracking)
        # Phase 1 placeholder: return no events until detectors are wired in.
        # TODO: add homography mapping from image coordinates to pitch coordinates.
        # TODO: add team color classification + jersey OCR.
        # TODO: add action recognition (actions + subactions).
        return []
