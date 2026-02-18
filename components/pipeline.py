from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from components.csv_writer import write_events_csv
from components.jersey_ocr import assign_jersey_numbers
from components.substitution_linking import link_substitutions
from components.phase_segmentation import segment_game_phases
from components.possession import infer_possessions
from components.team_color import assign_team_colors
from components.tracking import run_tracking
from components.types import ActionEvent, VideoMetadata
from components.video_discovery import discover_videos
from utils.pitch import select_pitch_dimensions


@dataclass
class PipelineConfig:
    videos_dir: Path
    data_dir: Path
    players_on_field: Optional[int] = None
    line_color: Optional[str] = None


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
            line_color = self._config.line_color
            if line_color is None:
                response = input(
                    f"Preferred field line color for {game_id} (e.g., white, yellow) [leave blank to skip]: "
                ).strip()
                line_color = response or None
            metadata = VideoMetadata(
                game_id=game_id,
                video_path=str(video_path),
                players_on_field=self._config.players_on_field,
            )
            events = self._process_video(metadata, line_color)
            write_events_csv(output_path, events)

    def _process_video(self, metadata: VideoMetadata, line_color: Optional[str]) -> List[ActionEvent]:
        _pitch = select_pitch_dimensions(metadata.players_on_field)
        _segments = segment_game_phases(metadata.video_path, line_color=line_color)
        _tracking = run_tracking(metadata.video_path)
        _tracking = assign_team_colors(metadata.video_path, _tracking)
        _tracking = assign_jersey_numbers(metadata.video_path, _tracking)
        _tracking = link_substitutions(_tracking)
        _possessions = infer_possessions(metadata.video_path, _tracking)
        # Phase 1 placeholder: return no events until detectors are wired in.
        # TODO: add homography mapping from image coordinates to pitch coordinates.
        # TODO: add team color classification + jersey OCR.
        # TODO: add action recognition (actions + subactions).
        return []
