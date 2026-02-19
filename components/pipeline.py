from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from components.csv_writer import write_events_csv
from components.action_recognition import recognize_actions
from components.event_builder import build_action_events
from components.jersey_ocr import assign_jersey_numbers
from components.substitution_linking import link_substitutions
from components.phase_segmentation import segment_game_phases
from components.pitch_mapper import PitchMapper
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
    team_names: Optional[dict[str, str]] = None
    enable_ocr: Optional[bool] = None


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self) -> None:
        videos = discover_videos(self._config.videos_dir)
        videos_list = list(videos)
        
        for video_path in (tqdm(videos_list, desc="Processing videos", unit="video") if tqdm else videos_list):
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
            
            team_names = self._config.team_names
            if team_names is None:
                us_color = input(f"Which jersey color is your team for {game_id} (e.g., blue, white): ").strip()
                them_color = input(f"Which jersey color is the opponent for {game_id} (e.g., red, yellow): ").strip()
                team_names = {us_color.lower(): "us", them_color.lower(): "them"}
            
            enable_ocr = self._config.enable_ocr
            if enable_ocr is None:
                response = input(f"Run jersey OCR for {game_id}? [Y/n]: ").strip().lower()
                enable_ocr = response != "n"

            metadata = VideoMetadata(
                game_id=game_id,
                video_path=str(video_path),
                players_on_field=self._config.players_on_field,
            )
            events = self._process_video(metadata, line_color, team_names, enable_ocr)
            write_events_csv(output_path, events)

    def _process_video(
        self,
        metadata: VideoMetadata,
        line_color: Optional[str],
        team_names: dict[str, str],
        enable_ocr: bool,
    ) -> List[ActionEvent]:
        _pitch = select_pitch_dimensions(metadata.players_on_field)
        
        steps = [
            ("Segmenting phases", lambda: segment_game_phases(metadata.video_path, line_color=line_color)),
            ("Tracking players/ball", lambda: run_tracking(metadata.video_path)),
            ("Assigning team colors", lambda: assign_team_colors(metadata.video_path, _tracking)),
            ("Linking substitutions", lambda: link_substitutions(_tracking)),
            ("Inferring possession", lambda: infer_possessions(metadata.video_path, _tracking)),
            ("Recognizing actions", lambda: recognize_actions(_tracking, (_pitch.length_m, _pitch.width_m))),
        ]

        if enable_ocr:
            steps.insert(3, ("Extracting jersey numbers", lambda: assign_jersey_numbers(metadata.video_path, _tracking)))
        
        _segments = None
        _tracking = None
        _tracking_copy = None
        _possessions = None
        _action_candidates = None
        
        for desc, func in (tqdm(steps, desc="Pipeline", unit="step") if tqdm else steps):
            if desc == "Segmenting phases":
                _segments = func()
            elif desc == "Tracking players/ball":
                _tracking = func()
            elif desc == "Assigning team colors":
                _tracking = func()
            elif desc == "Extracting jersey numbers":
                _tracking = func()
            elif desc == "Linking substitutions":
                _tracking = func()
            elif desc == "Inferring possession":
                _possessions = func()
            elif desc == "Recognizing actions":
                _action_candidates = func()
        
        _pitch_mapper = PitchMapper(
            metadata.video_path,
            _tracking,
            (_pitch.length_m, _pitch.width_m),
            line_color=line_color,
        )
        
        events = build_action_events(
            metadata.game_id,
            _action_candidates,
            _tracking,
            _pitch_mapper,
            _possessions,
            _segments,
            team_names,
        )
        return events
