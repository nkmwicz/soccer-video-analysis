from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from components.types import ActionEvent


CSV_COLUMNS = [
    "game_id",
    "event_id",
    "period",
    "phase",
    "possession_id",
    "team_color",
    "team_name",
    "player_number",
    "player_track_id",
    "ball_owner_track_id",
    "action",
    "subaction",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "confidence",
]


def write_events_csv(output_path: Path, events: Iterable[ActionEvent]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "game_id": event.game_id,
                    "event_id": event.event_id,
                    "period": event.period,
                    "phase": event.phase,
                    "possession_id": event.possession_id,
                    "team_color": event.team_color,
                    "team_name": event.team_name,
                    "player_number": event.player_number,
                    "player_track_id": event.player_track_id,
                    "ball_owner_track_id": event.ball_owner_track_id,
                    "action": event.action,
                    "subaction": event.subaction,
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "start_time_s": f"{event.start_time_s:.3f}",
                    "end_time_s": f"{event.end_time_s:.3f}",
                    "start_x": f"{event.start_x:.3f}",
                    "start_y": f"{event.start_y:.3f}",
                    "end_x": f"{event.end_x:.3f}",
                    "end_y": f"{event.end_y:.3f}",
                    "confidence": f"{event.confidence:.3f}",
                }
            )
