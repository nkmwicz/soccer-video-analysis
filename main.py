from __future__ import annotations

import argparse
from pathlib import Path

from components.pipeline import Pipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soccer video analysis pipeline")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("videos"),
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--players-on-field",
        type=int,
        default=None,
        help="Total players on field (used to pick pitch size)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = Pipeline(
        PipelineConfig(
            videos_dir=args.videos_dir,
            data_dir=args.data_dir,
            players_on_field=args.players_on_field,
        )
    )
    pipeline.run()


if __name__ == "__main__":
    main()