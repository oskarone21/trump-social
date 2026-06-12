from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_engine.cli import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build archive-backed events from CNN posts and licensed NQ/MNQ bars."
    )
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--posts", default="data/processed/cnn_archive_posts.parquet")
    parser.add_argument("--market", required=True)
    parser.add_argument("--out", default="data/processed/real_events.parquet")
    parser.add_argument("--limit-posts", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    command = [
        "--config",
        args.config,
        "build-archive-events",
        "--posts",
        args.posts,
        "--market",
        args.market,
        "--out",
        args.out,
    ]
    if args.limit_posts is not None:
        command.extend(["--limit-posts", str(args.limit_posts)])
    main(command)
