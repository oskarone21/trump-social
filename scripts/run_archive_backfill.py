from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_engine.cli import main

CONFIG_FLAG = "--config"
INGEST_ARCHIVE_COMMAND = "ingest-archive"
CHECK_FRESHNESS_COMMAND = "check-archive-freshness"
URL_FLAG = "--url"
LIMIT_FLAG = "--limit"
OUT_FLAG = "--out"
POSTS_FLAG = "--posts"
STALE_AFTER_FLAG = "--stale-after-minutes"
HTTP_PREFIXES = ("http://", "https://")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Trump Truth Social posts from the configured archive source."
    )
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/processed/cnn_archive_posts.parquet")
    parser.add_argument("--stale-after-minutes", type=int, default=30)
    parser.add_argument(
        "--skip-freshness",
        action="store_true",
        help="Skip remote HEAD freshness check. Useful for local fixture sources.",
    )
    return parser.parse_args()


def build_ingest_command(args: argparse.Namespace) -> list[str]:
    command = [CONFIG_FLAG, args.config, INGEST_ARCHIVE_COMMAND, OUT_FLAG, args.out]
    if args.url is not None:
        command.extend([URL_FLAG, args.url])
    if args.limit is not None:
        command.extend([LIMIT_FLAG, str(args.limit)])
    return command


def build_freshness_command(args: argparse.Namespace) -> list[str]:
    command = [
        CONFIG_FLAG,
        args.config,
        CHECK_FRESHNESS_COMMAND,
        POSTS_FLAG,
        args.out,
        STALE_AFTER_FLAG,
        str(args.stale_after_minutes),
    ]
    if args.url is not None:
        command.extend([URL_FLAG, args.url])
    return command


def should_run_freshness_check(args: argparse.Namespace) -> bool:
    if args.skip_freshness:
        return False
    if args.url is None:
        return True
    return args.url.startswith(HTTP_PREFIXES)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(build_ingest_command(parsed_args))
    if should_run_freshness_check(parsed_args):
        main(build_freshness_command(parsed_args))
