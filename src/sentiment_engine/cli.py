from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.ingestion.market_csv import audit_market_bars, load_market_csv
from sentiment_engine.ingestion.posts_fixture import audit_posts, load_fixture_posts, posts_to_frame
from sentiment_engine.research.event_study import build_event_study_report
from sentiment_engine.research.events import build_event_dataset
from sentiment_engine.utils.io import write_dataframe, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sentiment-engine")
    parser.add_argument("--config", default="configs/research.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest-posts")
    subparsers.add_parser("ingest-market")
    subparsers.add_parser("build-events")
    subparsers.add_parser("event-study")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_output_dirs(config)
    if args.command == "ingest-posts":
        _ingest_posts(config)
    elif args.command == "ingest-market":
        _ingest_market(config)
    elif args.command == "build-events":
        _build_events(config)
    elif args.command == "event-study":
        _event_study(config)


def _ingest_posts(config) -> None:
    posts = load_fixture_posts(config.paths.posts_fixture)
    frame = posts_to_frame(posts)
    write_dataframe(frame, config.paths.processed_dir / "posts.parquet")
    write_json(config.paths.report_dir / "post_ingestion_audit.json", audit_posts(posts))
    print(f"ingested {len(posts)} posts")


def _ingest_market(config) -> None:
    bars = load_market_csv(config.paths.market_fixture)
    write_dataframe(bars, config.paths.processed_dir / "market_bars.parquet")
    write_json(config.paths.report_dir / "market_ingestion_audit.json", audit_market_bars(bars))
    print(f"ingested {len(bars)} market bars")


def _build_events(config) -> None:
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    result = build_event_dataset(posts, bars, config)
    write_dataframe(result.events, config.paths.processed_dir / "events.parquet")
    write_json(config.paths.report_dir / "event_build_audit.json", {"skipped_posts": result.skipped_posts})
    print(f"built {len(result.events)} events")


def _event_study(config) -> None:
    events_path = config.paths.processed_dir / "events.parquet"
    if not Path(events_path).exists():
        _build_events(config)
    import pandas as pd

    events = pd.read_parquet(events_path)
    write_json(config.paths.report_dir / "event_study.json", build_event_study_report(events))
    print(f"event study written for {len(events)} events")


if __name__ == "__main__":
    main()
