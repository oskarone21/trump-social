from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.baselines import build_labeled_events, train_tradeability_baselines
from sentiment_engine.research.events import build_event_dataset
from sentiment_engine.utils.io import write_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train tradeability ML/DL baselines from labeled events."
    )
    parser.add_argument("config", nargs="?", default="configs/research.yaml")
    parser.add_argument("--events", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    labeled = _load_or_build_labeled_events(config, args.events)
    if args.out:
        write_dataframe(labeled, args.out)
    report = train_tradeability_baselines(
        labeled,
        report_dir=config.paths.report_dir,
        model_dir=config.paths.model_dir,
        config_path=args.config,
        seed=config.project.seed,
    )
    print(
        "model training complete: "
        f"rows={report['row_count']}, temporal_holdout_rows={report['test_rows']}"
    )
    print(f"classifier report: {config.paths.report_dir / 'classifier_baseline_report.json'}")


def _load_or_build_labeled_events(config, events_path: str | None) -> pd.DataFrame:
    if events_path:
        events = pd.read_parquet(events_path)
    else:
        posts = load_fixture_posts(config.paths.posts_fixture)
        bars = load_market_csv(config.paths.market_fixture)
        events = build_event_dataset(posts, bars, config).events
    return build_labeled_events(events)


if __name__ == "__main__":
    main()
