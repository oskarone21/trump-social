from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.baselines import build_labeled_events
from sentiment_engine.models.walk_forward import evaluate_walk_forward_classifiers
from sentiment_engine.research.events import build_event_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run expanding walk-forward classifier validation."
    )
    parser.add_argument("config", nargs="?", default="configs/research.yaml")
    parser.add_argument("--events", default=None)
    parser.add_argument("--min-train-rows", type=int, default=3)
    parser.add_argument("--test-window-rows", type=int, default=1)
    parser.add_argument("--step-rows", type=int, default=1)
    parser.add_argument("--embargo-rows", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    labeled = _load_or_build_labeled_events(config, args.events)
    report = evaluate_walk_forward_classifiers(
        labeled,
        report_dir=config.paths.report_dir,
        seed=config.project.seed,
        min_train_rows=args.min_train_rows,
        test_window_rows=args.test_window_rows,
        step_rows=args.step_rows,
        embargo_rows=args.embargo_rows,
    )
    print(
        "walk-forward complete: "
        f"status={report['status']}, folds={report['fold_count']}"
    )
    print(f"walk-forward report: {config.paths.report_dir / 'walk_forward_report.json'}")


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
