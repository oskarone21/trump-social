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
from sentiment_engine.models.finbert import DEFAULT_FINBERT_MODEL, score_finbert_sentiment
from sentiment_engine.research.events import build_event_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FinBERT inference over event text and write an audit report."
    )
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--events", default=None)
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--model-name", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use cached Hugging Face files only; do not attempt network access.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    events = _load_or_build_events(config, args.events)
    if args.limit is not None:
        events = events.head(args.limit).copy()
    report = score_finbert_sentiment(
        build_labeled_events(events),
        report_dir=config.paths.report_dir,
        scores_path=Path(args.scores_out) if args.scores_out else None,
        model_name=args.model_name,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    print(
        "finbert inference complete: "
        f"status={report['status']}, scored_rows={report.get('scored_rows', 0)}"
    )
    print(f"finbert report: {config.paths.report_dir / 'finbert_inference_report.json'}")


def _load_or_build_events(config, events_path: str | None) -> pd.DataFrame:
    if events_path:
        return pd.read_parquet(events_path)
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    return build_event_dataset(posts, bars, config).events


if __name__ == "__main__":
    main()
