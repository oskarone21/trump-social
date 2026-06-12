from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sentiment_engine.cli import main
from sentiment_engine.config import ensure_output_dirs, load_config

# Shared constants to avoid duplicated literal command fragments in this module.
COMMAND_CONFIG = "--config"
COMMAND_INGEST_ARCHIVE = "ingest-archive"
COMMAND_BUILD_ARCHIVE_EVENTS = "build-archive-events"
COMMAND_CHECK_ARCHIVE_FRESHNESS = "check-archive-freshness"
COMMAND_CHECK_PROVIDER_FRESHNESS = "check-provider-freshness"
COMMAND_INGEST_PROVIDER = "ingest-provider-posts"
COMMAND_INGEST_POSTS = "ingest-posts"
COMMAND_INGEST_MARKET_FILE = "ingest-market-file"
COMMAND_INGEST_MARKET = "ingest-market"
COMMAND_RUN_MODEL_TRAINING = "train-classifier"
COMMAND_SCORE_WHIPSAW = "score-whipsaw"
COMMAND_TUNE_WHIPSAW = "tune-whipsaw"
COMMAND_BACKTEST = "backtest"
COMMAND_INTERPRET = "interpret-results"
COMMAND_DASHBOARD = "dashboard"
DEFAULT_ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full research pipeline with explicit post + market sources and "
            "generate ML/DL-aware outputs where available."
        )
    )
    parser.add_argument("config", nargs="?", default="configs/research.yaml")
    parser.add_argument("--provider-source", default=None)
    parser.add_argument("--provider-name", default="truthsocial_provider_api")
    parser.add_argument("--provider-source-name", default="provider_posts")
    parser.add_argument("--provider-limit", type=int, default=None)
    parser.add_argument("--provider-api-key", default=None)
    parser.add_argument("--provider-api-key-env", default=None)
    parser.add_argument("--provider-api-key-header", default="x-api-key")
    parser.add_argument("--provider-header", action="append", default=[])
    parser.add_argument("--provider-out", default="data/processed/provider_posts.parquet")
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--archive-limit", type=int, default=None)
    parser.add_argument("--archive-out", default="data/processed/cnn_archive_posts.parquet")
    parser.add_argument("--posts-path", default=None)
    parser.add_argument("--use-archive", action="store_true")
    parser.add_argument("--no-post-freshness-check", action="store_true")

    parser.add_argument("--market-input", default=None)
    parser.add_argument("--market-source-name", default="databento_glbx_mdp3_ohlcv_1m")
    parser.add_argument("--market-symbol-root", default="NQ")
    parser.add_argument("--market-out", default="data/processed/market_bars.parquet")

    parser.add_argument("--events-out", default="data/processed/real_events.parquet")
    parser.add_argument("--events-limit", type=int, default=None)
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-whipsaw", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-interpret", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--skip-market-ingest", action="store_true")
    parser.add_argument("--require-real-input", action="store_true")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_output_dirs(config)

    posts_path = _resolve_posts_path(args, config)
    market_path = Path(args.market_out)

    if args.require_real_input and not args.provider_source and args.market_input is None:
        raise SystemExit(
            "--require-real-input requires --provider-source or --market-input and cannot fall back to fixtures."
        )

    if args.posts_path:
        posts_path = Path(args.posts_path)
        print(f"using existing posts path: {posts_path}")
    elif args.provider_source:
        _ingest_provider_posts(args)
        posts_path = Path(args.provider_out)
    elif args.use_archive:
        _ingest_archive_posts(args)
        posts_path = Path(args.archive_out)
        _check_archive_freshness_if_needed(args)
    else:
        _ingest_fixture_posts(args)
        posts_path = Path(config.paths.processed_dir / "posts.parquet")

    _ensure_market_data(args)
    market_path = Path(args.market_out)

    if not args.skip_events:
        _build_archive_events(posts_path, market_path, args)

    if not args.skip_train:
        _train(args.config)
    if not args.skip_whipsaw:
        _score_whipsaw(args.config)
    if not args.skip_tuning:
        _tune_whipsaw(args.config)
    if not args.skip_backtest:
        _backtest(args.config)
    if not args.skip_interpret:
        _interpret(args.config)
    if not args.skip_dashboard:
        _dashboard(args.config)


def _resolve_posts_path(args: argparse.Namespace, config) -> Path:
    if args.posts_path:
        return Path(args.posts_path)
    if args.provider_source:
        return Path(args.provider_out)
    if args.use_archive:
        return Path(args.archive_out)
    return Path(config.paths.processed_dir / "posts.parquet")



def _ingest_provider_posts(args: argparse.Namespace) -> None:
    command = [
        COMMAND_CONFIG,
        args.config,
        COMMAND_INGEST_PROVIDER,
        "--source",
        args.provider_source,
        "--provider-name",
        args.provider_name,
        "--source-name",
        args.provider_source_name,
        "--out",
        args.provider_out,
    ]
    if args.provider_limit is not None:
        command.extend(["--limit", str(args.provider_limit)])
    if args.provider_api_key is not None:
        command.extend(["--api-key", args.provider_api_key])
    if args.provider_api_key_env:
        command.extend(["--api-key-env", args.provider_api_key_env])
    command.extend(["--api-key-header", args.provider_api_key_header])
    for header in args.provider_header:
        command.extend(["--header", header])

    main(command)

    if not args.no_post_freshness_check:
        report_path = [
            COMMAND_CONFIG,
            args.config,
            COMMAND_CHECK_PROVIDER_FRESHNESS,
            "--source",
            args.provider_source,
            "--source-name",
            args.provider_source_name,
            "--source-provider",
            args.provider_name,
            "--posts",
            args.provider_out,
        ]
        if args.provider_api_key is not None:
            report_path.extend(["--api-key", args.provider_api_key])
        if args.provider_api_key_env:
            report_path.extend(["--api-key-env", args.provider_api_key_env])
        report_path.extend(["--api-key-header", args.provider_api_key_header])
        for header in args.provider_header:
            report_path.extend(["--header", header])
        main(report_path)


def _ingest_archive_posts(args: argparse.Namespace) -> None:
    command = [
        COMMAND_CONFIG,
        args.config,
        COMMAND_INGEST_ARCHIVE,
        "--url",
        args.archive_url,
        "--out",
        args.archive_out,
    ]
    if args.archive_limit is not None:
        command.extend(["--limit", str(args.archive_limit)])
    main(command)


def _ingest_fixture_posts(args: argparse.Namespace) -> None:
    main([COMMAND_CONFIG, args.config, COMMAND_INGEST_POSTS])


def _check_archive_freshness_if_needed(args: argparse.Namespace) -> None:
    if args.no_post_freshness_check:
        return
    command = [
        COMMAND_CONFIG,
        args.config,
        COMMAND_CHECK_ARCHIVE_FRESHNESS,
        "--url",
        args.archive_url,
        "--posts",
        args.archive_out,
    ]
    main(command)


def _ensure_market_data(args: argparse.Namespace) -> None:
    if args.skip_market_ingest:
        print("warning: skipping market ingest step (requires pre-existing data/processed/market_bars.parquet)")
        return
    if args.market_input:
        command = [
            COMMAND_CONFIG,
            args.config,
            COMMAND_INGEST_MARKET_FILE,
            "--input",
            args.market_input,
            "--source-name",
            args.market_source_name,
            "--symbol-root",
            args.market_symbol_root,
            "--out",
            args.market_out,
        ]
        main(command)
    else:
        main([COMMAND_CONFIG, args.config, COMMAND_INGEST_MARKET])


def _build_archive_events(posts_path: Path, market_path: Path, args: argparse.Namespace) -> None:
    command = [
        COMMAND_CONFIG,
        args.config,
        COMMAND_BUILD_ARCHIVE_EVENTS,
        "--posts",
        str(posts_path),
        "--market",
        str(market_path),
        "--out",
        args.events_out,
    ]
    if args.events_limit is not None:
        command.extend(["--limit-posts", str(args.events_limit)])
    main(command)


def _train(config: str) -> None:
    main([COMMAND_CONFIG, config, COMMAND_RUN_MODEL_TRAINING])


def _score_whipsaw(config: str) -> None:
    main([COMMAND_CONFIG, config, COMMAND_SCORE_WHIPSAW])


def _tune_whipsaw(config: str) -> None:
    main([COMMAND_CONFIG, config, COMMAND_TUNE_WHIPSAW])


def _backtest(config: str) -> None:
    main([COMMAND_CONFIG, config, COMMAND_BACKTEST])


def _interpret(config: str) -> None:
    main([COMMAND_CONFIG, config, COMMAND_INTERPRET])


def _dashboard(config: str) -> None:
    main([COMMAND_CONFIG, config, COMMAND_DASHBOARD])


if __name__ == "__main__":
    run_pipeline(parse_args())
