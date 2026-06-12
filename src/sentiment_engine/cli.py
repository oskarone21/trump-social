from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.backtest.simulator import run_kill_switch_backtest
from sentiment_engine.ingestion.market_csv import audit_market_bars, load_market_csv
from sentiment_engine.ingestion.posts_fixture import audit_posts, load_fixture_posts, posts_to_frame
from sentiment_engine.models.baselines import build_labeled_events, train_tradeability_baselines
from sentiment_engine.models.tuning import tune_whipsaw_parameters
from sentiment_engine.models.whipsaw import build_whipsaw_report, score_whipsaw_events
from sentiment_engine.live.dashboard import build_dashboard
from sentiment_engine.live.signal_engine import latest_signal_from_scores
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
    subparsers.add_parser("label-assist")
    subparsers.add_parser("train-classifier")
    subparsers.add_parser("score-whipsaw")
    subparsers.add_parser("tune-whipsaw")
    subparsers.add_parser("backtest")
    subparsers.add_parser("dashboard")
    subparsers.add_parser("latest-signal")
    subparsers.add_parser("run-full")
    subparsers.add_parser("serve")
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
    elif args.command == "label-assist":
        _label_assist(config)
    elif args.command == "train-classifier":
        _train_classifier(config, args.config)
    elif args.command == "score-whipsaw":
        _score_whipsaw(config)
    elif args.command == "tune-whipsaw":
        _tune_whipsaw(config)
    elif args.command == "backtest":
        _backtest(config)
    elif args.command == "dashboard":
        _dashboard(config)
    elif args.command == "latest-signal":
        _latest_signal(config)
    elif args.command == "run-full":
        _run_full(config, args.config)
    elif args.command == "serve":
        _serve(args.config)


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


def _label_assist(config) -> None:
    events = _read_or_build_events(config)
    labeled = build_labeled_events(events)
    write_dataframe(labeled, config.paths.processed_dir / "labeled_events.parquet")
    label_audit = {
        "row_count": int(len(labeled)),
        "rule_sentiment_counts": labeled["rule_sentiment_label"].value_counts().to_dict(),
        "rule_tradeability_counts": labeled["rule_tradeability_label"].value_counts().to_dict(),
        "target_tradeability_counts": labeled["tradeability_label"].value_counts().to_dict(),
    }
    write_json(config.paths.report_dir / "label_audit.json", label_audit)
    print(f"labeled {len(labeled)} events")


def _train_classifier(config, config_path: str) -> None:
    labeled_path = config.paths.processed_dir / "labeled_events.parquet"
    if not Path(labeled_path).exists():
        _label_assist(config)
    import pandas as pd

    labeled = pd.read_parquet(labeled_path)
    report = train_tradeability_baselines(
        labeled,
        report_dir=config.paths.report_dir,
        model_dir=config.paths.model_dir,
        config_path=config_path,
        seed=config.project.seed,
    )
    print(f"classifier baselines evaluated on {report['test_rows']} temporal holdout rows")


def _score_whipsaw(config) -> None:
    labeled = _read_or_build_labeled_events(config)
    scored = score_whipsaw_events(labeled, config)
    write_dataframe(scored, config.paths.processed_dir / "whipsaw_scores.parquet")
    write_json(config.paths.report_dir / "whipsaw_report.json", build_whipsaw_report(scored))
    print(f"scored whipsaw risk for {len(scored)} events")


def _backtest(config) -> None:
    scored = _read_or_build_whipsaw_scores(config)
    _trades, report = run_kill_switch_backtest(config.paths.trades_fixture, scored, config)
    print(
        "backtest complete: "
        f"before={report['net_pnl_before_usd']}, after={report['net_pnl_after_usd']}"
    )


def _tune_whipsaw(config) -> None:
    scored = _read_or_build_whipsaw_scores(config)
    report = tune_whipsaw_parameters(
        scored,
        report_path=config.paths.report_dir / "whipsaw_tuning_report.json",
        seed=config.project.seed,
    )
    print(
        "whipsaw tuning complete: "
        f"trials={report['n_trials']}, holdout_soft_recall="
        f"{report['holdout_metrics_best']['soft_risk']['recall']}"
    )


def _dashboard(config) -> None:
    scored = _read_or_build_whipsaw_scores(config)
    signal = latest_signal_from_scores(scored, config)
    whipsaw_report = _read_json_report(config.paths.report_dir / "whipsaw_report.json")
    backtest_report = _read_json_report(config.paths.report_dir / "backtest_report.json")
    output = build_dashboard(
        signal=signal,
        whipsaw_report=whipsaw_report,
        backtest_report=backtest_report,
        scored_events=scored,
        output_path=config.paths.report_dir / "dashboard.html",
    )
    print(f"dashboard written to {output}")


def _latest_signal(config) -> None:
    scored = _read_or_build_whipsaw_scores(config)
    signal = latest_signal_from_scores(scored, config)
    write_json(config.paths.report_dir / "latest_signal.json", signal.model_dump(mode="json"))
    print(f"latest signal: {signal.direction_signal} / {signal.kill_switch['action']}")


def _run_full(config, config_path: str) -> None:
    _ingest_posts(config)
    _ingest_market(config)
    _build_events(config)
    _event_study(config)
    _label_assist(config)
    _train_classifier(config, config_path)
    _score_whipsaw(config)
    _tune_whipsaw(config)
    _backtest(config)
    _dashboard(config)
    print("full pipeline completed")


def _serve(config_path: str) -> None:
    from sentiment_engine.live.service import create_app

    create_app(config_path)
    print("FastAPI app created. Run with: uvicorn sentiment_engine.live.service:create_app --factory")


def _read_or_build_events(config):
    events_path = config.paths.processed_dir / "events.parquet"
    if not Path(events_path).exists():
        _build_events(config)
    import pandas as pd

    return pd.read_parquet(events_path)


def _read_or_build_labeled_events(config):
    labeled_path = config.paths.processed_dir / "labeled_events.parquet"
    if not Path(labeled_path).exists():
        _label_assist(config)
    import pandas as pd

    return pd.read_parquet(labeled_path)


def _read_or_build_whipsaw_scores(config):
    scored_path = config.paths.processed_dir / "whipsaw_scores.parquet"
    if not Path(scored_path).exists():
        _score_whipsaw(config)
    import pandas as pd

    return pd.read_parquet(scored_path)


def _read_json_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing report {path}; run the upstream command first")
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
