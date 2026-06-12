from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.backtest.simulator import run_kill_switch_backtest
from sentiment_engine.ingestion.market_csv import audit_market_bars, load_market_csv
from sentiment_engine.ingestion.market_files import DATABENTO_OHLCV_SOURCE, load_market_file
from sentiment_engine.ingestion.posts_cnn_archive import (
    archive_posts_to_frame,
    audit_cnn_archive_posts,
    load_cnn_archive_posts,
)
from sentiment_engine.ingestion.posts_fixture import audit_posts, load_fixture_posts, posts_to_frame
from sentiment_engine.labels.review import (
    LABEL_VERSION_DEFAULT,
    build_label_queue,
    load_reviewed_labels,
)
from sentiment_engine.models.baselines import build_labeled_events, train_tradeability_baselines
from sentiment_engine.models.tuning import tune_whipsaw_parameters
from sentiment_engine.models.whipsaw import build_whipsaw_report, score_whipsaw_events
from sentiment_engine.live.dashboard import build_dashboard
from sentiment_engine.live.signal_engine import latest_signal_from_scores
from sentiment_engine.research.archive_events import build_archive_event_dataset
from sentiment_engine.research.event_study import build_event_study_report
from sentiment_engine.research.events import build_event_dataset
from sentiment_engine.research.interpretation import write_interpretation_report
from sentiment_engine.utils.io import write_dataframe, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sentiment-engine")
    parser.add_argument("--config", default="configs/research.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest-posts")
    archive_parser = subparsers.add_parser("ingest-archive")
    archive_parser.add_argument("--url", default=None)
    archive_parser.add_argument("--limit", type=int, default=None)
    archive_parser.add_argument("--out", default=None)
    subparsers.add_parser("ingest-market")
    market_file_parser = subparsers.add_parser("ingest-market-file")
    market_file_parser.add_argument("--input", required=True)
    market_file_parser.add_argument("--source-name", default=DATABENTO_OHLCV_SOURCE)
    market_file_parser.add_argument("--symbol-root", choices=["NQ", "MNQ"], default="NQ")
    market_file_parser.add_argument("--contract-symbol", default=None)
    market_file_parser.add_argument("--continuous-symbol", default=None)
    market_file_parser.add_argument("--out", default=None)
    subparsers.add_parser("build-events")
    archive_events_parser = subparsers.add_parser("build-archive-events")
    archive_events_parser.add_argument("--posts", default=None)
    archive_events_parser.add_argument("--market", default=None)
    archive_events_parser.add_argument("--out", default=None)
    archive_events_parser.add_argument("--limit-posts", type=int, default=None)
    subparsers.add_parser("event-study")
    subparsers.add_parser("label-assist")
    label_queue_parser = subparsers.add_parser("export-label-queue")
    label_queue_parser.add_argument("--events", default=None)
    label_queue_parser.add_argument("--out", default=None)
    label_queue_parser.add_argument("--limit", type=int, default=None)
    reviewed_parser = subparsers.add_parser("import-reviewed-labels")
    reviewed_parser.add_argument("--input", required=True)
    reviewed_parser.add_argument("--out", default=None)
    reviewed_parser.add_argument("--label-version", default=LABEL_VERSION_DEFAULT)
    subparsers.add_parser("train-classifier")
    subparsers.add_parser("score-whipsaw")
    subparsers.add_parser("tune-whipsaw")
    subparsers.add_parser("backtest")
    subparsers.add_parser("interpret-results")
    subparsers.add_parser("dashboard")
    subparsers.add_parser("latest-signal")
    subparsers.add_parser("run-full")
    subparsers.add_parser("serve")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_output_dirs(config)
    if args.command == "ingest-posts":
        _ingest_posts(config)
    elif args.command == "ingest-archive":
        _ingest_archive(config, args.url, args.limit, args.out)
    elif args.command == "ingest-market":
        _ingest_market(config)
    elif args.command == "ingest-market-file":
        _ingest_market_file(
            config,
            args.input,
            args.source_name,
            args.symbol_root,
            args.contract_symbol,
            args.continuous_symbol,
            args.out,
        )
    elif args.command == "build-events":
        _build_events(config)
    elif args.command == "build-archive-events":
        _build_archive_events(config, args.posts, args.market, args.out, args.limit_posts)
    elif args.command == "event-study":
        _event_study(config)
    elif args.command == "label-assist":
        _label_assist(config)
    elif args.command == "export-label-queue":
        _export_label_queue(config, args.events, args.out, args.limit)
    elif args.command == "import-reviewed-labels":
        _import_reviewed_labels(config, args.input, args.out, args.label_version)
    elif args.command == "train-classifier":
        _train_classifier(config, args.config)
    elif args.command == "score-whipsaw":
        _score_whipsaw(config)
    elif args.command == "tune-whipsaw":
        _tune_whipsaw(config)
    elif args.command == "backtest":
        _backtest(config)
    elif args.command == "interpret-results":
        _interpret_results(config)
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


def _ingest_archive(config, url: str | None, limit: int | None, out: str | None) -> None:
    archive_url = url or config.sources["posts"]["stiles_archive"]["latest_archive_url"]
    posts = load_cnn_archive_posts(archive_url, limit=limit)
    output_path = Path(out) if out else config.paths.processed_dir / "cnn_archive_posts.parquet"
    write_dataframe(archive_posts_to_frame(posts), output_path)
    write_json(
        config.paths.report_dir / "cnn_archive_ingestion_audit.json",
        audit_cnn_archive_posts(posts, source=archive_url),
    )
    print(f"ingested {len(posts)} archive posts from {archive_url}")


def _ingest_market(config) -> None:
    bars = load_market_csv(config.paths.market_fixture)
    write_dataframe(bars, config.paths.processed_dir / "market_bars.parquet")
    write_json(config.paths.report_dir / "market_ingestion_audit.json", audit_market_bars(bars))
    print(f"ingested {len(bars)} market bars")


def _ingest_market_file(
    config,
    input_path: str,
    source_name: str,
    symbol_root: str,
    contract_symbol: str | None,
    continuous_symbol: str | None,
    output_path: str | None,
) -> None:
    bars = load_market_file(
        input_path,
        source_name=source_name,
        symbol_root=symbol_root,
        contract_symbol=contract_symbol,
        continuous_symbol=continuous_symbol,
    )
    target = (
        Path(output_path) if output_path else config.paths.processed_dir / "market_bars.parquet"
    )
    write_dataframe(bars, target)
    write_json(config.paths.report_dir / "market_ingestion_audit.json", audit_market_bars(bars))
    print(f"ingested {len(bars)} market bars from {input_path}")


def _build_events(config) -> None:
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    result = build_event_dataset(posts, bars, config)
    write_dataframe(result.events, config.paths.processed_dir / "events.parquet")
    write_json(
        config.paths.report_dir / "event_build_audit.json",
        {"skipped_posts": result.skipped_posts},
    )
    print(f"built {len(result.events)} events")


def _build_archive_events(
    config,
    posts_path: str | None,
    market_path: str | None,
    output_path: str | None,
    limit_posts: int | None,
) -> None:
    posts_source = (
        Path(posts_path) if posts_path else config.paths.processed_dir / "cnn_archive_posts.parquet"
    )
    market_source = (
        Path(market_path) if market_path else config.paths.processed_dir / "market_bars.parquet"
    )
    target = (
        Path(output_path) if output_path else config.paths.processed_dir / "real_events.parquet"
    )
    result = build_archive_event_dataset(
        posts_path=posts_source,
        market_path=market_source,
        config=config,
        limit_posts=limit_posts,
    )
    write_dataframe(result.events, target)
    write_json(config.paths.report_dir / "real_event_build_audit.json", result.audit)
    print(
        "built archive events: "
        f"events={len(result.events)}, skipped={len(result.skipped_posts)}"
    )


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


def _export_label_queue(
    config, events_path: str | None, output_path: str | None, limit: int | None
) -> None:
    import pandas as pd

    source = (
        Path(events_path) if events_path else config.paths.processed_dir / "labeled_events.parquet"
    )
    if not source.exists():
        _label_assist(config)
    events = pd.read_parquet(source)
    queue = build_label_queue(events, limit=limit)
    target = Path(output_path) if output_path else config.paths.interim_dir / "label_queue.csv"
    write_dataframe(queue, target)
    write_json(
        config.paths.report_dir / "label_queue_audit.json",
        {
            "row_count": int(len(queue)),
            "output_path": str(target),
            "source_path": str(source),
            "post_event_target_columns_excluded": True,
        },
    )
    print(f"exported {len(queue)} label-review rows to {target}")


def _import_reviewed_labels(
    config, input_path: str, output_path: str | None, label_version: str
) -> None:
    labels, audit = load_reviewed_labels(input_path, label_version=label_version)
    target = (
        Path(output_path) if output_path else config.paths.processed_dir / "human_labels.parquet"
    )
    write_dataframe(labels, target)
    write_json(config.paths.report_dir / "human_label_audit.json", audit)
    print(f"imported {len(labels)} reviewed labels to {target}")


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
    interpretation_report = _read_optional_json_report(
        config.paths.report_dir / "research_interpretation.json"
    )
    output = build_dashboard(
        signal=signal,
        whipsaw_report=whipsaw_report,
        backtest_report=backtest_report,
        interpretation_report=interpretation_report,
        scored_events=scored,
        output_path=config.paths.report_dir / "dashboard.html",
    )
    print(f"dashboard written to {output}")


def _interpret_results(config) -> None:
    reports = {
        "classifier": _read_optional_json_report(
            config.paths.report_dir / "classifier_baseline_report.json"
        ),
        "post": _read_optional_json_report(config.paths.report_dir / "post_ingestion_audit.json"),
        "archive": _read_optional_json_report(
            config.paths.report_dir / "cnn_archive_ingestion_audit.json"
        ),
        "market": _read_optional_json_report(
            config.paths.report_dir / "market_ingestion_audit.json"
        ),
        "whipsaw": _read_optional_json_report(config.paths.report_dir / "whipsaw_report.json"),
        "backtest": _read_optional_json_report(config.paths.report_dir / "backtest_report.json"),
        "human_labels": _read_optional_json_report(
            config.paths.report_dir / "human_label_audit.json"
        ),
    }
    report = write_interpretation_report(reports=reports, report_dir=config.paths.report_dir)
    print(f"interpretation written: status={report['status']}")


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
    _interpret_results(config)
    _dashboard(config)
    print("full pipeline completed")


def _serve(config_path: str) -> None:
    from sentiment_engine.live.service import create_app

    create_app(config_path)
    print(
        "FastAPI app created. Run with: uvicorn "
        "sentiment_engine.live.service:create_app --factory"
    )


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


def _read_optional_json_report(path: Path):
    if not path.exists():
        return None
    return _read_json_report(path)


if __name__ == "__main__":
    main()
