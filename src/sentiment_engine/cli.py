from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.backtest.simulator import run_kill_switch_backtest
from sentiment_engine.ingestion.archive_monitor import (
    STALE_AFTER_MINUTES_DEFAULT,
    build_archive_freshness_report,
)
from sentiment_engine.ingestion.databento_provider import (
    DATABENTO_DATASET_DEFAULT,
    DATABENTO_SCHEMA_DEFAULT,
    DATABENTO_STYPE_IN_DEFAULT,
    DATABENTO_STYPE_OUT_DEFAULT,
    download_databento_ohlcv,
    parse_symbol_list,
)
from sentiment_engine.ingestion.market_csv import audit_market_bars, load_market_csv
from sentiment_engine.ingestion.market_files import DATABENTO_OHLCV_SOURCE, load_market_file
from sentiment_engine.ingestion.posts_cnn_archive import (
    archive_posts_to_frame,
    audit_cnn_archive_posts,
    load_cnn_archive_posts,
)
from sentiment_engine.ingestion.posts_trumpstruth_feed import (
    TRUMPSTRUTH_PROVIDER_NAME,
    TRUMPSTRUTH_SOURCE_NAME,
    audit_trumpstruth_feed_posts,
    trumpstruth_feed_posts_to_frame,
    load_trumpstruth_feed_posts,
)
from sentiment_engine.ingestion.posts_external_provider import (
    TRUTHSOCIAL_PROVIDER,
    audit_truthsocial_provider_posts,
    load_truthsocial_provider_posts,
    truthsocial_provider_posts_to_frame,
)
from sentiment_engine.ingestion.truthsocial_browser import (
    DEFAULT_CANONICAL_OUT,
    DEFAULT_POLL_SECONDS,
    DEFAULT_REPORT_OUT,
    DEFAULT_STALE_AFTER_SECONDS,
    BrowserScraperSettings,
    run_truthsocial_browser_scrape_once,
    run_truthsocial_fixture_scrape,
)
from sentiment_engine.ingestion.provider_monitor import (
    DEFAULT_STALE_AFTER_MINUTES,
    build_provider_freshness_report,
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
from sentiment_engine.live.signal_engine import latest_signal_from_scores, signal_from_scored_event
from sentiment_engine.research.archive_events import build_archive_event_dataset
from sentiment_engine.research.event_study import build_event_study_report
from sentiment_engine.research.events import build_event_dataset
from sentiment_engine.research.interpretation import write_interpretation_report
from sentiment_engine.utils.io import write_dataframe, write_json


def main(argv: list[str] | None = None) -> None:
    raw_args, config_path = _extract_config_arg(
        sys.argv[1:] if argv is None else argv,
        default="configs/research.yaml",
    )
    parser = argparse.ArgumentParser(prog="sentiment-engine")
    parser.add_argument("--config", default=config_path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest-posts")
    archive_parser = subparsers.add_parser("ingest-archive")
    archive_parser.add_argument("--url", default=None)
    archive_parser.add_argument("--limit", type=int, default=None)
    archive_parser.add_argument("--out", default=None)
    provider_parser = subparsers.add_parser("ingest-provider-posts")
    provider_parser.add_argument("--source", required=True)
    provider_parser.add_argument("--provider-name", default=TRUTHSOCIAL_PROVIDER)
    provider_parser.add_argument("--source-name", default="truthsocial_provider_dump")
    provider_parser.add_argument("--api-key")
    provider_parser.add_argument("--api-key-env")
    provider_parser.add_argument(
        "--api-key-header",
        default="x-api-key",
        help="Header name for --api-key or --api-key-env values.",
    )
    provider_parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional HTTP headers for remote provider sources (key:value).",
    )
    provider_parser.add_argument("--out", default=None)
    provider_parser.add_argument("--limit", type=int, default=None)
    browser_parser = subparsers.add_parser("scrape-truthsocial-live")
    mode_group = browser_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--once", action="store_true")
    mode_group.add_argument("--watch", action="store_true")
    browser_parser.add_argument("--fixture", default=None)
    browser_parser.add_argument("--max-iterations", type=int, default=None)
    browser_parser.add_argument("--poll-seconds", type=int, default=None)
    browser_parser.add_argument("--stale-after-seconds", type=int, default=None)
    browser_parser.add_argument("--storage-state", default=None)
    browser_parser.add_argument("--out", default=None)
    browser_parser.add_argument("--report-out", default=None)
    browser_parser.add_argument("--headful", action="store_true")
    trumpstruth_parser = subparsers.add_parser("ingest-trumpstruth-feed")
    trumpstruth_parser.add_argument("--url", default="https://www.trumpstruth.org/feed")
    trumpstruth_parser.add_argument(
        "--start-date",
        default=None,
        help="Optional YYYY-MM-DD filter (start_date) for feed endpoint.",
    )
    trumpstruth_parser.add_argument(
        "--end-date",
        default=None,
        help="Optional YYYY-MM-DD filter (end_date) for feed endpoint.",
    )
    trumpstruth_parser.add_argument("--provider-name", default=TRUMPSTRUTH_PROVIDER_NAME)
    trumpstruth_parser.add_argument("--source-name", default=TRUMPSTRUTH_SOURCE_NAME)
    trumpstruth_parser.add_argument("--out", default=None)
    trumpstruth_parser.add_argument("--limit", type=int, default=None)
    freshness_parser = subparsers.add_parser("check-archive-freshness")
    freshness_parser.add_argument("--url", default=None)
    freshness_parser.add_argument("--posts", default=None)
    freshness_parser.add_argument(
        "--stale-after-minutes",
        type=int,
        default=STALE_AFTER_MINUTES_DEFAULT,
    )
    provider_freshness_parser = subparsers.add_parser("check-provider-freshness")
    provider_freshness_parser.add_argument("--source", required=True)
    provider_freshness_parser.add_argument("--source-name", default="truthsocial_provider_dump")
    provider_freshness_parser.add_argument(
        "--source-provider",
        default=TRUTHSOCIAL_PROVIDER,
    )
    provider_freshness_parser.add_argument("--posts", default=None)
    provider_freshness_parser.add_argument(
        "--stale-after-minutes",
        type=int,
        default=DEFAULT_STALE_AFTER_MINUTES,
    )
    provider_freshness_parser.add_argument("--api-key")
    provider_freshness_parser.add_argument("--api-key-env")
    provider_freshness_parser.add_argument(
        "--api-key-header",
        default="x-api-key",
        help="Header name for --api-key or --api-key-env values.",
    )
    provider_freshness_parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional HTTP headers for remote provider checks (key:value).",
    )
    subparsers.add_parser("ingest-market")
    market_file_parser = subparsers.add_parser("ingest-market-file")
    market_file_parser.add_argument("--input", required=True)
    market_file_parser.add_argument("--source-name", default=DATABENTO_OHLCV_SOURCE)
    market_file_parser.add_argument("--symbol-root", choices=["NQ", "MNQ"], default="NQ")
    market_file_parser.add_argument("--contract-symbol", default=None)
    market_file_parser.add_argument("--continuous-symbol", default=None)
    market_file_parser.add_argument("--out", default=None)
    databento_parser = subparsers.add_parser("download-databento-market")
    databento_parser.add_argument("--start", required=True)
    databento_parser.add_argument("--end", default=None)
    databento_parser.add_argument("--symbols", default="NQ.c.0")
    databento_parser.add_argument("--dataset", default=DATABENTO_DATASET_DEFAULT)
    databento_parser.add_argument("--schema", default=DATABENTO_SCHEMA_DEFAULT)
    databento_parser.add_argument("--stype-in", default=DATABENTO_STYPE_IN_DEFAULT)
    databento_parser.add_argument("--stype-out", default=DATABENTO_STYPE_OUT_DEFAULT)
    databento_parser.add_argument("--limit", type=int, default=None)
    databento_parser.add_argument("--symbol-root", choices=["NQ", "MNQ"], default="NQ")
    databento_parser.add_argument("--contract-symbol", default=None)
    databento_parser.add_argument("--continuous-symbol", default=None)
    databento_parser.add_argument("--out", default=None)
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
    args = parser.parse_args(raw_args)
    config = load_config(args.config)
    ensure_output_dirs(config)
    if args.command == "ingest-posts":
        _ingest_posts(config)
    elif args.command == "ingest-archive":
        _ingest_archive(config, args.url, args.limit, args.out)
    elif args.command == "ingest-provider-posts":
        _ingest_provider_posts(
            config,
            args.source,
            args.provider_name,
            args.source_name,
            args.limit,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            api_key_header=args.api_key_header,
            headers=args.header,
            out=args.out,
        )
    elif args.command == "scrape-truthsocial-live":
        _scrape_truthsocial_live(
            config,
            fixture=args.fixture,
            watch=args.watch,
            max_iterations=args.max_iterations,
            poll_seconds=args.poll_seconds,
            stale_after_seconds=args.stale_after_seconds,
            storage_state=args.storage_state,
            out=args.out,
            report_out=args.report_out,
            headless=not args.headful,
        )
    elif args.command == "ingest-trumpstruth-feed":
        _ingest_trumpstruth_feed(
            config,
            args.url,
            args.start_date,
            args.end_date,
            args.provider_name,
            args.source_name,
            args.limit,
            args.out,
        )
    elif args.command == "check-archive-freshness":
        _check_archive_freshness(config, args.url, args.posts, args.stale_after_minutes)
    elif args.command == "check-provider-freshness":
        _check_provider_freshness(
            config,
            args.source,
            args.source_name,
            args.source_provider,
            args.posts,
            args.stale_after_minutes,
            args.api_key,
            args.api_key_env,
            args.api_key_header,
            args.header,
        )
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
    elif args.command == "download-databento-market":
        _download_databento_market(
            config,
            args.start,
            args.end,
            args.symbols,
            args.dataset,
            args.schema,
            args.stype_in,
            args.stype_out,
            args.limit,
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


def _ingest_provider_posts(
    config,
    source: str,
    provider_name: str,
    source_name: str,
    limit: int | None,
    api_key: str | None,
    api_key_env: str | None,
    api_key_header: str,
    headers: list[str],
    out: str | None,
) -> None:
    live_cfg = config.sources["posts"].get("live_provider", {})
    if not live_cfg.get("enabled", False):
        print("warning: live_provider is disabled in config; ingesting anyway for backfill/testing")
    provider_headers = _build_provider_headers(
        extra_headers=headers,
        api_key=api_key,
        api_key_env=api_key_env or live_cfg.get("requires_env_key"),
        api_key_header=api_key_header,
        is_remote=source.startswith("http://") or source.startswith("https://"),
    )
    posts = load_truthsocial_provider_posts(
        source,
        source_name=source_name,
        source_provider=provider_name,
        limit=limit,
        request_headers=provider_headers,
    )
    output_path = (
        Path(out)
        if out
        else config.paths.processed_dir / f"{source_name.replace(' ', '_')}_posts.parquet"
    )
    write_dataframe(truthsocial_provider_posts_to_frame(posts), output_path)
    write_json(
        config.paths.report_dir / "provider_posts_ingestion_audit.json",
        audit_truthsocial_provider_posts(
            posts, source=source, source_provider=provider_name
        ),
    )
    print(f"ingested {len(posts)} posts from provider {provider_name} source {source}")


def _scrape_truthsocial_live(
    config,
    *,
    fixture: str | None,
    watch: bool,
    max_iterations: int | None,
    poll_seconds: int | None,
    stale_after_seconds: int | None,
    storage_state: str | None,
    out: str | None,
    report_out: str | None,
    headless: bool,
) -> None:
    scraper_cfg = config.sources["posts"].get("truthsocial_browser", {})
    settings = BrowserScraperSettings(
        profile_url=str(scraper_cfg.get("profile_url", "https://truthsocial.com/@realDonaldTrump")),
        account_id=str(scraper_cfg.get("account_id", "107780257626128497")),
        source_name=str(scraper_cfg.get("source_name", "truthsocial_browser_live")),
        storage_state_path=Path(
            storage_state
            or scraper_cfg.get(
                "storage_state_path",
                "data/interim/truthsocial_browser_storage_state.json",
            )
        ),
        username_env=str(scraper_cfg.get("username_env", "TRUTHSOCIAL_USERNAME")),
        password_env=str(scraper_cfg.get("password_env", "TRUTHSOCIAL_PASSWORD")),
        totp_secret_env=str(scraper_cfg.get("totp_secret_env", "TRUTHSOCIAL_TOTP_SECRET")),
        poll_seconds=int(
            poll_seconds
            if poll_seconds is not None
            else scraper_cfg.get("poll_seconds", DEFAULT_POLL_SECONDS)
        ),
        stale_after_seconds=int(
            stale_after_seconds
            if stale_after_seconds is not None
            else scraper_cfg.get("stale_after_seconds", DEFAULT_STALE_AFTER_SECONDS)
        ),
        canonical_out=Path(out) if out else DEFAULT_CANONICAL_OUT,
        report_out=Path(report_out) if report_out else DEFAULT_REPORT_OUT,
        headless=headless,
    )
    iterations = max_iterations if max_iterations is not None else (None if watch else 1)
    completed = 0
    while iterations is None or completed < iterations:
        result = (
            run_truthsocial_fixture_scrape(fixture_path=fixture, settings=settings)
            if fixture
            else run_truthsocial_browser_scrape_once(settings=settings)
        )
        print(
            "truthsocial browser scrape: "
            f"auth={result.report['auth_status']}, stale={result.report['is_stale']}, "
            f"raw_rows={result.report['raw_rows_seen']}, "
            f"canonical_rows={result.report['canonical_rows_written']}, "
            f"out={result.canonical_out}"
        )
        completed += 1
        if iterations is not None and completed >= iterations:
            break
        time.sleep(settings.poll_seconds)


def _build_provider_headers(
    *,
    extra_headers: list[str],
    api_key: str | None,
    api_key_env: str | None,
    api_key_header: str,
    is_remote: bool,
) -> dict[str, str]:
    if not is_remote:
        return {}
    headers = {name: value for name, value in _parse_headers(extra_headers)}
    header_name = (api_key_header or "x-api-key").strip()
    if not header_name:
        raise SystemExit("--api-key-header cannot be blank")
    configured_key = (api_key or "").strip()
    resolved_key = configured_key
    if not resolved_key and api_key_env:
        resolved_key = os.getenv(api_key_env, "").strip()
        if not resolved_key:
            print(
                f"warning: API key env var {api_key_env} is not set; continuing without authentication header"
            )
    if resolved_key:
        headers[header_name] = resolved_key
    return headers


def _parse_headers(entries: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if ":" not in entry:
            raise SystemExit(f"Invalid header format '{entry}'. Expected 'key:value'.")
        name, value = entry.split(":", 1)
        key = name.strip()
        if not key:
            raise SystemExit("Header key cannot be empty.")
        parsed.append((key, value.strip()))
    return parsed


def _ingest_trumpstruth_feed(
    config,
    url: str,
    start_date: str | None,
    end_date: str | None,
    provider_name: str,
    source_name: str,
    limit: int | None,
    out: str | None,
) -> None:
    posts = load_trumpstruth_feed_posts(
        url,
        source_name=source_name,
        source_provider=provider_name,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    output_path = (
        Path(out)
        if out
        else config.paths.processed_dir / f"{source_name.replace(' ', '_')}_posts.parquet"
    )
    write_dataframe(trumpstruth_feed_posts_to_frame(posts), output_path)
    write_json(
        config.paths.report_dir / "trumpstruth_feed_ingestion_audit.json",
        audit_trumpstruth_feed_posts(
            posts,
            source=url,
        ),
    )
    print(
        f"ingested {len(posts)} posts from {provider_name} feed {url} "
        f"filters start_date={start_date or 'none'} end_date={end_date or 'none'}"
    )


def _check_archive_freshness(
    config, url: str | None, posts_path: str | None, stale_after_minutes: int
) -> None:
    archive_url = url or config.sources["posts"]["stiles_archive"]["latest_archive_url"]
    local_path = (
        Path(posts_path) if posts_path else config.paths.processed_dir / "cnn_archive_posts.parquet"
    )
    report = build_archive_freshness_report(
        source_url=archive_url,
        local_archive_path=local_path,
        stale_after_minutes=stale_after_minutes,
    )
    write_json(config.paths.report_dir / "archive_freshness_report.json", report)
    print(
        "archive freshness checked: "
        f"http_ok={report['is_http_ok']}, stale={report['is_stale_by_post_time']}"
    )


def _check_provider_freshness(
    config,
    source: str,
    source_name: str,
    source_provider: str,
    posts_path: str | None,
    stale_after_minutes: int,
    api_key: str | None,
    api_key_env: str | None,
    api_key_header: str,
    headers: list[str],
) -> None:
    is_remote = source.startswith("http://") or source.startswith("https://")
    provider_headers = _build_provider_headers(
        extra_headers=headers,
        api_key=api_key,
        api_key_env=api_key_env or config.sources["posts"].get("live_provider", {}).get(
            "requires_env_key"
        ),
        api_key_header=api_key_header,
        is_remote=is_remote,
    )
    local_posts = Path(posts_path) if posts_path else None
    report = build_provider_freshness_report(
        source_url=source,
        local_posts_path=local_posts,
        stale_after_minutes=stale_after_minutes,
        source_name=source_name,
        source_provider=source_provider,
        request_headers=provider_headers if is_remote else None,
    )
    write_json(
        config.paths.report_dir / "provider_posts_freshness_report.json",
        report,
    )
    print(
        "provider freshness checked: "
        f"http_ok={report['is_http_ok']}, stale={report['is_stale_by_post_time']}, "
        f"schema_drift={report['local_provider'].get('schema_drift_detected')}"
    )


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


def _download_databento_market(
    config,
    start: str,
    end: str | None,
    symbols: str,
    dataset: str,
    schema: str,
    stype_in: str,
    stype_out: str,
    limit: int | None,
    symbol_root: str,
    contract_symbol: str | None,
    continuous_symbol: str | None,
    output_path: str | None,
) -> None:
    try:
        bars, audit = download_databento_ohlcv(
            start=start,
            end=end,
            symbols=parse_symbol_list(symbols),
            dataset=dataset,
            schema=schema,
            stype_in=stype_in,
            stype_out=stype_out,
            limit=limit,
            symbol_root=symbol_root,
            contract_symbol=contract_symbol,
            continuous_symbol=continuous_symbol,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    target = (
        Path(output_path) if output_path else config.paths.processed_dir / "market_bars.parquet"
    )
    write_dataframe(bars, target)
    write_json(config.paths.report_dir / "market_ingestion_audit.json", audit_market_bars(bars))
    write_json(config.paths.report_dir / "databento_download_audit.json", audit)
    print(f"downloaded {len(bars)} Databento market bars to {target}")


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
    signal = _latest_signal_with_provider_health(scored, config)
    whipsaw_report = _read_json_report(config.paths.report_dir / "whipsaw_report.json")
    backtest_report = _read_json_report(config.paths.report_dir / "backtest_report.json")
    interpretation_report = _read_optional_json_report(
        config.paths.report_dir / "research_interpretation.json"
    )
    provider_freshness_report = _read_optional_json_report(
        config.paths.report_dir / "provider_posts_freshness_report.json"
    )
    output = build_dashboard(
        signal=signal,
        whipsaw_report=whipsaw_report,
        backtest_report=backtest_report,
        interpretation_report=interpretation_report,
        provider_freshness_report=provider_freshness_report,
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
        "finbert": _read_optional_json_report(
            config.paths.report_dir / "finbert_inference_report.json"
        ),
        "walk_forward": _read_optional_json_report(
            config.paths.report_dir / "walk_forward_report.json"
        ),
        "human_labels": _read_optional_json_report(
            config.paths.report_dir / "human_label_audit.json"
        ),
    }
    report = write_interpretation_report(reports=reports, report_dir=config.paths.report_dir)
    print(f"interpretation written: status={report['status']}")


def _latest_signal(config) -> None:
    scored = _read_or_build_whipsaw_scores(config)
    signal = _latest_signal_with_provider_health(scored, config)
    write_json(config.paths.report_dir / "latest_signal.json", signal.model_dump(mode="json"))
    print(f"latest signal: {signal.direction_signal} / {signal.kill_switch['action']}")


def _latest_signal_with_provider_health(scored, config):
    if scored.empty:
        return latest_signal_from_scores(scored, config)
    latest = scored.sort_values("received_at_utc").iloc[-1]
    return signal_from_scored_event(
        latest,
        config,
        provider_stale=_provider_stale_from_reports(config),
    )


def _provider_stale_from_reports(config) -> bool:
    posts_cfg = config.sources.get("posts", {})
    browser_enabled = bool(posts_cfg.get("truthsocial_browser", {}).get("enabled", False))
    provider_enabled = bool(posts_cfg.get("live_provider", {}).get("enabled", False))
    if not browser_enabled and not provider_enabled:
        return False

    if browser_enabled:
        report = _read_optional_json_report(
            config.paths.report_dir / "truthsocial_browser_scraper_report.json"
        )
        if report is None:
            return True
        if report.get("is_stale") is True or report.get("schema_drift_detected") is True:
            return True
        if report.get("auth_status") not in {"authenticated", "fixture"}:
            return True

    if provider_enabled:
        report = _read_optional_json_report(
            config.paths.report_dir / "provider_posts_freshness_report.json"
        )
        if report is None:
            return True
        local = report.get("local_provider") or {}
        if report.get("is_http_ok") is False or report.get("is_stale_by_post_time") is True:
            return True
        if local.get("schema_drift_detected") is True:
            return True

    return False


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


def _extract_config_arg(args: list[str], *, default: str) -> tuple[list[str], str]:
    cleaned: list[str] = []
    config_path = default
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--config":
            if index + 1 >= len(args):
                raise SystemExit("--config requires a value")
            config_path = args[index + 1]
            index += 2
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            index += 1
        else:
            cleaned.append(arg)
            index += 1
    return cleaned, config_path


if __name__ == "__main__":
    main()
