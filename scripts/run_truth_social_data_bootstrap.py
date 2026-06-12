from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from sentiment_engine.cli import main
from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.ingestion.posts_external_provider import TRUTHSOCIAL_PROVIDER
from sentiment_engine.ingestion.posts_trumpstruth_feed import TRUMPSTRUTH_PROVIDER_NAME, TRUMPSTRUTH_SOURCE_NAME

DEFAULT_REPORT_PATH = Path("reports/bootstrap_run_report.json")
DEFAULT_OUT_DIR = Path("data/processed")
ARCHIVE_OUT = DEFAULT_OUT_DIR / "cnn_archive_posts.parquet"
TRUMPSTRUTH_OUT = DEFAULT_OUT_DIR / "trumpstruth_posts.parquet"
MARKET_OUT = DEFAULT_OUT_DIR / "market_bars.parquet"
EVENTS_OUT = DEFAULT_OUT_DIR / "real_events.parquet"
HTTP_PREFIXES = ("http://", "https://")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap post and market ingestion for the Truth Social engine. "
            "Use provider-first source selection, then deterministic archive and RSS fallback."
        )
    )
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--provider-source", default=None)
    parser.add_argument("--provider-name", default=TRUTHSOCIAL_PROVIDER)
    parser.add_argument("--provider-source-name", default="provider_dump")
    parser.add_argument("--provider-limit", type=int, default=None)
    parser.add_argument("--provider-api-key")
    parser.add_argument("--provider-api-key-env")
    parser.add_argument("--provider-api-key-header", default="x-api-key")
    parser.add_argument("--provider-header", action="append", default=[])
    parser.add_argument("--use-archive", action="store_true")
    parser.add_argument("--archive-url", default=None)
    parser.add_argument("--archive-limit", type=int, default=None)

    parser.add_argument("--trumpstruth-feed", action="store_true")
    parser.add_argument("--trumpstruth-url", default="https://www.trumpstruth.org/feed")
    parser.add_argument("--trumpstruth-start-date", default=None)
    parser.add_argument("--trumpstruth-end-date", default=None)
    parser.add_argument("--trumpstruth-limit", type=int, default=None)
    parser.add_argument("--trumpstruth-provider-name", default=TRUMPSTRUTH_PROVIDER_NAME)
    parser.add_argument("--trumpstruth-source-name", default=TRUMPSTRUTH_SOURCE_NAME)

    parser.add_argument("--market-input", default=None)
    parser.add_argument("--market-source-name", default="databento_glbx_mdp3_ohlcv_1m")
    parser.add_argument("--market-symbol-root", default="NQ", choices=["NQ", "MNQ"])
    parser.add_argument("--market-out", default=str(MARKET_OUT))
    parser.add_argument("--market-input-only", action="store_true")

    parser.add_argument("--skip-event-build", action="store_true")
    parser.add_argument("--events-out", default=str(EVENTS_OUT))
    parser.add_argument("--posts-limit", type=int, default=None)
    parser.add_argument("--no-post-freshness-check", action="store_true")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def _resolve_archive_url(args: argparse.Namespace, config) -> str | None:
    return args.archive_url or config.sources["posts"]["stiles_archive"]["latest_archive_url"]


def _is_remote_source(source: str | None) -> bool:
    return bool(source and source.startswith(HTTP_PREFIXES))


def main_entry() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    resolved_archive_url = _resolve_archive_url(args, config)

    report: dict[str, Any] = {
        "config_path": args.config,
        "run_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "steps": [],
        "posts_path": None,
        "market_path": None,
        "events_path": None,
        "post_source": None,
    }

    if args.market_input_only:
        _ingest_market(args, report)
        _write_report(args, report)
        return

    posts_path = ""
    if args.provider_source:
        posts_path = _run_provider_ingest(args, report)
        if posts_path:
            report["posts_path"] = posts_path
            report["post_source"] = "provider"
            if not args.no_post_freshness_check:
                _run_provider_freshness(args, posts_path, report)
            report["steps"].append({"name": "provider", "status": "complete"})

    if not posts_path and args.trumpstruth_feed:
        posts_path = _run_trumpstruth_feed(args, report)
        if posts_path:
            report["posts_path"] = posts_path
            report["post_source"] = "trumpstruth_rss"
            report["steps"].append({"name": "trumpstruth_feed", "status": "complete"})

    if not posts_path and (args.use_archive or args.archive_url):
        posts_path = _run_archive_ingest(args, report, resolved_archive_url)
        if posts_path:
            report["posts_path"] = posts_path
            report["post_source"] = "cnn_archive"
            if _is_remote_source(resolved_archive_url):
                _run_archive_freshness(args, resolved_archive_url, posts_path)
            report["steps"].append({"name": "archive", "status": "complete"})

    if not posts_path:
        raise SystemExit(
            "No post source produced rows. Set --provider-source, --trumpstruth-feed, or --use-archive/--archive-url."
        )

    market_path = None
    if args.market_input:
        market_path = _ingest_market(args, report)

    if not args.skip_event_build:
        events_path = _build_events(posts_path, market_path, args)
        report["events_path"] = events_path

    _write_report(args, report)
    print(
        f"bootstrap complete: posts={posts_path}, market={market_path or 'not_provided'}, "
        f"events={report.get('events_path')}, report={args.report_out}"
    )


def _run_provider_ingest(args: argparse.Namespace, report: dict[str, Any]) -> str:
    source_name = args.provider_source_name.replace(" ", "_")
    posts_out = str(DEFAULT_OUT_DIR / f"{source_name}.parquet")
    command = [
        "--config",
        args.config,
        "ingest-provider-posts",
        "--source",
        args.provider_source,
        "--provider-name",
        args.provider_name,
        "--source-name",
        args.provider_source_name,
        "--out",
        posts_out,
        "--api-key-header",
        args.provider_api_key_header,
    ]
    if args.provider_limit is not None:
        command.extend(["--limit", str(args.provider_limit)])
    if args.provider_api_key is not None:
        command.extend(["--api-key", args.provider_api_key])
    if args.provider_api_key_env is not None:
        command.extend(["--api-key-env", args.provider_api_key_env])
    for header in args.provider_header:
        command.extend(["--header", header])

    try:
        main(command)
        return posts_out
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is None:
            report["steps"].append({"name": "provider", "status": "failed", "code": exc.__class__.__name__})
            print(f"provider ingest failed with {exc.__class__.__name__}: {exc}; continuing fallback")
            return ""
        report["steps"].append({"name": "provider", "status": "failed", "code": str(code)})
        if int(str(code) or 0) != 0:
            print(f"provider ingest failed with code {code}; continuing fallback")
            return ""
        return posts_out


def _run_archive_ingest(args: argparse.Namespace, report: dict[str, Any], archive_url: str) -> str:
    command = [
        "--config",
        args.config,
        "ingest-archive",
        "--out",
        str(ARCHIVE_OUT),
    ]
    if archive_url:
        command.extend(["--url", archive_url])
    if args.archive_limit is not None:
        command.extend(["--limit", str(args.archive_limit)])
    try:
        main(command)
        return str(ARCHIVE_OUT)
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is None:
            report["steps"].append({"name": "archive", "status": "failed", "code": exc.__class__.__name__})
            print(f"archive ingest failed with {exc.__class__.__name__}: {exc}")
            return ""
        report["steps"].append({"name": "archive", "status": "failed", "code": str(code)})
        if int(str(code) or 0) != 0:
            print(f"archive ingest failed with code {code}; stopping")
            return ""
        return str(ARCHIVE_OUT)


def _run_trumpstruth_feed(args: argparse.Namespace, report: dict[str, Any]) -> str:
    command = [
        "--config",
        args.config,
        "ingest-trumpstruth-feed",
        "--url",
        args.trumpstruth_url,
        "--source-name",
        args.trumpstruth_source_name,
        "--provider-name",
        args.trumpstruth_provider_name,
        "--out",
        str(TRUMPSTRUTH_OUT),
    ]
    if args.trumpstruth_start_date:
        command.extend(["--start-date", args.trumpstruth_start_date])
    if args.trumpstruth_end_date:
        command.extend(["--end-date", args.trumpstruth_end_date])
    if args.trumpstruth_limit:
        command.extend(["--limit", str(args.trumpstruth_limit)])
    try:
        main(command)
        return str(TRUMPSTRUTH_OUT)
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is None:
            report["steps"].append(
                {"name": "trumpstruth_feed", "status": "failed", "code": exc.__class__.__name__}
            )
            print(f"trumpstruth feed ingest failed with {exc.__class__.__name__}: {exc}")
            return ""
        report["steps"].append({"name": "trumpstruth_feed", "status": "failed", "code": str(code)})
        if int(str(code) or 0) != 0:
            print(f"trumpstruth feed ingest failed with code {code}; continuing")
            return ""
        return str(TRUMPSTRUTH_OUT)


def _run_archive_freshness(args: argparse.Namespace, archive_url: str, posts_path: str) -> None:
    if not _is_remote_source(archive_url):
        return
    command = [
        "--config",
        args.config,
        "check-archive-freshness",
        "--posts",
        posts_path,
    ]
    command.extend(["--url", archive_url])
    try:
        main(command)
    except Exception as exc:
        print(f"archive freshness check failed with {exc.__class__.__name__}: {exc}")


def _run_provider_freshness(args: argparse.Namespace, posts_path: str, report: dict[str, Any]) -> None:
    command = [
        "--config",
        args.config,
        "check-provider-freshness",
        "--source",
        args.provider_source,
        "--source-name",
        args.provider_source_name,
        "--source-provider",
        args.provider_name,
        "--posts",
        posts_path,
        "--api-key-header",
        args.provider_api_key_header,
    ]
    if args.provider_api_key is not None:
        command.extend(["--api-key", args.provider_api_key])
    if args.provider_api_key_env is not None:
        command.extend(["--api-key-env", args.provider_api_key_env])
    for header in args.provider_header:
        command.extend(["--header", header])

    try:
        main(command)
    except Exception as exc:
        code = getattr(exc, "code", None)
        report["steps"].append(
            {
                "name": "provider_freshness",
                "status": "failed",
                "code": str(code) if code is not None else exc.__class__.__name__,
            }
        )


def _ingest_market(args: argparse.Namespace, report: dict[str, Any]) -> str:
    command = [
        "--config",
        args.config,
        "ingest-market-file",
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
    report["market_path"] = args.market_out
    report["steps"].append({"name": "market_file", "status": "complete"})
    return args.market_out


def _build_events(posts_path: str, market_path: str | None, args: argparse.Namespace) -> str:
    if market_path is None:
        market_path = str(MARKET_OUT)

    command = [
        "--config",
        args.config,
        "build-archive-events",
        "--posts",
        posts_path,
        "--market",
        market_path,
        "--out",
        args.events_out,
    ]
    if args.posts_limit is not None:
        command.extend(["--limit-posts", str(args.posts_limit)])
    main(command)
    return args.events_out


def _write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"bootstrap report written to {report_path}")


if __name__ == "__main__":
    main_entry()
