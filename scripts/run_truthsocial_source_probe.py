from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from sentiment_engine.utils.io import write_json


DEFAULT_OUTPUT_PATH = Path("reports") / "truthsocial_source_probe.json"
DEFAULT_TIMEOUT = 12
OFFICIAL_PROFILE_URL = "https://truthsocial.com/@realDonaldTrump"
OFFICIAL_V1_STATUSES = "https://truthsocial.com/api/v1/accounts/107780257626128497/statuses?limit=1"
OFFICIAL_ACCOUNT_LOOKUP = "https://truthsocial.com/api/v1/accounts/lookup?acct=realDonaldTrump"
TRUMPSOCIAL_DOCS = "https://truthsocial.com/api/docs"
SC_DOCS = "https://docs.scrapecreators.com/v1/truthsocial/user/posts/"
SC_URL_WITH_HANDLE = "https://api.scrapecreators.com/v1/truthsocial/user/posts?handle=realDonaldTrump&limit=1"
SC_URL_WITH_ID = "https://api.scrapecreators.com/v1/truthsocial/user/posts?user_id=107780257626128497&limit=1"
SC_SOCIALCRAWL_URL = "https://www.socialcrawl.dev/v1/truthsocial/user/posts?handle=realDonaldTrump&limit=1"
TRUMPSTRUTH_FEED = "https://www.trumpstruth.org/feed"
CNN_ARCHIVE_PARQUET = "https://ix.cnn.io/data/truth-social/truth_archive.parquet"


@dataclass
class ProbeResult:
    name: str
    url: str
    status_code: int | None
    ok: bool
    elapsed_ms: float
    error: str | None
    snippet: str | None
    headers: dict[str, str] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe likely Truth Social source endpoints and emit a simple availability "
            "report for procurement/runtime decisioning."
        )
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--socialcrawl-key-env",
        default="SOCIALCRAWL_API_KEY",
        help="Environment variable holding Socialcrawl API key, if available.",
    )
    parser.add_argument(
        "--scrapecreators-key-env",
        default="SCRAPECREATORS_API_KEY",
        help="Environment variable holding ScrapeCreators API key, if available.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for JSON probe output.",
    )
    return parser.parse_args()


def _snippet(bytes_payload: bytes) -> str:
    if not bytes_payload:
        return ""
    text = bytes_payload[:512].decode("utf-8", errors="replace")
    return text.replace("\n", " ").strip()


def _safe_headers(raw_headers: list[tuple[str, str]]) -> dict[str, str]:
    return {name: value for name, value in raw_headers}


def _build_request(url: str, headers: dict[str, str] | None = None) -> Request:
    request_headers = {
        "User-Agent": "trump-social-source-prober/1.0",
        "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
    }
    if headers:
        request_headers.update(headers)
    return Request(url, headers=request_headers)


def _run_probe(name: str, url: str, headers: dict[str, str] | None, timeout: float) -> ProbeResult:
    start = time.perf_counter()
    try:
        request = _build_request(url, headers)
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(2048)
            status_code = getattr(response, "status", None)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return ProbeResult(
                name=name,
                url=url,
                status_code=status_code,
                ok=200 <= int(status_code) < 300,
                elapsed_ms=round(elapsed_ms, 2),
                error=None,
                snippet=_snippet(payload),
                headers=_safe_headers(response.getheaders()),
            )
    except HTTPError as exc:
        return ProbeResult(
            name=name,
            url=url,
            status_code=exc.code,
            ok=False,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            error=f"HTTPError: {exc.reason}",
            snippet=_snippet(getattr(exc, "read")()[:2048]) if hasattr(exc, "read") else None,
            headers=_safe_headers(exc.headers.items()) if getattr(exc, "headers", None) is not None else None,
        )
    except (URLError, TimeoutError, OSError) as exc:
        return ProbeResult(
            name=name,
            url=url,
            status_code=None,
            ok=False,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            error=f"{type(exc).__name__}: {exc}",
            snippet=None,
            headers=None,
        )


def _build_probe_plan(args: argparse.Namespace) -> list[tuple[str, str, dict[str, str] | None]]:
    socialcrawl_key = os.getenv(args.socialcrawl_key_env, "").strip()
    scrapecreators_key = os.getenv(args.scrapecreators_key_env, "").strip()

    probe_plan: list[tuple[str, str, dict[str, str] | None]] = [
        ("truthsocial_api_docs", TRUMPSOCIAL_DOCS, None),
        ("truthsocial_profile", OFFICIAL_PROFILE_URL, None),
        ("truthsocial_api_accounts_statuses", OFFICIAL_V1_STATUSES, None),
        ("truthsocial_api_account_lookup", OFFICIAL_ACCOUNT_LOOKUP, None),
        ("scrapecreators_docs", SC_DOCS, None),
        ("scrapecreators_user_posts_handle", SC_URL_WITH_HANDLE, {"x-api-key": scrapecreators_key} if scrapecreators_key else None),
        ("scrapecreators_user_posts_user_id", SC_URL_WITH_ID, {"x-api-key": scrapecreators_key} if scrapecreators_key else None),
        ("socialcrawl_user_posts", SC_SOCIALCRAWL_URL, {"x-api-key": socialcrawl_key} if socialcrawl_key else None),
        ("trumpstruth_rss", TRUMPSTRUTH_FEED, None),
        ("cnn_truth_archive_parquet", CNN_ARCHIVE_PARQUET, None),
    ]

    return probe_plan


def _recommendation(results: list[ProbeResult]) -> list[str]:
    ok_200 = [r for r in results if r.ok]
    if not ok_200:
        return [
            "No endpoint returned HTTP 2xx.",
            "Fallback posture: use archive/RSS and approved provider dumps only until key-enabled provider endpoints are validated.",
        ]

    preferred = [r for r in ok_200 if "official" not in r.name]
    if preferred:
        preferred.sort(key=lambda r: r.elapsed_ms)
        return [
            "Candidate endpoints with 2xx response observed:",
            *[f"- {r.name}: {r.url} (status {r.status_code}, {r.elapsed_ms} ms)" for r in preferred],
        ]

    return [
        f"Only official endpoints returned 2xx: {[r.name for r in ok_200]}",
        "Run full ingestion with explicit provider contract before enabling live use.",
    ]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = [
        asdict(_run_probe(name, url, headers, args.timeout))
        for name, url, headers in _build_probe_plan(args)
    ]

    ranked = sorted(
        [r for r in results if r["status_code"] is not None],
        key=lambda r: r["elapsed_ms"],
    )

    payload = {
        "run_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "timeout_seconds": args.timeout,
        "socialcrawl_key_env_present": bool(os.getenv(args.socialcrawl_key_env, "").strip()),
        "scrapecreators_key_env_present": bool(os.getenv(args.scrapecreators_key_env, "").strip()),
        "results": results,
        "ranked_by_latency": ranked,
        "recommendations": _recommendation([ProbeResult(**r) for r in results]),
    }

    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"source probe report written: {output_path}")


if __name__ == "__main__":
    main()
