from __future__ import annotations

from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from sentiment_engine.utils.time import isoformat_z, parse_utc
from sentiment_engine.ingestion.posts_external_provider import TRUTHSOCIAL_PROVIDER

REQUIRED_POST_COLUMNS = {
    "source_name",
    "source_provider",
    "post_id",
    "author_id",
    "created_at_utc",
    "received_at_utc",
    "ingested_at_utc",
    "text_raw",
    "text_clean",
    "language",
    "post_type",
    "parent_post_id",
    "quoted_post_id",
    "urls",
    "media_urls",
    "has_image",
    "has_video",
    "engagement_metrics_json",
    "content_hash",
}
DEFAULT_STALE_AFTER_MINUTES = 30


def fetch_remote_provider_metadata(
    url: str,
    *,
    request_headers: dict[str, str] | None = None,
    checked_at_utc: datetime | None = None,
) -> dict[str, Any]:
    checked_at = checked_at_utc or datetime.now(UTC)
    request_headers = _default_headers(request_headers)
    request = Request(url, headers=request_headers, method="HEAD")
    response_status: int | None = None
    headers: dict[str, str] = {}
    error: str | None = None
    method = "HEAD"
    try:
        with urlopen(request, timeout=30) as response:
            response_status = response.status
            headers = {key.lower(): value for key, value in response.headers.items()}
    except HTTPError as exc:
        response_status = exc.code
        error = f"{type(exc).__name__}: {exc.code}"
        if exc.code == 405:
            method = "GET"
    except URLError as exc:
        error = f"{type(exc).__name__}: {exc.reason}"
        response_status = None
    except (HTTPException, OSError, TimeoutError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    else:
        method = "HEAD"

    if method == "GET":
        request = Request(url, headers=request_headers, method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                response_status = response.status
                headers = {key.lower(): value for key, value in response.headers.items()}
                error = None
        except HTTPError as exc:
            response_status = exc.code
            error = f"{type(exc).__name__}: {exc.code}"
        except (HTTPException, OSError, TimeoutError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"

    return {
        "url": url,
        "checked_at_utc": isoformat_z(checked_at),
        "http_status": response_status,
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "content_length": _int_or_none(headers.get("content-length")),
        "content_type": headers.get("content-type"),
        "error": error,
        "method": method,
    }


def build_provider_freshness_report(
    *,
    source_url: str,
    local_posts_path: str | Path | None,
    stale_after_minutes: int = DEFAULT_STALE_AFTER_MINUTES,
    source_name: str = "truthsocial_provider_dump",
    source_provider: str = TRUTHSOCIAL_PROVIDER,
    request_headers: dict[str, str] | None = None,
    checked_at_utc: datetime | None = None,
    remote_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checked_at = checked_at_utc or datetime.now(UTC)
    is_local_source = not str(source_url).startswith(("http://", "https://"))
    remote = remote_metadata
    if remote is None:
        remote = {
            "url": source_url,
            "checked_at_utc": isoformat_z(checked_at),
            "http_status": None,
            "etag": None,
            "last_modified": None,
            "content_length": None,
            "content_type": None,
            "error": None,
            "method": "local_source",
            "note": "Local source provided; remote check skipped",
        }
        if not is_local_source:
            remote = fetch_remote_provider_metadata(
                source_url,
                request_headers=request_headers,
                checked_at_utc=checked_at,
            )
    local = _local_provider_audit(local_posts_path)

    max_post_lag_seconds = None
    is_stale_by_post_time = None
    if local.get("exists") and local.get("max_created_at_utc"):
        max_created_at = parse_utc(local["max_created_at_utc"])
        max_post_lag_seconds = int((checked_at - max_created_at).total_seconds())
        is_stale_by_post_time = max_post_lag_seconds > stale_after_minutes * 60

    return {
        "source_url": source_url,
        "source_name": source_name,
        "source_provider": source_provider,
        "checked_at_utc": isoformat_z(checked_at),
        "stale_after_minutes": int(stale_after_minutes),
        "remote": remote,
        "local_provider": local,
        "max_post_lag_seconds": max_post_lag_seconds,
        "is_stale_by_post_time": is_stale_by_post_time,
        "is_http_ok": _is_http_ok(remote.get("http_status")),
        "methodology_notes": [
            "Provider freshness is one health check for operational gating; it is not a reliability guarantee.",
            "Run with the same headers used for production ingest to avoid false negatives.",
            "Schema checks validate the local canonical post snapshot, not raw third-party payloads.",
        ],
    }


def _local_provider_audit(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None, "reason": "no_local_posts_path"}
    source = Path(path)
    if not source.exists():
        return {"exists": False, "path": str(source), "reason": "path_not_found"}

    frame = _read_posts_frame(source)
    row_count = int(len(frame))
    missing_columns = sorted(REQUIRED_POST_COLUMNS - set(frame.columns))
    created = (
        _coerce_to_utc(frame["created_at_utc"])
        if "created_at_utc" in frame
        else pd.Series([], dtype="datetime64[ns, UTC]")
    )
    received = (
        _coerce_to_utc(frame["received_at_utc"])
        if "received_at_utc" in frame
        else pd.Series([], dtype="datetime64[ns, UTC]")
    )
    is_valid_timestamp = created.notna().all() and received.notna().all()
    text_series = (
        frame["text_clean"].fillna("")
        if "text_clean" in frame
        else pd.Series([], dtype="string")
    )
    media_flags = pd.Series([False] * row_count, dtype=bool)
    if "has_image" in frame and "has_video" in frame:
        media_flags = frame["has_image"].fillna(False).astype(bool) | frame[
            "has_video"
        ].fillna(False).astype(bool)

    return {
        "path": str(source),
        "exists": True,
        "row_count": row_count,
        "required_columns_present": len(missing_columns) == 0,
        "missing_required_columns": missing_columns,
        "schema_drift_detected": bool(missing_columns),
        "duplicate_post_ids": _duplicate_count(frame, "post_id"),
        "duplicate_content_hashes": _duplicate_count(frame, "content_hash"),
        "empty_text_rows": int((text_series.astype(str).str.strip() == "").sum()),
        "media_only_rows": int(((text_series.astype(str).str.strip() == "") & media_flags).sum())
        if row_count
        else 0,
        "min_created_at_utc": isoformat_z(created.min().to_pydatetime()) if len(created) else None,
        "max_created_at_utc": isoformat_z(created.max().to_pydatetime()) if len(created) else None,
        "invalid_timestamp_rows": int((~created.notna()).sum() + (~received.notna()).sum()),
        "is_timestamp_schema_valid": bool(is_valid_timestamp),
    }


def _read_posts_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported local provider post format: {suffix}")


def _coerce_to_utc(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    return converted


def _duplicate_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.Series(frame[column]).dropna().duplicated().sum())


def _default_headers(request_headers: dict[str, str] | None) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "trump-social-sentiment-engine"}
    if request_headers:
        headers.update(request_headers)
    return headers


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _is_http_ok(status: int | None) -> bool:
    if status is None:
        return False
    return 200 <= status < 400
