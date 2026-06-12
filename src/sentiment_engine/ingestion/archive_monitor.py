from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from sentiment_engine.utils.time import isoformat_z, parse_utc, to_utc_series

POST_ID_COLUMN = "post_id"
CONTENT_HASH_COLUMN = "content_hash"
TEXT_COLUMN = "text_clean"
CREATED_AT_COLUMN = "created_at_utc"
MEDIA_COLUMNS = ["has_image", "has_video"]
STALE_AFTER_MINUTES_DEFAULT = 30


def fetch_remote_archive_metadata(
    url: str, *, checked_at_utc: datetime | None = None
) -> dict[str, Any]:
    checked_at = checked_at_utc or datetime.now(UTC)
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=30) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return {
            "url": url,
            "checked_at_utc": isoformat_z(checked_at),
            "http_status": int(response.status),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "content_length": _int_or_none(headers.get("content-length")),
            "content_type": headers.get("content-type"),
        }


def build_archive_freshness_report(
    *,
    source_url: str,
    local_archive_path: str | Path | None,
    stale_after_minutes: int = STALE_AFTER_MINUTES_DEFAULT,
    remote_metadata: dict[str, Any] | None = None,
    checked_at_utc: datetime | None = None,
) -> dict[str, Any]:
    checked_at = checked_at_utc or datetime.now(UTC)
    remote = remote_metadata or fetch_remote_archive_metadata(source_url, checked_at_utc=checked_at)
    local = _local_archive_audit(local_archive_path) if local_archive_path else None
    max_post_lag_seconds = None
    is_stale_by_post_time = None
    if local and local.get("max_created_at_utc"):
        max_created_at = parse_utc(local["max_created_at_utc"])
        max_post_lag_seconds = int((checked_at - max_created_at).total_seconds())
        is_stale_by_post_time = max_post_lag_seconds > stale_after_minutes * 60
    return {
        "source_url": source_url,
        "checked_at_utc": isoformat_z(checked_at),
        "stale_after_minutes": int(stale_after_minutes),
        "remote": remote,
        "local_archive": local,
        "max_post_lag_seconds": max_post_lag_seconds,
        "is_stale_by_post_time": is_stale_by_post_time,
        "is_http_ok": 200 <= int(remote.get("http_status", 0)) < 400,
        "methodology_notes": [
            "CNN archive freshness is suitable for backfill monitoring, not trading-grade latency.",
            "Text-only model training must filter empty-text rows or route media-only posts "
            "separately.",
            "Local archive freshness depends on the latest ingested snapshot, not only "
            "remote headers.",
        ],
    }


def _local_archive_audit(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"path": str(source), "exists": False}
    frame = _read_archive_frame(source)
    created = (
        to_utc_series(frame[CREATED_AT_COLUMN]) if CREATED_AT_COLUMN in frame else pd.Series([])
    )
    text = frame[TEXT_COLUMN].fillna("").astype(str) if TEXT_COLUMN in frame else pd.Series([])
    has_media = _media_mask(frame)
    return {
        "path": str(source),
        "exists": True,
        "row_count": int(len(frame)),
        "duplicate_post_ids": _duplicate_count(frame, POST_ID_COLUMN),
        "duplicate_content_hashes": _duplicate_count(frame, CONTENT_HASH_COLUMN),
        "empty_text_rows": int((text.str.strip() == "").sum()),
        "media_only_rows": int(((text.str.strip() == "") & has_media).sum()) if len(text) else 0,
        "min_created_at_utc": isoformat_z(created.min().to_pydatetime()) if len(created) else None,
        "max_created_at_utc": isoformat_z(created.max().to_pydatetime()) if len(created) else None,
    }


def _read_archive_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported local archive format: {suffix}")


def _media_mask(frame: pd.DataFrame) -> pd.Series:
    if all(column in frame for column in MEDIA_COLUMNS):
        return frame[MEDIA_COLUMNS].fillna(False).astype(bool).any(axis=1)
    return pd.Series([False] * len(frame))


def _duplicate_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].dropna().duplicated().sum())


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
