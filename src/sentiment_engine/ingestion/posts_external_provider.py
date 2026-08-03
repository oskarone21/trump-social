from __future__ import annotations
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from sentiment_engine.ingestion.posts_fixture import audit_posts, posts_to_frame
from sentiment_engine.preprocessing.text import clean_post_text, extract_urls
from sentiment_engine.schemas import PostRecord
from sentiment_engine.utils.hashing import stable_hash
from sentiment_engine.utils.time import parse_utc

TRUTHSOCIAL_PROVIDER = "truth_social_provider_dump"
TRUTHSOCIAL_PROVIDER_TIME_FIELDS = (
    "created_at_utc",
    "createdAtUtc",
    "created_at",
    "createdAt",
    "created",
    "published_at",
    "publishedAt",
    "timestamp",
)
TRUTHSOCIAL_PROVIDER_PAYLOAD_KEYS = ("data", "items", "results", "posts", "response")
HTTP_USER_AGENT = "trump-social-sentiment-engine"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm")
SUPPORTED_SUFFIXES = {".json", ".csv", ".parquet"}
HTTP_PREFIXES = ("http://", "https://")


def load_truthsocial_provider_posts(
    source: str | Path,
    *,
    source_name: str,
    source_provider: str,
    limit: int | None = None,
    request_headers: dict[str, str] | None = None,
) -> list[PostRecord]:
    rows = _read_provider_rows(source, request_headers=request_headers)
    if limit is not None:
        rows = rows[:limit]
    records = [
        normalise_provider_record(
            row,
            source_name=source_name,
            source_provider=source_provider,
        )
        for row in rows
    ]
    return sorted(_dedupe_posts(records), key=lambda record: (record.created_at_utc, record.post_id))


def truthsocial_provider_posts_to_frame(records: list[PostRecord]) -> pd.DataFrame:
    return posts_to_frame(records)


def audit_truthsocial_provider_posts(
    records: list[PostRecord], *, source: str | Path, source_provider: str
) -> dict[str, Any]:
    audit = audit_posts(records)
    audit.update(
        {
            "source": str(source),
            "source_provider": source_provider,
            "historical_backfill_only": False,
            "source_is_live_capable": True,
            "dl_readiness_note": (
                "This adapter accepts third-party provider exports. "
                "Provider terms, latency, and coverage must be validated separately."
            ),
        }
    )
    return audit


def normalise_provider_record(
    item: dict[str, Any],
    *,
    source_name: str,
    source_provider: str,
) -> PostRecord:
    post_id = str(_required_field(item, "post id", ("post_id", "id", "postId", "post_id_str")))
    created_at_utc = _parse_timestamp(
        _required_field(
            item,
            "created_at",
            TRUTHSOCIAL_PROVIDER_TIME_FIELDS,
        )
    )
    received_at_utc = _parse_timestamp(
        _optional_field(
            item,
            (
                "received_at",
                "received_at_utc",
                "receivedAtUtc",
                "receivedAt",
                "fetched_at",
                "fetchedAt",
                "scraped_at",
                "scrapedAt",
                "collected_at",
            ),
        )
        or created_at_utc
    )
    ingested_at_utc = _parse_timestamp(
        _optional_field(item, ("ingested_at", "ingested_at_utc", "ingestedAt", "ingestedAtUtc"), default="now")
    )
    author_id = str(
        _optional_field(item, ("author_id", "authorId", "account_id", "accountId", "account.id"), default="unknown_author")
    )
    text_raw = str(
        _optional_field(
            item,
            ("text", "text_raw", "content", "body", "status", "full_text", "description"),
            default="",
        )
    )
    media_input = _optional_field(
        item,
        ("media", "media_urls", "mediaUrls", "media_attachments", "mediaAttachments"),
        default=[],
    )
    media_urls = sorted(set(_collect_urls(media_input)))
    post_type = _normalise_post_type(
        _optional_field(item, ("post_type", "type", "kind"), default="original")
    )
    urls = sorted(
        set(_extract_urls_from_record(item, text_raw))
    )
    content_clean = clean_post_text(text_raw)
    content_hash = stable_hash("|".join([post_id, content_clean, *media_urls, *urls]))

    return PostRecord(
        source_name=source_name,
        source_provider=source_provider,
        post_id=post_id,
        author_id=author_id,
        created_at_utc=created_at_utc,
        received_at_utc=received_at_utc,
        ingested_at_utc=ingested_at_utc,
        text_raw=text_raw,
        text_clean=content_clean,
        language=_optional_field(item, ("language", "lang"), default="en"),
        post_type=post_type,
        parent_post_id=_optional_field(item, ("parent_post_id", "in_reply_to_id"), default=None),
        quoted_post_id=_optional_field(item, ("quoted_post_id", "quote_id"), default=None),
        urls=urls,
        media_urls=media_urls,
        has_image=any(url.lower().endswith(IMAGE_SUFFIXES) for url in media_urls),
        has_video=any(url.lower().endswith(VIDEO_SUFFIXES) for url in media_urls),
        engagement_metrics_json={
            "replies_count": int(_metric(item, ("replies_count", "reply_count", "replies", "replyCount"), 0)),
            "reblogs_count": int(_metric(item, ("reblogs_count", "repost_count", "reblogs", "retweet_count"), 0)),
            "favourites_count": int(_metric(item, ("favourites_count", "favourite_count", "likes_count", "likes"), 0)),
        },
        content_hash=content_hash,
        raw_json=item,
    )


def _read_provider_rows(
    source: str | Path,
    request_headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    suffix = _suffix(source)
    is_remote = str(source).startswith(HTTP_PREFIXES)
    if suffix not in SUPPORTED_SUFFIXES and not (is_remote and suffix == ""):
        raise ValueError(
            f"Unsupported provider format {suffix}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    payload = _read_provider_payload(source, request_headers=request_headers)
    if suffix == ".json" or (is_remote and suffix == ""):
        data = _parse_json_payload(payload)
        return [_clean_row(record) for record in data]
    if suffix == ".csv":
        frame = pd.read_csv(BytesIO(payload))
        return [_clean_row(record) for record in frame.to_dict("records")]
    frame = pd.read_parquet(BytesIO(payload))
    return [_clean_row(record) for record in frame.to_dict("records")]


def _read_provider_payload(
    source: str | Path,
    request_headers: dict[str, str] | None = None,
) -> bytes:
    source_text = str(source)
    if source_text.startswith(HTTP_PREFIXES):
        headers = {"Accept": "application/json", "User-Agent": HTTP_USER_AGENT}
        if request_headers:
            headers.update(request_headers)
        request = Request(source_text, headers=headers)
        with urlopen(request, timeout=30) as response:
            return response.read()
    return Path(source).read_bytes()


def _parse_json_payload(payload: bytes) -> list[dict[str, Any]]:
    payload_obj = json.loads(payload.decode("utf-8"))
    payload_rows = _extract_provider_rows(payload_obj)
    if not isinstance(payload_rows, list):
        raise ValueError("Provider JSON payload must contain a list of post records")
    if not payload_rows:
        raise ValueError("Provider JSON payload was empty")
    if not all(isinstance(record, dict) for record in payload_rows):
        raise ValueError("Provider JSON payload must be a list of objects")
    return [_clean_row(record) for record in payload_rows]


def _extract_provider_rows(payload_obj: dict[str, Any] | list[dict[str, Any]]) -> object:
    if isinstance(payload_obj, list):
        return payload_obj
    if not isinstance(payload_obj, dict):
        return payload_obj

    for key in TRUTHSOCIAL_PROVIDER_PAYLOAD_KEYS:
        value = payload_obj.get(key)
        if isinstance(value, list):
            return value

    for key in TRUTHSOCIAL_PROVIDER_PAYLOAD_KEYS:
        value = payload_obj.get(key)
        if isinstance(value, dict):
            nested = _extract_provider_rows(value)
            if nested is not None:
                return nested

    return payload_obj


def _extract_urls_from_record(item: dict[str, Any], text_raw: str) -> list[str]:
    fields = _collect_urls(
        _optional_field(item, ("url", "permalink", "uri", "post_url", "postUrl"), default=[])
    )
    fields.extend(extract_urls(text_raw))
    return fields


def _collect_urls(value: object) -> list[str]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        collected: list[str] = []
        for nested_value in value.values():
            collected.extend(_collect_urls(nested_value))
        return collected
    if isinstance(value, (list, tuple)):
        collected = []
        for item in value:
            if isinstance(item, dict):
                collected.extend(_collect_urls(item.get("url") or item.get("src") or item.get("href")))
            else:
                collected.extend(_collect_urls(item))
        return collected
    return [str(value)]


def _clean_row(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if _is_missing(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _is_missing(value: object) -> bool:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _required_field(
    row: dict[str, Any], label: str, names: tuple[str, ...]
) -> Any:
    value = _optional_field(row, names, default=None)
    if value is None:
        raise ValueError(f"Missing required field {label}; tried keys {names}")
    return value


def _optional_field(
    row: dict[str, Any],
    names: tuple[str, ...],
    *,
    default: Any = None,
) -> Any:
    for name in names:
        value = _get_nested(row, name)
        if value is not None:
            return value
    return default


def _get_nested(row: dict[str, Any], key: str) -> Any:
    if "." not in key:
        return row.get(key)
    value: object = row
    for path in key.split("."):
        if not isinstance(value, dict) or path not in value:
            return None
        value = value[path]
    return value


def _parse_timestamp(value: Any) -> datetime:
    if value == "now":
        return datetime.now(UTC)
    try:
        parsed = parse_utc(value)
        return parsed
    except (TypeError, ValueError, OverflowError):
        if isinstance(value, (int, float)):
            seconds = value / 1000 if value > 10_000_000_000 else value
            return pd.Timestamp.fromtimestamp(seconds, tz="UTC").to_pydatetime()
    raise ValueError("Could not parse timestamp")


def _normalise_post_type(value: object) -> str:
    if value in (None, ""):
        return "unknown"
    candidate = str(value).strip().lower()
    if candidate in {"original", "reply", "retruth", "quote", "deleted", "edited", "unknown"}:
        return candidate
    return "original"


def _dedupe_posts(records: list[PostRecord]) -> list[PostRecord]:
    by_key: dict[tuple[str, str], PostRecord] = {}
    for record in records:
        key = (record.post_id, record.content_hash)
        existing = by_key.get(key)
        if existing is None or record.received_at_utc >= existing.received_at_utc:
            by_key[key] = record
    return list(by_key.values())


def _metric(row: dict[str, Any], names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = _optional_field(row, (name,))
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return default


def _suffix(source: str | Path) -> str:
    parsed = urlparse(str(source))
    path = parsed.path if parsed.scheme else str(source)
    return Path(path).suffix.lower()
