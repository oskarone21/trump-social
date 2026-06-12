from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.constants import POST_FIXTURE_PROVIDER
from sentiment_engine.preprocessing.text import clean_post_text, extract_urls
from sentiment_engine.schemas import PostRecord
from sentiment_engine.utils.hashing import stable_hash
from sentiment_engine.utils.io import read_json_records
from sentiment_engine.utils.time import isoformat_z, parse_utc

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm")
TRUMP_AUTHOR_ID = "realDonaldTrump"


def load_fixture_posts(path: str | Path) -> list[PostRecord]:
    records = [
        normalise_stiles_record(
            item,
            source_name="fixture_stiles_archive",
            source_provider=POST_FIXTURE_PROVIDER,
        )
        for item in read_json_records(path)
    ]
    deduped = _dedupe_posts(records)
    return sorted(deduped, key=lambda record: (record.created_at_utc, record.post_id))


def posts_to_frame(records: list[PostRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump(mode="json") for record in records])


def audit_posts(records: list[PostRecord]) -> dict[str, Any]:
    if not records:
        return {"row_count": 0, "valid_rows": 0, "duplicate_post_ids": 0, "min_created_at_utc": None}
    post_ids = [record.post_id for record in records]
    empty_text_rows = sum(1 for record in records if not record.text_clean)
    return {
        "row_count": len(records),
        "valid_rows": len(records),
        "duplicate_post_ids": len(post_ids) - len(set(post_ids)),
        "empty_text_rows": empty_text_rows,
        "media_only_rows": sum(
            1 for record in records if not record.text_clean and (record.has_image or record.has_video)
        ),
        "min_created_at_utc": isoformat_z(min(record.created_at_utc for record in records)),
        "max_created_at_utc": isoformat_z(max(record.created_at_utc for record in records)),
        "max_feed_lag_ms": max(
            int((record.received_at_utc - record.created_at_utc).total_seconds() * 1000)
            for record in records
        ),
    }


def normalise_stiles_record(
    item: dict[str, Any],
    *,
    source_name: str,
    source_provider: str,
) -> PostRecord:
    created_at = parse_utc(item["created_at"])
    received_at = parse_utc(item.get("received_at", created_at))
    text_raw = str(item.get("content", ""))
    text_clean = clean_post_text(text_raw)
    media_urls = _list_value(item.get("media", []))
    content_hash = stable_hash("|".join([str(item["id"]), text_clean, *media_urls]))
    urls = sorted(set(extract_urls(text_raw) + _list_value(item.get("url", ""))))
    return PostRecord(
        source_name=source_name,
        source_provider=source_provider,
        post_id=str(item["id"]),
        author_id=TRUMP_AUTHOR_ID,
        created_at_utc=created_at,
        received_at_utc=received_at,
        ingested_at_utc=datetime.now(UTC),
        text_raw=text_raw,
        text_clean=text_clean,
        language="en",
        post_type=str(item.get("post_type", "original")),
        parent_post_id=item.get("parent_post_id"),
        quoted_post_id=item.get("quoted_post_id"),
        urls=[url for url in urls if url],
        media_urls=media_urls,
        has_image=any(url.lower().endswith(IMAGE_SUFFIXES) for url in media_urls),
        has_video=any(url.lower().endswith(VIDEO_SUFFIXES) for url in media_urls),
        engagement_metrics_json={
            "replies_count": int(item.get("replies_count", 0) or 0),
            "reblogs_count": int(item.get("reblogs_count", 0) or 0),
            "favourites_count": int(item.get("favourites_count", 0) or 0),
        },
        content_hash=content_hash,
        raw_json=item,
    )


def _dedupe_posts(records: list[PostRecord]) -> list[PostRecord]:
    by_key: dict[tuple[str, str], PostRecord] = {}
    for record in records:
        key = (record.post_id, record.content_hash)
        existing = by_key.get(key)
        if existing is None or record.ingested_at_utc >= existing.ingested_at_utc:
            by_key[key] = record
    return list(by_key.values())


def _list_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist() if item]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]
