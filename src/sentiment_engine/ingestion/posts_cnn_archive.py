from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from sentiment_engine.ingestion.posts_fixture import (
    audit_posts,
    normalise_stiles_record,
    posts_to_frame,
)
from sentiment_engine.schemas import PostRecord

CNN_ARCHIVE_PROVIDER = "cnn_ix_archive"
CNN_ARCHIVE_SOURCE_NAME = "cnn_truth_social_archive"
SUPPORTED_SUFFIXES = {".json", ".csv", ".parquet"}


def load_cnn_archive_posts(source: str | Path, *, limit: int | None = None) -> list[PostRecord]:
    rows = _read_archive_rows(source)
    if limit is not None:
        rows = rows[:limit]
    records = [
        normalise_stiles_record(
            row,
            source_name=CNN_ARCHIVE_SOURCE_NAME,
            source_provider=CNN_ARCHIVE_PROVIDER,
        )
        for row in rows
    ]
    return sorted(records, key=lambda record: (record.created_at_utc, record.post_id))


def archive_posts_to_frame(records: list[PostRecord]) -> pd.DataFrame:
    return posts_to_frame(records)


def audit_cnn_archive_posts(records: list[PostRecord], *, source: str | Path) -> dict[str, Any]:
    audit = audit_posts(records)
    audit.update(
        {
            "source": str(source),
            "source_provider": CNN_ARCHIVE_PROVIDER,
            "historical_backfill_only": True,
            "dl_readiness_note": (
                "This source supplies raw post text/metadata, not market labels. "
                "Deep learning requires event labels and temporal validation."
            ),
        }
    )
    return audit


def _read_archive_rows(source: str | Path) -> list[dict[str, Any]]:
    suffix = _suffix(source)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported archive format {suffix}; expected one of {sorted(SUPPORTED_SUFFIXES)}")
    payload = _read_bytes(source)
    if suffix == ".json":
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError("CNN archive JSON must contain a list of post records")
        return [_clean_record(record) for record in data]
    if suffix == ".csv":
        frame = pd.read_csv(BytesIO(payload))
    else:
        frame = pd.read_parquet(BytesIO(payload))
    return [_clean_record(record) for record in frame.to_dict("records")]


def _read_bytes(source: str | Path) -> bytes:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        with urlopen(source_text, timeout=30) as response:
            return response.read()
    return Path(source).read_bytes()


def _suffix(source: str | Path) -> str:
    parsed = urlparse(str(source))
    path = parsed.path if parsed.scheme else str(source)
    return Path(path).suffix.lower()


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if _is_missing(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _is_missing(value: object) -> bool:
    if isinstance(value, (list, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
