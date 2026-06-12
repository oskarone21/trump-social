from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_engine.config import EngineConfig
from sentiment_engine.research.events import EventBuildResult, build_event_dataset
from sentiment_engine.schemas import PostRecord


@dataclass(frozen=True)
class ArchiveEventBuildResult:
    events: pd.DataFrame
    skipped_posts: list[dict[str, str]]
    audit: dict[str, object]


def build_archive_event_dataset(
    *,
    posts_path: str | Path,
    market_path: str | Path,
    config: EngineConfig,
    limit_posts: int | None = None,
) -> ArchiveEventBuildResult:
    posts = _load_post_records(posts_path, limit=limit_posts)
    market_bars = pd.read_parquet(market_path)
    result = build_event_dataset(posts, market_bars, config)
    return ArchiveEventBuildResult(
        events=result.events,
        skipped_posts=result.skipped_posts,
        audit=_audit_archive_events(posts, market_bars, result, posts_path, market_path),
    )


def _load_post_records(path: str | Path, *, limit: int | None) -> list[PostRecord]:
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(limit)
    return [PostRecord.model_validate(_json_safe_row(row)) for row in frame.to_dict("records")]


def _json_safe_row(row: dict[str, object]) -> dict[str, object]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return value


def _audit_archive_events(
    posts: list[PostRecord],
    market_bars: pd.DataFrame,
    result: EventBuildResult,
    posts_path: str | Path,
    market_path: str | Path,
) -> dict[str, object]:
    skipped_by_reason: dict[str, int] = {}
    for item in result.skipped_posts:
        reason = item["reason"]
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
    return {
        "posts_path": str(posts_path),
        "market_path": str(market_path),
        "input_post_count": len(posts),
        "market_bar_count": int(len(market_bars)),
        "market_valid_bar_count": int(market_bars["is_valid_bar"].sum()),
        "event_count": int(len(result.events)),
        "skipped_post_count": int(len(result.skipped_posts)),
        "skipped_by_reason": skipped_by_reason,
    }
