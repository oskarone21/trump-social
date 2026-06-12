from __future__ import annotations

from datetime import UTC, datetime

from sentiment_engine.ingestion.posts_external_provider import truthsocial_provider_posts_to_frame
from sentiment_engine.ingestion.posts_external_provider import (
    load_truthsocial_provider_posts,
)
from sentiment_engine.ingestion.provider_monitor import build_provider_freshness_report


def _fixture_provider_posts():
    return load_truthsocial_provider_posts(
        "data/fixtures/posts_truthsocial_provider_sample.json",
        source_name="provider_dump",
        source_provider="truthsocial_provider_dump",
    )


def test_provider_freshness_audits_local_snapshot(tmp_path) -> None:
    local_path = tmp_path / "provider_posts.parquet"
    posts = truthsocial_provider_posts_to_frame(_fixture_provider_posts())
    posts.to_parquet(local_path, index=False)

    report = build_provider_freshness_report(
        source_url="https://example.test/provider.json",
        local_posts_path=local_path,
        stale_after_minutes=240,
        source_name="provider_dump",
        source_provider="truthsocial_provider_dump",
        remote_metadata={"http_status": 200, "etag": "abc"},
        checked_at_utc=datetime(2026, 1, 10, 10, 30, tzinfo=UTC),
    )

    assert report["is_http_ok"] is True
    assert report["local_provider"]["row_count"] == 2
    assert report["local_provider"]["required_columns_present"] is True
    assert report["local_provider"]["schema_drift_detected"] is False
    assert report["is_stale_by_post_time"] is False


def test_provider_freshness_detects_stale_snapshot(tmp_path) -> None:
    local_path = tmp_path / "provider_posts.parquet"
    posts = truthsocial_provider_posts_to_frame(_fixture_provider_posts())
    posts.to_parquet(local_path, index=False)

    report = build_provider_freshness_report(
        source_url="https://example.test/provider.json",
        local_posts_path=local_path,
        stale_after_minutes=1,
        source_name="provider_dump",
        source_provider="truthsocial_provider_dump",
        remote_metadata={"http_status": 200},
        checked_at_utc=datetime(2026, 1, 10, 11, 30, tzinfo=UTC),
    )

    assert report["is_stale_by_post_time"] is True
    assert report["max_post_lag_seconds"] > 3600


def test_provider_freshness_missing_local_path():
    report = build_provider_freshness_report(
        source_url="https://example.test/provider.json",
        local_posts_path="missing-provider.parquet",
        source_name="provider_dump",
        remote_metadata={"http_status": 200},
        checked_at_utc=datetime(2026, 1, 10, 11, 30, tzinfo=UTC),
    )

    assert report["local_provider"]["exists"] is False
    assert report["local_provider"]["reason"] == "path_not_found"
    assert report["is_stale_by_post_time"] is None
    assert report["is_http_ok"] is True


def test_provider_freshness_local_source_skips_remote_lookup() -> None:
    local_path = "data/fixtures/posts_truthsocial_provider_sample.json"
    report = build_provider_freshness_report(
        source_url=local_path,
        local_posts_path=local_path,
        stale_after_minutes=30,
        source_name="provider_dump",
        source_provider="truthsocial_provider_dump",
        checked_at_utc=datetime(2026, 1, 10, 11, 30, tzinfo=UTC),
    )

    assert report["local_provider"]["exists"] is True
    assert report["remote"]["method"] == "local_source"
    assert report["is_http_ok"] is False
