from __future__ import annotations

from datetime import UTC, datetime

from sentiment_engine.ingestion.archive_monitor import build_archive_freshness_report
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts, posts_to_frame


def test_archive_freshness_report_audits_local_archive_without_network(tmp_path) -> None:
    archive_path = tmp_path / "archive.parquet"
    posts_to_frame(load_fixture_posts("data/fixtures/posts_stiles_sample.json")).to_parquet(
        archive_path,
        index=False,
    )

    report = build_archive_freshness_report(
        source_url="https://example.test/archive.parquet",
        local_archive_path=archive_path,
        stale_after_minutes=60,
        checked_at_utc=datetime(2026, 1, 5, 16, 20, tzinfo=UTC),
        remote_metadata={"http_status": 200, "etag": "abc"},
    )

    assert report["is_http_ok"] is True
    assert report["local_archive"]["row_count"] == 8
    assert report["local_archive"]["duplicate_post_ids"] == 0
    assert report["local_archive"]["max_created_at_utc"] == "2026-01-05T16:10:25Z"
    assert report["is_stale_by_post_time"] is False


def test_archive_freshness_report_flags_stale_local_snapshot(tmp_path) -> None:
    archive_path = tmp_path / "archive.parquet"
    posts_to_frame(load_fixture_posts("data/fixtures/posts_stiles_sample.json")).to_parquet(
        archive_path,
        index=False,
    )

    report = build_archive_freshness_report(
        source_url="https://example.test/archive.parquet",
        local_archive_path=archive_path,
        stale_after_minutes=30,
        checked_at_utc=datetime(2026, 1, 6, 16, 20, tzinfo=UTC),
        remote_metadata={"http_status": 200},
    )

    assert report["is_stale_by_post_time"] is True
    assert report["max_post_lag_seconds"] > 24 * 60 * 60


def test_archive_freshness_report_handles_missing_local_archive() -> None:
    report = build_archive_freshness_report(
        source_url="https://example.test/archive.parquet",
        local_archive_path="missing.parquet",
        remote_metadata={"http_status": 200},
        checked_at_utc=datetime(2026, 1, 6, 16, 20, tzinfo=UTC),
    )

    assert report["local_archive"]["exists"] is False
    assert report["is_stale_by_post_time"] is None
