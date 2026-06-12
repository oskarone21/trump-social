from __future__ import annotations

from sentiment_engine.ingestion.posts_trumpstruth_feed import (
    TRUMPSTRUTH_PROVIDER_NAME,
    audit_trumpstruth_feed_posts,
    load_trumpstruth_feed_posts,
)


def test_trumpstruth_feed_adapter_parses_rss_sample() -> None:
    posts = load_trumpstruth_feed_posts(
        "data/fixtures/trumpstruth_feed_sample.xml",
        source_name="trumpstruth_feed",
        source_provider=TRUMPSTRUTH_PROVIDER_NAME,
    )

    assert len(posts) == 2
    assert posts[0].source_provider == TRUMPSTRUTH_PROVIDER_NAME
    assert posts[0].post_id == "ts-2026-01-05-1"
    assert posts[0].author_id == "realDonaldTrump"
    assert posts[0].has_image
    assert posts[1].post_type == "reply"
    assert posts[1].author_id == "TeamTrump"


def test_trumpstruth_feed_adapter_supports_limit() -> None:
    posts = load_trumpstruth_feed_posts(
        "data/fixtures/trumpstruth_feed_sample.xml",
        source_name="trumpstruth_feed",
        source_provider=TRUMPSTRUTH_PROVIDER_NAME,
        limit=1,
    )

    assert len(posts) == 1
    assert posts[0].post_id == "ts-2026-01-05-1"


def test_trumpstruth_feed_audit_marks_backfill_only() -> None:
    posts = load_trumpstruth_feed_posts(
        "data/fixtures/trumpstruth_feed_sample.xml",
        source_name="trumpstruth_feed",
        source_provider=TRUMPSTRUTH_PROVIDER_NAME,
    )

    audit = audit_trumpstruth_feed_posts(
        posts,
        source="data/fixtures/trumpstruth_feed_sample.xml",
    )

    assert audit["source_provider"] == TRUMPSTRUTH_PROVIDER_NAME
    assert audit["historical_backfill_only"] is True
    assert audit["source_is_live_capable"] is False
    assert audit["row_count"] == 2
