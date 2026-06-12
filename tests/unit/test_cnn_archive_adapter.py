from __future__ import annotations

from sentiment_engine.ingestion.posts_cnn_archive import (
    CNN_ARCHIVE_PROVIDER,
    load_cnn_archive_posts,
)


def test_cnn_archive_adapter_normalises_local_json_fixture() -> None:
    posts = load_cnn_archive_posts("data/fixtures/posts_stiles_sample.json", limit=2)

    assert len(posts) == 2
    assert posts[0].source_provider == CNN_ARCHIVE_PROVIDER
    assert posts[0].created_at_utc.tzinfo is not None
    assert posts[0].post_id == "fixture-001"
