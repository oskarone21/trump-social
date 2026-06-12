from __future__ import annotations

from sentiment_engine.ingestion.posts_external_provider import (
    TRUTHSOCIAL_PROVIDER,
    audit_truthsocial_provider_posts,
    load_truthsocial_provider_posts,
)


def test_provider_adapter_normalises_json_fixture() -> None:
    posts = load_truthsocial_provider_posts(
        "data/fixtures/posts_truthsocial_provider_sample.json",
        source_name="provider_dump",
        source_provider=TRUTHSOCIAL_PROVIDER,
        limit=2,
    )

    assert len(posts) == 2
    assert posts[0].post_id == "provider-101"
    assert posts[0].author_id == "realDonaldTrump"
    assert posts[0].source_provider == TRUTHSOCIAL_PROVIDER
    assert posts[0].has_video is True
    assert posts[1].post_type == "reply"
    assert posts[1].raw_json["author"]["id"] == "realDonaldTrump"


def test_provider_adapter_audit_contains_live_source_metadata() -> None:
    posts = load_truthsocial_provider_posts(
        "data/fixtures/posts_truthsocial_provider_sample.json",
        source_name="provider_dump",
        source_provider=TRUTHSOCIAL_PROVIDER,
        limit=1,
    )
    audit = audit_truthsocial_provider_posts(
        posts,
        source="data/fixtures/posts_truthsocial_provider_sample.json",
        source_provider=TRUTHSOCIAL_PROVIDER,
    )

    assert audit["source_provider"] == TRUTHSOCIAL_PROVIDER
    assert audit["historical_backfill_only"] is False
    assert audit["source_is_live_capable"] is True


def test_provider_adapter_parses_nested_json_payload_wrapper(tmp_path) -> None:
    payload_path = tmp_path / "provider_items.json"
    payload_path.write_text(
        '{"items": [{"id": "provider-201", "createdAt": "2026-01-11T11:00:00Z", '
        '"text": "A wrapped payload should parse.", "authorId": "realDonaldTrump"}]}'
    )

    posts = load_truthsocial_provider_posts(
        payload_path,
        source_name="provider_dump",
        source_provider=TRUTHSOCIAL_PROVIDER,
        limit=1,
    )

    assert len(posts) == 1
    assert posts[0].post_id == "provider-201"
    assert posts[0].source_provider == TRUTHSOCIAL_PROVIDER


def test_provider_adapter_sends_headers_for_remote_sources(monkeypatch) -> None:
    payload = b'[{"id":"provider-301","created_at":"2026-01-12T12:00:00Z","text":"Remote request test","authorId":"realDonaldTrump"}]'
    captured_headers: dict[str, str] = {}

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(request, timeout=30):
        captured_headers.update(dict(request.headers))
        return FakeResponse(payload)

    monkeypatch.setattr(
        "sentiment_engine.ingestion.posts_external_provider.urlopen", fake_urlopen
    )

    posts = load_truthsocial_provider_posts(
        "https://example.test/api.json",
        source_name="provider_api",
        source_provider=TRUTHSOCIAL_PROVIDER,
        request_headers={"x-api-key": "key-123", "Authorization": "Bearer abc"},
    )

    normalised = {name.lower(): value for name, value in captured_headers.items()}
    assert normalised["x-api-key"] == "key-123"
    assert normalised["authorization"] == "Bearer abc"
    assert len(posts) == 1
    assert posts[0].post_id == "provider-301"


def test_provider_adapter_handles_extensionless_remote_url(monkeypatch) -> None:
    payload = b'[{"id":"provider-302","created_at":"2026-01-12T12:05:00Z","text":"Extensionless endpoint","author":{"id":"realDonaldTrump"}}]'

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(request, timeout=30):
        return FakeResponse(payload)

    monkeypatch.setattr(
        "sentiment_engine.ingestion.posts_external_provider.urlopen", fake_urlopen
    )

    posts = load_truthsocial_provider_posts(
        "https://example.test/api",
        source_name="provider_api",
        source_provider=TRUTHSOCIAL_PROVIDER,
    )

    assert len(posts) == 1
    assert posts[0].post_id == "provider-302"
