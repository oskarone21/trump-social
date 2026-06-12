import pytest

from sentiment_engine.cli import _build_provider_headers, _parse_headers


def test_parse_headers_converts_key_value_args() -> None:
    parsed = _parse_headers(["Authorization: Bearer 1", "x-api-key:key-abc"])
    assert parsed == [("Authorization", "Bearer 1"), ("x-api-key", "key-abc")]


def test_parse_headers_rejects_invalid_format() -> None:
    with pytest.raises(SystemExit):
        _parse_headers(["not-a-header"])


def test_build_provider_headers_uses_env_when_key_not_supplied(monkeypatch) -> None:
    monkeypatch.setenv("TRUTH_SOCIAL_TEST_KEY", "env-key-123")

    headers = _build_provider_headers(
        extra_headers=[],
        api_key=None,
        api_key_env="TRUTH_SOCIAL_TEST_KEY",
        api_key_header="X-API-KEY",
        is_remote=True,
    )

    assert headers["X-API-KEY"] == "env-key-123"
