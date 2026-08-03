from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import run_truth_social_data_bootstrap as bootstrap


def _provider_args(**overrides):
    values = {
        "config": "configs/research.yaml",
        "provider_source": "provider.json",
        "provider_name": "truthsocial_provider",
        "provider_source_name": "provider_dump",
        "provider_limit": None,
        "provider_api_key": None,
        "provider_api_key_env": None,
        "provider_api_key_header": "x-api-key",
        "provider_header": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bootstrap_provider_fallback_records_system_exit(monkeypatch) -> None:
    report = {"steps": []}

    def fail_cli(_command):
        raise SystemExit(2)

    monkeypatch.setattr(bootstrap, "main", fail_cli)

    posts_path = bootstrap._run_provider_ingest(_provider_args(), report)

    assert posts_path == ""
    assert report["steps"] == [{"name": "provider", "status": "failed", "code": "2"}]


def test_bootstrap_provider_fallback_does_not_swallow_programming_errors(monkeypatch) -> None:
    report = {"steps": []}

    def fail_programming_error(_command):
        raise TypeError("bad command shape")

    monkeypatch.setattr(bootstrap, "main", fail_programming_error)

    with pytest.raises(TypeError, match="bad command shape"):
        bootstrap._run_provider_ingest(_provider_args(), report)
