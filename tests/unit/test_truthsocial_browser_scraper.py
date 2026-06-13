from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sentiment_engine.config import load_config
from sentiment_engine.cli import main
from sentiment_engine.ingestion.posts_external_provider import load_truthsocial_provider_posts
from sentiment_engine.ingestion.truthsocial_browser import (
    AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED,
    AUTH_STATUS_FIXTURE,
    AUTH_STATUS_MISSING_CREDENTIALS,
    TRUTHSOCIAL_BROWSER_PROVIDER,
    BrowserScraperSettings,
    load_truthsocial_status_rows,
    normalise_truthsocial_status_rows,
    run_truthsocial_browser_scrape_once,
    run_truthsocial_fixture_scrape,
)
from sentiment_engine.cli import _latest_signal_with_provider_health


FIXTURE_PATH = "data/fixtures/truthsocial_statuses_sample.json"


def _settings(tmp_path) -> BrowserScraperSettings:
    return BrowserScraperSettings(
        storage_state_path=tmp_path / "storage.json",
        canonical_out=tmp_path / "truthsocial_browser_posts.parquet",
        report_out=tmp_path / "truthsocial_browser_scraper_report.json",
        raw_dir=tmp_path / "raw",
    )


def test_truthsocial_status_fixture_normalises_to_post_records() -> None:
    rows = load_truthsocial_status_rows(FIXTURE_PATH)
    posts = normalise_truthsocial_status_rows(
        rows,
        received_at_utc=datetime(2026, 6, 13, 15, 27, 8, tzinfo=UTC),
    )

    assert len(posts) == 2
    assert posts[0].source_provider == TRUTHSOCIAL_BROWSER_PROVIDER
    assert posts[0].author_id == "107780257626128497"
    assert posts[0].text_clean == "NASDAQ and jobs numbers are looking strong. America is winning!"
    assert posts[0].has_image is True
    assert posts[0].engagement_metrics_json["favourites_count"] == 912
    assert posts[1].post_type == "reply"
    assert posts[1].parent_post_id == "114315232218538121"


def test_truthsocial_status_normalisation_dedupes_repeated_rows() -> None:
    rows = load_truthsocial_status_rows(FIXTURE_PATH)
    posts = normalise_truthsocial_status_rows(
        [rows[0], rows[0]],
        received_at_utc=datetime(2026, 6, 13, 15, 27, 8, tzinfo=UTC),
    )

    assert len(posts) == 1
    assert posts[0].post_id == "114315232218538121"


def test_fixture_scrape_writes_raw_canonical_and_safe_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRUTHSOCIAL_USERNAME", "user@example.test")
    monkeypatch.setenv("TRUTHSOCIAL_PASSWORD", "super-secret-password")
    settings = _settings(tmp_path)

    result = run_truthsocial_fixture_scrape(
        fixture_path=FIXTURE_PATH,
        settings=settings,
        checked_at_utc=datetime(2026, 6, 13, 15, 27, 9, tzinfo=UTC),
    )

    assert result.report["auth_status"] == AUTH_STATUS_FIXTURE
    assert result.report["is_stale"] is False
    assert result.canonical_out.exists()
    assert result.raw_out.exists()
    assert result.report_out.exists()

    report_text = result.report_out.read_text(encoding="utf-8")
    assert "super-secret-password" not in report_text
    assert "storageState" not in report_text
    assert len(pd.read_parquet(result.canonical_out)) == 2


def test_browser_scrape_fails_closed_without_usable_live_session(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRUTHSOCIAL_USERNAME", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PASSWORD", raising=False)

    result = run_truthsocial_browser_scrape_once(settings=_settings(tmp_path))

    assert result.report["auth_status"] in {
        AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED,
        AUTH_STATUS_MISSING_CREDENTIALS,
    }
    assert result.report["is_stale"] is True
    assert result.report["browser_status"] == "failed"


def test_scraper_canonical_output_can_be_reingested_as_provider_snapshot(tmp_path) -> None:
    result = run_truthsocial_fixture_scrape(
        fixture_path=FIXTURE_PATH,
        settings=_settings(tmp_path),
        checked_at_utc=datetime(2026, 6, 13, 15, 27, 9, tzinfo=UTC),
    )

    posts = load_truthsocial_provider_posts(
        result.canonical_out,
        source_name="truthsocial_browser_live",
        source_provider=TRUTHSOCIAL_BROWSER_PROVIDER,
    )

    assert len(posts) == 2
    assert posts[0].post_id == "114315232218538121"
    assert posts[0].source_provider == TRUTHSOCIAL_BROWSER_PROVIDER


def test_enabled_browser_scraper_stale_report_blocks_latest_signal(tmp_path) -> None:
    config = load_config("configs/research.yaml")
    config.paths.report_dir = tmp_path
    config.sources["posts"]["truthsocial_browser"]["enabled"] = True
    (tmp_path / "truthsocial_browser_scraper_report.json").write_text(
        json.dumps(
            {
                "auth_status": AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED,
                "is_stale": True,
                "schema_drift_detected": False,
            }
        ),
        encoding="utf-8",
    )
    scored = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "post_id": "post-1",
                "source_provider": TRUTHSOCIAL_BROWSER_PROVIDER,
                "created_at_utc": datetime(2026, 6, 13, 15, 27, tzinfo=UTC),
                "received_at_utc": datetime(2026, 6, 13, 15, 27, 1, tzinfo=UTC),
                "text_clean": "Test",
                "rule_sentiment_label": "bullish_market",
                "rule_sentiment_confidence": 0.95,
                "rule_topic_labels": ["equities_direct"],
                "rule_topic_confidence": {"equities_direct": 0.9},
                "rule_tradeability_label": "tradeable_directional",
                "whipsaw_risk_level": "NONE",
                "whipsaw_score": 0.0,
                "headline_risk_score": 0.1,
                "nq_delta_5m_ticks": 5.0,
                "nq_delta_15m_ticks": 7.0,
                "nq_delta_30m_ticks": 9.0,
            }
        ]
    )

    signal = _latest_signal_with_provider_health(scored, config)

    assert signal.direction_signal == "NO_TRADE"
    assert signal.kill_switch["action"] == "BLOCK_NEW_ENTRIES"
    assert signal.data_quality["features_complete"] is False


def test_cli_fixture_scrape_output_passes_provider_freshness(tmp_path) -> None:
    config_path = tmp_path / "research_test.yaml"
    report_dir = tmp_path / "reports"
    config_path.write_text(
        "\n".join(
            [
                f"extends: {Path('configs/research.yaml').resolve()}",
                "paths:",
                f"  interim_dir: {tmp_path / 'interim'}",
                f"  processed_dir: {tmp_path / 'processed'}",
                f"  report_dir: {report_dir}",
                f"  model_dir: {tmp_path / 'models'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "truthsocial_browser_posts.parquet"
    scraper_report = tmp_path / "truthsocial_browser_scraper_report.json"

    main(
        [
            "--config",
            str(config_path),
            "scrape-truthsocial-live",
            "--fixture",
            FIXTURE_PATH,
            "--once",
            "--out",
            str(output),
            "--report-out",
            str(scraper_report),
        ]
    )
    main(
        [
            "--config",
            str(config_path),
            "check-provider-freshness",
            "--source",
            str(output),
            "--posts",
            str(output),
            "--source-provider",
            TRUTHSOCIAL_BROWSER_PROVIDER,
            "--stale-after-minutes",
            "1000000",
        ]
    )

    freshness_report = json.loads(
        (report_dir / "provider_posts_freshness_report.json").read_text(encoding="utf-8")
    )
    assert freshness_report["local_provider"]["required_columns_present"] is True
    assert freshness_report["local_provider"]["schema_drift_detected"] is False
