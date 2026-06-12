from __future__ import annotations

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.research.event_study import build_event_study_report
from sentiment_engine.research.events import build_event_dataset


def test_event_study_report_includes_path_dependent_target_summary() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events

    report = build_event_study_report(events)

    assert report["event_count"] == 8
    assert "target_summary" in report
    assert "realised_volatility" in report["target_summary"]
    assert "realised_volatility_30m_ticks" in report["target_summary"]["realised_volatility"]
    assert (
        "max_favourable_excursion_30m_ticks"
        in report["target_summary"]["max_favourable_excursion"]
    )
    assert report["cluster_summary"]["cluster_count"] == 4
    assert report["cluster_summary"]["isolated_event_count"] == 1
    assert report["cluster_summary"]["burst_event_count"] == 7
    assert "mean_realised_volatility_30m_ticks" in next(iter(report["segments"].values()))
