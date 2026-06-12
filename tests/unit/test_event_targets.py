from __future__ import annotations

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.research.events import build_event_dataset


def test_fixture_event_targets_are_built_without_skips() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    result = build_event_dataset(posts, bars, config)

    assert len(result.events) == 8
    assert result.skipped_posts == []
    assert result.events["received_at_utc"].str.endswith("Z").all()
    assert result.events["market_whipsaw_flag"].sum() == 3
    assert result.events["event_cluster_id"].nunique() == 4
    assert result.events["is_burst_event"].sum() == 7
    assert result.events["is_isolated_event"].sum() == 1
    assert result.events.iloc[0]["event_cluster_size"] == 3
    for horizon in (5, 15, 30):
        assert f"max_favourable_excursion_{horizon}m_ticks" in result.events.columns
        assert f"max_adverse_excursion_{horizon}m_ticks" in result.events.columns
        assert f"realised_range_{horizon}m_ticks" in result.events.columns
        assert f"realised_volatility_{horizon}m_ticks" in result.events.columns
        assert (result.events[f"realised_volatility_{horizon}m_ticks"] >= 0).all()


def test_target_alignment_uses_first_bar_after_received_timestamp() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    result = build_event_dataset(posts, bars, config)
    first = result.events.iloc[0]

    assert first["post_id"] == "fixture-001"
    assert first["received_at_utc"] == "2026-01-02T14:30:17Z"
    assert first["aligned_bar_ts_utc"] == "2026-01-02T14:31:00Z"
    assert first["max_favourable_excursion_5m_ticks"] >= 0.0
    assert first["max_adverse_excursion_5m_ticks"] >= 0.0
    assert bool(first["overlaps_next_event_window"]) is True
    assert bool(first["overlaps_prior_event_window"]) is False
