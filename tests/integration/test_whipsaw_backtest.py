from __future__ import annotations

from sentiment_engine.backtest.simulator import run_kill_switch_backtest
from sentiment_engine.config import load_config
from sentiment_engine.features.text_features import enrich_with_text_features
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.whipsaw import score_whipsaw_events
from sentiment_engine.research.events import build_event_dataset


def test_whipsaw_scores_feed_kill_switch_backtest() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    labeled = enrich_with_text_features(events)
    scored = score_whipsaw_events(labeled, config)

    first = scored.loc[scored["post_id"].eq("fixture-001")].iloc[0]
    geo = scored.loc[scored["post_id"].eq("fixture-006")].iloc[0]
    assert first["whipsaw_risk_level"] == "SOFT_RISK"
    assert geo["whipsaw_risk_level"] == "HARD_KILL"

    _trades, report = run_kill_switch_backtest(config.paths.trades_fixture, scored, config)
    assert report["net_pnl_after_usd"] > report["net_pnl_before_usd"]
    assert report["filtered_trades"] == 2
    assert report["reduced_trades"] == 1
