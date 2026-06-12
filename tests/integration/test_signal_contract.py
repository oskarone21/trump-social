from __future__ import annotations

from sentiment_engine.config import load_config
from sentiment_engine.features.text_features import enrich_with_text_features
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.live.signal_engine import latest_signal_from_scores
from sentiment_engine.models.whipsaw import score_whipsaw_events
from sentiment_engine.research.events import build_event_dataset


def test_latest_signal_matches_contract_shape() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    labeled = enrich_with_text_features(events)
    scored = score_whipsaw_events(labeled, config)

    signal = latest_signal_from_scores(scored, config)
    payload = signal.model_dump(mode="json")

    assert payload["event_id"]
    assert payload["post_id"] == "fixture-008"
    assert payload["kill_switch"]["action"] == "ALLOW"
    assert payload["risk"]["whipsaw_risk_level"] == "WATCH"
    for horizon in (5, 15, 30):
        total = (
            payload["p_direction"][f"up_{horizon}m"]
            + payload["p_direction"][f"down_{horizon}m"]
            + payload["p_direction"][f"flat_{horizon}m"]
        )
        assert abs(total - 1.0) < 0.00001
