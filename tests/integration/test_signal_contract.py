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

    assert "actual_minutes_to_next_post" not in scored.columns

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


def test_signal_contract_does_not_use_realized_target_deltas_as_predictions() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    labeled = enrich_with_text_features(events)
    scored = score_whipsaw_events(labeled, config)

    payload = latest_signal_from_scores(scored.iloc[[0]].assign(nq_delta_30m_ticks=999.0), config)
    output = payload.model_dump(mode="json")

    assert output["expected_delta_ticks"]["30m"] == 0.0
    assert output["p_direction"]["up_30m"] == 0.333333


def test_signal_contract_uses_explicit_prediction_columns() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    labeled = enrich_with_text_features(events)
    scored = score_whipsaw_events(labeled, config).copy()
    scored.loc[scored.index[0], "expected_delta_30m_ticks"] = 12.5
    scored.loc[scored.index[0], "p_up_30m"] = 0.7
    scored.loc[scored.index[0], "p_down_30m"] = 0.2
    scored.loc[scored.index[0], "p_flat_30m"] = 0.1

    signal = latest_signal_from_scores(scored.iloc[[0]], config)
    output = signal.model_dump(mode="json")

    assert output["expected_delta_ticks"]["30m"] == 12.5
    assert output["p_direction"]["up_30m"] == 0.7
