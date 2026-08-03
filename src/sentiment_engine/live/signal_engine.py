from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from sentiment_engine.config import EngineConfig
from sentiment_engine.constants import (
    FEATURE_SET_VERSION,
    IMPACT_MODEL_VERSION,
    RULE_CLASSIFIER_VERSION,
    WHIPSAW_MODEL_VERSION,
)
from sentiment_engine.models import taxonomy as tx
from sentiment_engine.models.whipsaw import RISK_HARD, RISK_SOFT, RISK_WATCH
from sentiment_engine.schemas import SignalRecord

NO_TRADE_SIGNAL = "NO_TRADE"
NEUTRAL_SIGNAL = "NEUTRAL"
BULLISH_SIGNAL = "BULLISH"
BEARISH_SIGNAL = "BEARISH"
ALLOW_ACTION = "ALLOW"
BLOCK_ACTION = "BLOCK_NEW_ENTRIES"
REDUCE_ACTION = "REDUCE_SIZE"
NO_TRADE_DATA_STALE = "NO_TRADE_DATA_STALE"
NO_TRADE_LOW_CONFIDENCE = "NO_TRADE_LOW_CONFIDENCE"
WHIPSAW_RISK = "WHIPSAW_RISK"
VOLATILITY_RISK = "VOLATILITY_RISK"


def signal_from_scored_event(
    row: pd.Series | dict[str, Any],
    config: EngineConfig,
    *,
    market_data_stale: bool = False,
    provider_stale: bool = False,
) -> SignalRecord:
    payload = dict(row)
    risk_level = str(payload.get("whipsaw_risk_level", "NONE"))
    sentiment_label = str(payload.get("rule_sentiment_label", tx.SENTIMENT_LOW_CONFIDENCE))
    confidence = float(payload.get("rule_sentiment_confidence", 0.0))
    direction_signal = _direction_signal(sentiment_label, confidence, risk_level, config)
    kill_switch = _kill_switch(direction_signal, risk_level, config, provider_stale, market_data_stale)
    if provider_stale or market_data_stale:
        sentiment_label = tx.SENTIMENT_LOW_CONFIDENCE
        direction_signal = NO_TRADE_SIGNAL
    signal = SignalRecord(
        event_id=str(payload["event_id"]),
        post_id=str(payload["post_id"]),
        source_provider=str(payload["source_provider"]),
        created_at_utc=payload["created_at_utc"],
        received_at_utc=payload["received_at_utc"],
        generated_at_utc=datetime.now(UTC),
        text_clean=str(payload["text_clean"]),
        sentiment_label=sentiment_label,
        sentiment_confidence=round(confidence, 6),
        topic_labels=_list(payload.get("rule_topic_labels")),
        topic_confidence=_dict(payload.get("rule_topic_confidence")),
        tradeability_label=_tradeability(payload, risk_level, provider_stale, market_data_stale),
        direction_signal=direction_signal,
        p_direction=_direction_probabilities(payload),
        expected_delta_ticks=_expected_delta_ticks(payload),
        risk={
            "whipsaw_risk_level": risk_level,
            "whipsaw_score": float(payload.get("whipsaw_score", 0.0)),
            "volatility_risk_score": float(payload.get("headline_risk_score", 0.0)),
            "p_next_post_15m": float(payload.get("p_next_post_15m", 0.0)),
            "p_next_post_30m": float(payload.get("p_next_post_30m", 0.0)),
            "p_next_post_60m": float(payload.get("p_next_post_60m", 0.0)),
        },
        kill_switch=kill_switch,
        data_quality={
            "feed_lag_ms": _feed_lag_ms(payload),
            "market_data_lag_ms": 0,
            "market_data_stale": market_data_stale,
            "features_complete": not (provider_stale or market_data_stale),
        },
        model_versions={
            "classifier": RULE_CLASSIFIER_VERSION,
            "impact_model": IMPACT_MODEL_VERSION,
            "whipsaw_model": WHIPSAW_MODEL_VERSION,
            "feature_set": FEATURE_SET_VERSION,
        },
        explanation={
            "top_features": _top_features(payload),
            "contradicting_post_ids": _list(payload.get("contradicting_post_ids")),
            "human_readable_reason": _human_reason(payload, kill_switch),
        },
    )
    return signal


def latest_signal_from_scores(scored_events: pd.DataFrame, config: EngineConfig) -> SignalRecord:
    if scored_events.empty:
        raise ValueError("Cannot compose latest signal from empty scored events")
    latest = scored_events.sort_values("received_at_utc").iloc[-1]
    return signal_from_scored_event(latest, config)


def _direction_signal(sentiment: str, confidence: float, risk_level: str, config: EngineConfig) -> str:
    if risk_level in {RISK_SOFT, RISK_HARD}:
        return NO_TRADE_SIGNAL
    if confidence < config.thresholds.min_direction_confidence:
        return NO_TRADE_SIGNAL
    if sentiment == tx.SENTIMENT_BULLISH:
        return BULLISH_SIGNAL
    if sentiment in {tx.SENTIMENT_BEARISH, tx.SENTIMENT_GEOPOLITICAL}:
        return BEARISH_SIGNAL if sentiment == tx.SENTIMENT_BEARISH else NO_TRADE_SIGNAL
    return NEUTRAL_SIGNAL


def _kill_switch(
    direction_signal: str,
    risk_level: str,
    config: EngineConfig,
    provider_stale: bool,
    market_data_stale: bool,
) -> dict[str, Any]:
    if provider_stale or market_data_stale:
        return {
            "action": BLOCK_ACTION,
            "risk_multiplier": 0.0,
            "ttl_seconds": 0,
            "reason_codes": [NO_TRADE_DATA_STALE],
        }
    if risk_level == RISK_HARD:
        return {
            "action": str(config.live_actions["hard_risk_action"]),
            "risk_multiplier": 0.0,
            "ttl_seconds": int(config.live_actions["hard_risk_ttl_seconds"]),
            "reason_codes": [WHIPSAW_RISK],
        }
    if risk_level == RISK_SOFT:
        action = str(config.live_actions["soft_risk_action"])
        return {
            "action": action,
            "risk_multiplier": 0.5 if action == REDUCE_ACTION else 0.0,
            "ttl_seconds": 1800,
            "reason_codes": [WHIPSAW_RISK],
        }
    if direction_signal == NO_TRADE_SIGNAL:
        return {
            "action": BLOCK_ACTION,
            "risk_multiplier": 0.0,
            "ttl_seconds": 900,
            "reason_codes": [NO_TRADE_LOW_CONFIDENCE],
        }
    reason_codes = [RISK_WATCH] if risk_level == RISK_WATCH else []
    return {"action": ALLOW_ACTION, "risk_multiplier": 1.0, "ttl_seconds": 900, "reason_codes": reason_codes}


def _tradeability(
    payload: dict[str, Any],
    risk_level: str,
    provider_stale: bool,
    market_data_stale: bool,
) -> str:
    if provider_stale or market_data_stale:
        return tx.TRADEABILITY_AMBIGUOUS
    if risk_level in {RISK_SOFT, RISK_HARD}:
        return tx.TRADEABILITY_NO_TRADE_WHIPSAW
    return str(payload.get("rule_tradeability_label", tx.TRADEABILITY_AMBIGUOUS))


def _direction_probabilities(payload: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for horizon in (5, 15, 30):
        up = float(payload.get(f"p_up_{horizon}m", 1.0 / 3.0))
        down = float(payload.get(f"p_down_{horizon}m", 1.0 / 3.0))
        flat = float(payload.get(f"p_flat_{horizon}m", 1.0 / 3.0))
        total = up + down + flat
        if total <= 0:
            up = down = flat = 1.0 / 3.0
            total = 1.0
        output[f"up_{horizon}m"] = round(up / total, 6)
        output[f"down_{horizon}m"] = round(down / total, 6)
        output[f"flat_{horizon}m"] = round(flat / total, 6)
    return output


def _expected_delta_ticks(payload: dict[str, Any]) -> dict[str, float]:
    return {
        f"{horizon}m": float(payload.get(f"expected_delta_{horizon}m_ticks", 0.0))
        for horizon in (5, 15, 30)
    }


def _feed_lag_ms(payload: dict[str, Any]) -> int:
    created = pd.Timestamp(payload["created_at_utc"])
    received = pd.Timestamp(payload["received_at_utc"])
    return int((received - created).total_seconds() * 1000)


def _top_features(payload: dict[str, Any]) -> list[str]:
    candidates = {
        "headline_risk_score": float(payload.get("headline_risk_score", 0.0)),
        "market_relevance_score": float(payload.get("market_relevance_score", 0.0)),
        "text_contradiction_score": float(payload.get("text_contradiction_score", 0.0)),
        "post_burst_score": float(payload.get("post_burst_score", 0.0)),
        "direction_flip_score": float(payload.get("direction_flip_score", 0.0)),
    }
    return [key for key, value in sorted(candidates.items(), key=lambda item: item[1], reverse=True) if value > 0][:3]


def _human_reason(payload: dict[str, Any], kill_switch: dict[str, Any]) -> str:
    reason = str(payload.get("whipsaw_reason", "no material risk component"))
    return f"{kill_switch['action']}: {reason}"


def _list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dict(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): float(val) for key, val in value.items() if val is not None}
    return {}
