from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from sentiment_engine.config import EngineConfig
from sentiment_engine.features.burst_features import add_burst_features
from sentiment_engine.models.contradiction import (
    contradiction_score,
    direction_flip_score,
    same_topic_score,
)
from sentiment_engine.models import taxonomy as tx
from sentiment_engine.models.timing import add_empirical_timing_features

RISK_NONE = "NONE"
RISK_WATCH = "WATCH"
RISK_SOFT = "SOFT_RISK"
RISK_HARD = "HARD_KILL"
WATCH_THRESHOLD = 0.25
DEFAULT_WEIGHTS = {
    "headline_risk": 0.35,
    "market_relevance": 0.30,
    "text_contradiction": 0.15,
    "post_burst": 0.10,
    "direction_flip": 0.07,
    "same_topic": 0.03,
    "volatility_regime": 0.00,
}


def score_whipsaw_events(labeled_events: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    enriched = add_empirical_timing_features(add_burst_features(labeled_events))
    enriched = enriched.sort_values("received_at_utc").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    timestamps = pd.to_datetime(enriched["received_at_utc"], utc=True)
    for idx, row in enriched.iterrows():
        window_start = timestamps.iloc[idx] - pd.Timedelta(minutes=config.windows.contradiction_window_minutes)
        prior = enriched[(timestamps < timestamps.iloc[idx]) & (timestamps >= window_start)]
        components, contradicting_ids = _components(row, prior)
        score = _weighted_score(components)
        risk_level = _risk_level(score, config)
        record = row.to_dict()
        record.update(components)
        record.update(
            {
                "whipsaw_score": score,
                "whipsaw_risk_level": risk_level,
                "contradicting_post_ids": contradicting_ids,
                "whipsaw_reason": _reason(components, contradicting_ids),
                "risk_ttl_seconds": _ttl_seconds(risk_level, config),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def build_whipsaw_report(scored_events: pd.DataFrame) -> dict[str, Any]:
    soft_pred = scored_events["whipsaw_risk_level"].isin([RISK_SOFT, RISK_HARD])
    hard_pred = scored_events["whipsaw_risk_level"].eq(RISK_HARD)
    actual = scored_events["market_whipsaw_flag"].astype(bool)
    return {
        "row_count": int(len(scored_events)),
        "risk_level_counts": scored_events["whipsaw_risk_level"].value_counts().to_dict(),
        "soft_risk": _binary_metrics(actual, soft_pred),
        "hard_kill": _binary_metrics(actual, hard_pred),
        "mean_whipsaw_score_actual_true": _safe_mean(scored_events.loc[actual, "whipsaw_score"]),
        "mean_whipsaw_score_actual_false": _safe_mean(scored_events.loc[~actual, "whipsaw_score"]),
        "methodology_notes": [
            "Actual market whipsaw labels are post-event research targets.",
            "Risk scores use text/burst/timing fields and do not require post-event prices.",
            "Volatility regime score is zero until point-in-time market features are available.",
            "Fixture metrics are smoke checks; real thresholds require walk-forward validation.",
        ],
    }


def _components(row: pd.Series, prior: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    current_topics = _as_topic_list(row.get("rule_topic_labels"))
    current_sentiment = str(row.get("rule_sentiment_label"))
    text_score = 0.0
    topic_score = 0.0
    flip_score = 0.0
    contradicting_ids: list[str] = []
    for prior_row in prior.to_dict("records"):
        prior_topics = _as_topic_list(prior_row.get("rule_topic_labels"))
        prior_sentiment = str(prior_row.get("rule_sentiment_label"))
        pair_text_score = contradiction_score(
            current_sentiment,
            prior_sentiment,
            current_topics,
            prior_topics,
        )
        pair_topic_score = same_topic_score(current_topics, prior_topics)
        pair_flip_score = direction_flip_score(current_sentiment, prior_sentiment)
        if pair_text_score > text_score:
            text_score = pair_text_score
            contradicting_ids = [str(prior_row["post_id"])]
        topic_score = max(topic_score, pair_topic_score)
        flip_score = max(flip_score, pair_flip_score)
    burst_score = min(1.0, float(row.get("posts_last_60m", 0)) / 3.0)
    headline_score = _headline_risk_score(current_sentiment)
    relevance_score = _market_relevance_score(current_topics)
    return (
        {
            "headline_risk_score": round(headline_score, 6),
            "market_relevance_score": round(relevance_score, 6),
            "text_contradiction_score": round(text_score, 6),
            "same_topic_score": round(topic_score, 6),
            "post_burst_score": round(burst_score, 6),
            "direction_flip_score": round(flip_score, 6),
            "volatility_regime_score": 0.0,
        },
        contradicting_ids,
    )


def _weighted_score(components: dict[str, float]) -> float:
    score = (
        DEFAULT_WEIGHTS["headline_risk"] * components["headline_risk_score"]
        + DEFAULT_WEIGHTS["market_relevance"] * components["market_relevance_score"]
        + DEFAULT_WEIGHTS["text_contradiction"] * components["text_contradiction_score"]
        + DEFAULT_WEIGHTS["post_burst"] * components["post_burst_score"]
        + DEFAULT_WEIGHTS["direction_flip"] * components["direction_flip_score"]
        + DEFAULT_WEIGHTS["same_topic"] * components["same_topic_score"]
        + DEFAULT_WEIGHTS["volatility_regime"] * components["volatility_regime_score"]
    )
    return round(min(1.0, max(0.0, score)), 6)


def _risk_level(score: float, config: EngineConfig) -> str:
    if score >= config.thresholds.hard_whipsaw_threshold:
        return RISK_HARD
    if score >= config.thresholds.soft_whipsaw_threshold:
        return RISK_SOFT
    if score >= WATCH_THRESHOLD:
        return RISK_WATCH
    return RISK_NONE


def _ttl_seconds(risk_level: str, config: EngineConfig) -> int:
    if risk_level == RISK_HARD:
        return int(config.live_actions["hard_risk_ttl_seconds"])
    if risk_level == RISK_SOFT:
        return 1800
    if risk_level == RISK_WATCH:
        return 900
    return 0


def _reason(components: dict[str, float], contradicting_ids: list[str]) -> str:
    top_component = max(components.items(), key=lambda item: item[1])
    if contradicting_ids:
        return f"{top_component[0]}={top_component[1]:.2f}; prior={','.join(contradicting_ids)}"
    return f"{top_component[0]}={top_component[1]:.2f}"


def _as_topic_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return []


def _headline_risk_score(sentiment: str) -> float:
    if sentiment in {tx.SENTIMENT_BEARISH, tx.SENTIMENT_GEOPOLITICAL}:
        return 1.0
    if sentiment == tx.SENTIMENT_VOLATILITY_ONLY:
        return 0.65
    if sentiment == tx.SENTIMENT_BULLISH:
        return 0.20
    return 0.0


def _market_relevance_score(topics: list[str]) -> float:
    high_relevance_topics = {
        tx.TOPIC_CHINA_TARIFFS,
        tx.TOPIC_TRADE_POLICY,
        tx.TOPIC_FED_MONETARY,
        tx.TOPIC_IRAN_ENERGY,
        tx.TOPIC_MIDDLE_EAST,
        tx.TOPIC_EQUITIES,
        tx.TOPIC_TECH,
        tx.TOPIC_TAX_FISCAL,
    }
    if not topics:
        return 0.0
    matches = high_relevance_topics.intersection(topics)
    return min(1.0, 0.40 + 0.20 * len(matches)) if matches else 0.0


def _binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    true_positive = int((actual & predicted).sum())
    false_positive = int((~actual & predicted).sum())
    true_negative = int((~actual & ~predicted).sum())
    false_negative = int((actual & ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    }


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.mean()), 6)
