from __future__ import annotations

from sentiment_engine.models import taxonomy as tx

DIRECTIONAL_SENTIMENTS = {
    tx.SENTIMENT_BULLISH: 1,
    tx.SENTIMENT_BEARISH: -1,
    tx.SENTIMENT_GEOPOLITICAL: -1,
    tx.SENTIMENT_VOLATILITY_ONLY: 0,
    tx.SENTIMENT_NEUTRAL: 0,
    tx.SENTIMENT_LOW_CONFIDENCE: 0,
}


def same_topic_score(current_topics: list[str], prior_topics: list[str]) -> float:
    current = set(current_topics)
    prior = set(prior_topics)
    if not current or not prior:
        return 0.0
    return len(current.intersection(prior)) / len(current.union(prior))


def contradiction_score(
    current_sentiment: str,
    prior_sentiment: str,
    current_topics: list[str],
    prior_topics: list[str],
) -> float:
    topic_overlap = same_topic_score(current_topics, prior_topics)
    current_direction = DIRECTIONAL_SENTIMENTS.get(current_sentiment, 0)
    prior_direction = DIRECTIONAL_SENTIMENTS.get(prior_sentiment, 0)
    if current_direction == 0 or prior_direction == 0:
        return 0.0
    if current_direction == prior_direction:
        return 0.0
    return round(max(0.0, min(1.0, 0.40 + 0.60 * topic_overlap)), 4)


def direction_flip_score(current_sentiment: str, prior_sentiment: str) -> float:
    current_direction = DIRECTIONAL_SENTIMENTS.get(current_sentiment, 0)
    prior_direction = DIRECTIONAL_SENTIMENTS.get(prior_sentiment, 0)
    if current_direction == 0 or prior_direction == 0:
        return 0.0
    return 1.0 if current_direction != prior_direction else 0.0
