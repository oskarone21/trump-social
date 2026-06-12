from __future__ import annotations

from dataclasses import dataclass
import re

from sentiment_engine.models import taxonomy as tx


@dataclass(frozen=True)
class RuleClassification:
    sentiment_label: str
    sentiment_confidence: float
    topic_labels: list[str]
    topic_confidence: dict[str, float]
    tradeability_label: str
    reason_codes: list[str]


def classify_text(text: str) -> RuleClassification:
    lower = text.lower()
    topics = _topic_scores(lower)
    sentiment, confidence, reasons = _sentiment(lower, topics)
    tradeability = _tradeability(sentiment, confidence, topics)
    return RuleClassification(
        sentiment_label=sentiment,
        sentiment_confidence=confidence,
        topic_labels=list(topics.keys()) if topics else [tx.TOPIC_OTHER],
        topic_confidence=topics or {tx.TOPIC_OTHER: 0.5},
        tradeability_label=tradeability,
        reason_codes=reasons,
    )


def _topic_scores(lower_text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for topic, keywords in tx.TOPIC_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if _contains_keyword(lower_text, keyword))
        if hits:
            scores[topic] = round(min(0.95, 0.50 + 0.15 * hits), 4)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _sentiment(lower_text: str, topics: dict[str, float]) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    geo_hits = _keyword_hits(lower_text, tx.GEOPOLITICAL_KEYWORDS)
    bullish_hits = _keyword_hits(lower_text, tx.BULLISH_KEYWORDS)
    bearish_hits = _keyword_hits(lower_text, tx.BEARISH_KEYWORDS)
    neutral_hits = _keyword_hits(lower_text, tx.NEUTRAL_KEYWORDS)
    if geo_hits:
        reasons.extend([f"geo:{keyword}" for keyword in geo_hits])
        return tx.SENTIMENT_GEOPOLITICAL, _confidence(geo_hits, topics), reasons
    if bullish_hits and not bearish_hits:
        reasons.extend([f"bullish:{keyword}" for keyword in bullish_hits])
        return tx.SENTIMENT_BULLISH, _confidence(bullish_hits, topics), reasons
    if bearish_hits and not bullish_hits:
        reasons.extend([f"bearish:{keyword}" for keyword in bearish_hits])
        return tx.SENTIMENT_BEARISH, _confidence(bearish_hits, topics), reasons
    if bullish_hits and bearish_hits:
        reasons.extend(["mixed_bullish_bearish_keywords"])
        return tx.SENTIMENT_VOLATILITY_ONLY, 0.62, reasons
    if neutral_hits:
        reasons.extend([f"neutral:{keyword}" for keyword in neutral_hits])
        return tx.SENTIMENT_NEUTRAL, 0.68, reasons
    return tx.SENTIMENT_LOW_CONFIDENCE, 0.45, ["no_market_keyword"]


def _tradeability(sentiment: str, confidence: float, topics: dict[str, float]) -> str:
    if confidence < 0.50:
        return tx.TRADEABILITY_AMBIGUOUS
    if sentiment == tx.SENTIMENT_GEOPOLITICAL:
        return tx.TRADEABILITY_VOLATILITY_ONLY
    if sentiment in {tx.SENTIMENT_BULLISH, tx.SENTIMENT_BEARISH}:
        market_topics = {
            tx.TOPIC_CHINA_TARIFFS,
            tx.TOPIC_FED_MONETARY,
            tx.TOPIC_EQUITIES,
            tx.TOPIC_TECH,
            tx.TOPIC_TAX_FISCAL,
        }
        return (
            tx.TRADEABILITY_DIRECTIONAL
            if market_topics.intersection(topics)
            else tx.TRADEABILITY_AMBIGUOUS
        )
    if sentiment == tx.SENTIMENT_VOLATILITY_ONLY:
        return tx.TRADEABILITY_VOLATILITY_ONLY
    if sentiment == tx.SENTIMENT_NEUTRAL:
        return tx.TRADEABILITY_NO_IMPACT
    return tx.TRADEABILITY_AMBIGUOUS


def _keyword_hits(lower_text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if _contains_keyword(lower_text, keyword)]


def _contains_keyword(lower_text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in lower_text
    return re.search(rf"\b{re.escape(keyword)}\b", lower_text) is not None


def _confidence(keyword_hits: list[str], topics: dict[str, float]) -> float:
    topic_boost = max(topics.values()) if topics else 0.0
    return round(min(0.95, 0.55 + 0.10 * len(keyword_hits) + 0.15 * topic_boost), 4)
