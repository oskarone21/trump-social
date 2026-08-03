from __future__ import annotations

from sentiment_engine.models.rules import classify_text
from sentiment_engine.models.taxonomy import (
    TOPIC_CHINA_TARIFFS,
    TOPIC_EQUITIES,
    TOPIC_FX_USD,
    TOPIC_GOLD_METALS,
    TOPIC_IRAN_ENERGY,
    TOPIC_MEDIA,
    TOPIC_TECH,
    TOPIC_TRADE_POLICY,
)


def test_ai_keyword_does_not_match_again() -> None:
    result = classify_text("Fake news media is at it again. So unfair!")
    assert TOPIC_MEDIA in result.topic_labels
    assert TOPIC_TECH not in result.topic_labels


def test_market_critical_keywords_are_detected() -> None:
    result = classify_text(
        "China tarriffs, Iran oil risk, the US dollar, gold, S&P 500 and Nasdaq matter."
    )

    assert TOPIC_TRADE_POLICY in result.topic_labels
    assert TOPIC_CHINA_TARIFFS in result.topic_labels
    assert TOPIC_IRAN_ENERGY in result.topic_labels
    assert TOPIC_FX_USD in result.topic_labels
    assert TOPIC_GOLD_METALS in result.topic_labels
    assert TOPIC_EQUITIES in result.topic_labels
