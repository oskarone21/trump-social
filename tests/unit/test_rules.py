from __future__ import annotations

from sentiment_engine.models.rules import classify_text
from sentiment_engine.models.taxonomy import TOPIC_MEDIA, TOPIC_TECH


def test_ai_keyword_does_not_match_again() -> None:
    result = classify_text("Fake news media is at it again. So unfair!")
    assert TOPIC_MEDIA in result.topic_labels
    assert TOPIC_TECH not in result.topic_labels
