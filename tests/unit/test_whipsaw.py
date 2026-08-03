from __future__ import annotations

from sentiment_engine.models import taxonomy as tx
from sentiment_engine.models.whipsaw import _as_topic_list, _market_relevance_score


def test_whipsaw_topic_parser_splits_pipe_separated_labels() -> None:
    assert _as_topic_list("trade_policy|fx_usd") == ["trade_policy", "fx_usd"]


def test_whipsaw_relevance_includes_fx_and_gold_topics() -> None:
    assert _market_relevance_score([tx.TOPIC_FX_USD]) > 0
    assert _market_relevance_score([tx.TOPIC_GOLD_METALS]) > 0
