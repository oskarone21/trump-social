from __future__ import annotations

import pandas as pd
import pytest

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.baselines import build_labeled_events
from sentiment_engine.research.events import build_event_dataset
from sentiment_engine.labels.review import build_label_queue, load_reviewed_labels


def test_label_queue_excludes_post_event_target_columns() -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    labeled = build_labeled_events(events)

    queue = build_label_queue(labeled, limit=3)

    assert len(queue) == 3
    assert "nq_delta_30m_ticks" not in queue.columns
    assert "market_whipsaw_flag" not in queue.columns
    assert queue["human_sentiment_label"].eq("").all()
    assert queue["rule_sentiment_label"].ne("").all()


def test_reviewed_labels_validate_and_report_agreement(tmp_path) -> None:
    source = tmp_path / "reviewed.csv"
    pd.DataFrame(
        [
            _review_row("event-1", "reviewer-a"),
            _review_row("event-1", "reviewer-b"),
            _review_row("event-2", "reviewer-a", sentiment="neutral"),
        ]
    ).to_csv(source, index=False)

    labels, audit = load_reviewed_labels(source, label_version="human_test_v1")

    assert labels["label_version"].unique().tolist() == ["human_test_v1"]
    assert audit["row_count"] == 3
    assert audit["unique_event_count"] == 2
    assert audit["multi_review_event_count"] == 1
    assert audit["agreement"]["sentiment_unanimous_rate"] == 1.0


def test_reviewed_labels_reject_invalid_values(tmp_path) -> None:
    source = tmp_path / "reviewed.csv"
    row = _review_row("event-1", "reviewer-a")
    row["human_sentiment_label"] = "positive"
    pd.DataFrame([row]).to_csv(source, index=False)

    with pytest.raises(ValueError, match="invalid human_sentiment_label"):
        load_reviewed_labels(source, label_version="human_test_v1")


def _review_row(event_id: str, reviewer_id: str, *, sentiment: str = "bullish_market") -> dict:
    return {
        "event_id": event_id,
        "post_id": f"post-{event_id}",
        "human_sentiment_label": sentiment,
        "human_tradeability_label": "tradeable_directional",
        "human_topic_labels": "trade_policy|equities_direct",
        "market_relevance_label": "market_relevant",
        "contradiction_label": "none",
        "label_confidence": 0.8,
        "reviewer_id": reviewer_id,
        "reviewed_at_utc": "2026-06-12T12:00:00Z",
        "review_notes": "",
    }
