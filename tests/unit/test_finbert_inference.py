from __future__ import annotations

import pandas as pd

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.baselines import build_labeled_events
from sentiment_engine.models.finbert import score_finbert_sentiment
from sentiment_engine.research.events import build_event_dataset


class FakeFinbertClassifier:
    def __call__(self, texts: list[str], **_kwargs):
        return [
            [
                {"label": "positive", "score": 0.70},
                {"label": "neutral", "score": 0.20},
                {"label": "negative", "score": 0.10},
            ]
            for _text in texts
        ]


def test_finbert_inference_writes_scores_and_report(tmp_path) -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_labeled_events(build_event_dataset(posts, bars, config).events.head(2))

    report = score_finbert_sentiment(
        events,
        report_dir=tmp_path / "reports",
        classifier=FakeFinbertClassifier(),
    )
    scores = pd.read_parquet(tmp_path / "reports" / "finbert_scores.parquet")

    assert report["status"] == "scored"
    assert report["scored_rows"] == 2
    assert report["mode"] == "inference_only"
    assert report["label_counts"] == {"positive": 2}
    assert scores["finbert_label"].tolist() == ["positive", "positive"]
    assert scores["finbert_positive_score"].tolist() == [0.7, 0.7]


def test_finbert_inference_skips_empty_text_rows(tmp_path) -> None:
    events = pd.DataFrame(
        [
            {"event_id": "with-text", "post_id": "p1", "text_clean": "tariffs rally"},
            {"event_id": "empty", "post_id": "p2", "text_clean": ""},
        ]
    )

    report = score_finbert_sentiment(
        events,
        report_dir=tmp_path / "reports",
        classifier=FakeFinbertClassifier(),
    )

    assert report["status"] == "scored"
    assert report["scored_rows"] == 1
    assert report["skipped_empty_text_rows"] == 1
