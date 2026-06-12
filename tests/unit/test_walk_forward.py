from __future__ import annotations

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.baselines import build_labeled_events
from sentiment_engine.models.walk_forward import evaluate_walk_forward_classifiers
from sentiment_engine.research.events import build_event_dataset


def test_walk_forward_report_uses_chronological_embargoed_folds(tmp_path) -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    labeled = build_labeled_events(build_event_dataset(posts, bars, config).events)

    report = evaluate_walk_forward_classifiers(
        labeled,
        report_dir=tmp_path / "reports",
        seed=config.project.seed,
        min_train_rows=3,
        test_window_rows=1,
        step_rows=1,
        embargo_rows=1,
    )

    assert report["status"] == "evaluated"
    assert report["fold_count"] == 4
    assert report["split_method"] == "expanding_walk_forward_with_row_embargo"
    assert report["folds"][0]["train_rows"] == 3
    assert report["folds"][0]["test_rows"] == 1
    assert report["model_summary"]["tfidf_logreg"]["status"] == "evaluated"
    assert report["model_summary"]["tfidf_logreg"]["prediction_count"] == 4
    assert (tmp_path / "reports" / "walk_forward_report.json").exists()
