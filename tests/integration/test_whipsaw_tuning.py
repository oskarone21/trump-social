from __future__ import annotations

from sentiment_engine.config import load_config
from sentiment_engine.features.text_features import enrich_with_text_features
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.tuning import tune_whipsaw_parameters
from sentiment_engine.models.whipsaw import score_whipsaw_events
from sentiment_engine.research.events import build_event_dataset


def test_optuna_whipsaw_tuning_reports_holdout_metrics(tmp_path) -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    scored = score_whipsaw_events(enrich_with_text_features(events), config)

    report = tune_whipsaw_parameters(
        scored,
        report_path=tmp_path / "whipsaw_tuning_report.json",
        n_trials=5,
        seed=7,
    )

    assert report["status"] == "completed"
    assert report["optimizer"] == "optuna.tpe"
    assert report["n_trials"] == 5
    assert "weights" in report["best_params"]
    assert "holdout_metrics_best" in report
