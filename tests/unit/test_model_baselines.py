from __future__ import annotations

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import load_market_csv
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.models.baselines import build_labeled_events, train_tradeability_baselines
from sentiment_engine.research.events import build_event_dataset


def test_tradeability_baselines_write_ml_and_dl_reports(tmp_path) -> None:
    config = load_config("configs/research.yaml")
    posts = load_fixture_posts(config.paths.posts_fixture)
    bars = load_market_csv(config.paths.market_fixture)
    events = build_event_dataset(posts, bars, config).events
    labeled = build_labeled_events(events)

    report = train_tradeability_baselines(
        labeled,
        report_dir=tmp_path / "reports",
        model_dir=tmp_path / "models",
        config_path="configs/research.yaml",
        seed=config.project.seed,
    )

    assert report["tfidf_logreg"]["macro_f1"] >= 0.0
    assert report["tfidf_logreg"]["probability_metrics"]["negative_log_loss"] >= 0.0
    assert report["tfidf_linear_svm"]["macro_f1"] >= 0.0
    assert report["tfidf_linear_svm"]["probability_metrics"]["negative_log_loss"] >= 0.0
    assert report["lightgbm"]["status"] in {"trained", "skipped"}
    assert report["neural_tfidf_mlp"]["status"] in {"trained", "skipped"}
    assert (tmp_path / "reports" / "classifier_baseline_report.json").exists()
    assert (tmp_path / "reports" / "lightgbm_baseline_report.json").exists()
    assert (tmp_path / "reports" / "neural_baseline_report.json").exists()
