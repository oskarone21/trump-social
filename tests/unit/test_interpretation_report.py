from __future__ import annotations

from sentiment_engine.research.interpretation import (
    build_interpretation_report,
    render_interpretation_markdown,
)


def test_interpretation_report_marks_fixture_results_not_research_ready() -> None:
    payload = build_interpretation_report(
        {
            "classifier": {
                "test_rows": 3,
                "naive": _metric(0.25, 0.333333),
                "tfidf_logreg": _metric(0.666667, 0.666667),
                "neural_tfidf_mlp": {"status": "trained", "metrics": _metric(0.25, 0.333333)},
            },
            "archive": {"row_count": 33899},
            "market": {"row_count": 340, "valid_rows": 340, "source_names": ["fixture_csv"]},
            "human_labels": {"row_count": 3, "unique_event_count": 3},
            "backtest": {"trade_count": 4, "kill_switch_value_usd": 54.42},
        }
    )

    assert payload["status"] == "not_research_ready"
    assert payload["readiness_gates"]["real_post_archive_verified"] is True
    assert payload["readiness_gates"]["licensed_market_data_loaded"] is False
    assert payload["readiness_gates"]["human_label_minimum_met"] is False
    assert payload["best_fixture_model"]["model"] == "TF-IDF logistic regression"


def test_interpretation_markdown_contains_model_and_gate_tables() -> None:
    payload = build_interpretation_report(
        {
            "classifier": {"test_rows": 3, "tfidf_logreg": _metric(0.5, 0.5)},
            "market": {"source_names": ["fixture_csv"]},
            "human_labels": {"row_count": 0},
        }
    )

    markdown = render_interpretation_markdown(payload)

    assert "| Model | Type | Status | Accuracy | Macro F1 |" in markdown
    assert "| licensed_market_data_loaded | False |" in markdown
    assert "FinBERT/DeBERTa fine-tuning is not methodologically ready." in markdown


def _metric(macro_f1: float, accuracy: float) -> dict:
    return {
        "macro_f1": macro_f1,
        "classification_report": {"accuracy": accuracy},
    }
