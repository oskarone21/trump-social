from __future__ import annotations

import pandas as pd

from sentiment_engine.live.dashboard import build_dashboard
from sentiment_engine.schemas import SignalRecord


def test_dashboard_renders_model_comparison_when_interpretation_is_available(tmp_path) -> None:
    output = build_dashboard(
        signal=_signal(),
        whipsaw_report={
            "soft_risk": {"precision": 0.5, "recall": 0.4},
            "hard_kill": {"precision": 1.0, "recall": 0.2},
        },
        backtest_report={
            "kill_switch_value_usd": 10.0,
            "filtered_trades": 1,
            "reduced_trades": 0,
        },
        interpretation_report={
            "model_comparison": [
                {
                    "model": "TF-IDF logistic regression",
                    "model_type": "ml",
                    "status": "trained",
                    "accuracy": 0.5,
                    "macro_f1": 0.5,
                    "negative_log_loss": 1.2,
                    "ece": 0.25,
                }
            ],
            "finbert_summary": {
                "status": "scored",
                "model_name": "ProsusAI/finbert",
                "mode": "inference_only",
                "scored_rows": 2,
                "label_counts": {"positive": 2},
                "mean_scores": {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
            },
            "readiness_gates": {"licensed_market_data_loaded": False},
        },
        scored_events=pd.DataFrame(
            [
                {
                    "post_id": "fixture-001",
                    "whipsaw_risk_level": "WATCH",
                    "whipsaw_score": 0.25,
                    "rule_topic_labels": ["trade_policy"],
                    "text_clean": "test post",
                }
            ]
        ),
        output_path=tmp_path / "dashboard.html",
    )

    html = output.read_text()
    assert "Model Comparison" in html
    assert "TF-IDF logistic regression" in html
    assert "Log Loss" in html
    assert "ECE" in html
    assert "FinBERT Inference" in html
    assert "ProsusAI/finbert" in html
    assert "licensed_market_data_loaded" in html


def test_dashboard_renders_provider_health_section_if_report_present(tmp_path) -> None:
    output = build_dashboard(
        signal=_signal(),
        whipsaw_report={
            "soft_risk": {"precision": 0.5, "recall": 0.4},
            "hard_kill": {"precision": 1.0, "recall": 0.2},
        },
        backtest_report={
            "kill_switch_value_usd": 10.0,
            "filtered_trades": 1,
            "reduced_trades": 0,
        },
        interpretation_report=None,
        provider_freshness_report={
            "is_http_ok": False,
            "remote": {"method": "local_source"},
            "source_name": "provider_dump",
            "local_provider": {
                "schema_drift_detected": False,
                "required_columns_present": True,
            },
            "is_stale_by_post_time": False,
        },
        scored_events=pd.DataFrame(
            [
                {
                    "post_id": "fixture-002",
                    "whipsaw_risk_level": "NONE",
                    "whipsaw_score": 0.10,
                    "rule_topic_labels": ["trade_policy"],
                    "text_clean": "test post two",
                }
            ]
        ),
        output_path=tmp_path / "dashboard_with_provider.html",
    )

    html = output.read_text()
    assert "Provider Health" in html
    assert "provider_dump" in html


def _signal() -> SignalRecord:
    return SignalRecord.model_validate(
        {
            "event_id": "event-1",
            "post_id": "post-1",
            "source_provider": "fixture",
            "created_at_utc": "2026-01-01T00:00:00Z",
            "received_at_utc": "2026-01-01T00:00:01Z",
            "generated_at_utc": "2026-01-01T00:00:02Z",
            "text_clean": "test post",
            "sentiment_label": "bullish_market",
            "sentiment_confidence": 0.8,
            "topic_labels": ["trade_policy"],
            "topic_confidence": {"trade_policy": 0.8},
            "tradeability_label": "tradeable_directional",
            "direction_signal": "BULLISH",
            "p_direction": {"up_5m": 0.5, "down_5m": 0.2, "flat_5m": 0.3},
            "expected_delta_ticks": {"5m": 4.0},
            "risk": {"whipsaw_risk_level": "WATCH", "whipsaw_score": 0.25},
            "kill_switch": {"action": "ALLOW"},
            "data_quality": {"features_complete": True},
            "model_versions": {"rules": "test"},
            "explanation": {"human_readable_reason": "test"},
        }
    )
