from __future__ import annotations

import pandas as pd

from sentiment_engine.research.shadow import build_shadow_report, render_shadow_markdown


def test_shadow_report_attributes_false_positive_and_false_negative_dollars() -> None:
    report = build_shadow_report(
        latest_signal={
            "event_id": "event-1",
            "post_id": "post-1",
            "direction_signal": "BULLISH",
            "tradeability_label": "tradeable_directional",
            "risk": {"whipsaw_risk_level": "WATCH", "whipsaw_score": 0.3},
            "kill_switch": {"action": "ALLOW", "risk_multiplier": 1.0},
            "explanation": {"human_readable_reason": "ALLOW: test"},
        },
        backtest_report={
            "net_pnl_before_usd": -10.0,
            "net_pnl_after_usd": -2.0,
            "kill_switch_value_usd": 8.0,
            "avoided_losing_trade_usd": 12.0,
            "missed_winning_trade_usd": 4.0,
        },
        backtest_trades=pd.DataFrame(
            [
                {
                    "trade_id": "winner-blocked",
                    "entry_ts_utc": "2026-01-01T00:00:00Z",
                    "kill_switch_status": "filtered",
                    "blocking_event_id": "event-1",
                    "blocking_risk_level": "SOFT_RISK",
                    "net_pnl_before_usd": 4.0,
                    "net_pnl_after_usd": 0.0,
                    "kill_switch_value_usd": -4.0,
                },
                {
                    "trade_id": "loser-unchanged",
                    "entry_ts_utc": "2026-01-01T00:05:00Z",
                    "kill_switch_status": "unchanged",
                    "blocking_event_id": None,
                    "blocking_risk_level": None,
                    "net_pnl_before_usd": -6.0,
                    "net_pnl_after_usd": -6.0,
                    "kill_switch_value_usd": 0.0,
                },
            ]
        ),
        interpretation_report={
            "readiness_gates": {
                "licensed_market_data_loaded": False,
                "human_label_minimum_met": False,
            }
        },
    )

    assert report["status"] == "fixture_shadow_only"
    assert report["latest_signal"]["kill_switch_action"] == "ALLOW"
    assert report["trade_summary"]["false_positive_cost_usd"] == 4.0
    assert report["trade_summary"]["false_negative_loss_usd"] == 6.0
    assert report["action_attribution"][0]["attribution_label"] == "false_positive_cost"
    assert report["action_attribution"][1]["attribution_label"] == "false_negative_loss"
    markdown = render_shadow_markdown(report)
    assert "false_positive_cost_usd" in markdown
    assert "false_negative_loss_usd" in markdown
