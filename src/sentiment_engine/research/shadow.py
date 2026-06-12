from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.utils.io import write_json

SHADOW_REPORT_JSON = "shadow_report.json"
SHADOW_REPORT_MARKDOWN = "shadow_report.md"
FILTERED_STATUS = "filtered"
REDUCED_STATUS = "reduced"
UNCHANGED_STATUS = "unchanged"


def build_shadow_report(
    *,
    latest_signal: dict[str, Any],
    backtest_report: dict[str, Any],
    backtest_trades: pd.DataFrame,
    interpretation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    readiness = (interpretation_report or {}).get("readiness_gates", {})
    return {
        "status": _shadow_status(readiness),
        "latest_signal": _latest_signal_summary(latest_signal),
        "readiness_gates": readiness,
        "trade_summary": _trade_summary(backtest_report, backtest_trades),
        "action_attribution": _action_attribution(backtest_trades),
        "methodology_notes": [
            "Shadow report is an advisory review artefact; it is not live order routing.",
            "Dollar attribution comes from fixture backtest trades with configured costs and "
            "latency.",
            "False-positive cost is the model overlay cost on trades that would have made "
            "money before blocking or reduction.",
            "False-negative loss is losing exposure left unchanged by the overlay.",
            "Real shadow-mode promotion requires licensed market data, provider terms, and "
            "human-reviewed labels.",
        ],
    }


def write_shadow_report(
    *,
    report_dir: Path,
    latest_signal: dict[str, Any],
    backtest_report: dict[str, Any],
    backtest_trades: pd.DataFrame,
    interpretation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    report = build_shadow_report(
        latest_signal=latest_signal,
        backtest_report=backtest_report,
        backtest_trades=backtest_trades,
        interpretation_report=interpretation_report,
    )
    write_json(report_dir / SHADOW_REPORT_JSON, report)
    (report_dir / SHADOW_REPORT_MARKDOWN).write_text(
        render_shadow_markdown(report), encoding="utf-8"
    )
    return report


def render_shadow_markdown(report: dict[str, Any]) -> str:
    signal = report["latest_signal"]
    trade = report["trade_summary"]
    lines = [
        "# Truth Social NQ Shadow Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Latest Advisory Signal",
        "",
        f"- `post_id`: `{signal.get('post_id')}`",
        f"- `direction_signal`: `{signal.get('direction_signal')}`",
        f"- `kill_switch_action`: `{signal.get('kill_switch_action')}`",
        f"- `whipsaw_risk_level`: `{signal.get('whipsaw_risk_level')}`",
        f"- `human_reason`: `{signal.get('human_reason')}`",
        "",
        "## Shadow Trade Attribution",
        "",
        f"- `trade_count`: `{trade.get('trade_count')}`",
        f"- `filtered_trades`: `{trade.get('filtered_trades')}`",
        f"- `reduced_trades`: `{trade.get('reduced_trades')}`",
        f"- `unchanged_trades`: `{trade.get('unchanged_trades')}`",
        f"- `kill_switch_value_usd`: `{trade.get('kill_switch_value_usd')}`",
        f"- `false_positive_cost_usd`: `{trade.get('false_positive_cost_usd')}`",
        f"- `false_negative_loss_usd`: `{trade.get('false_negative_loss_usd')}`",
        "",
        "## Readiness Gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    for gate, passed in report["readiness_gates"].items():
        lines.append(f"| {gate} | {passed} |")
    lines.extend(["", "## Methodology Notes", ""])
    for note in report["methodology_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _shadow_status(readiness: dict[str, Any]) -> str:
    if readiness and all(bool(value) for value in readiness.values()):
        return "shadow_review_ready"
    return "fixture_shadow_only"


def _latest_signal_summary(signal: dict[str, Any]) -> dict[str, Any]:
    risk = signal.get("risk", {})
    kill_switch = signal.get("kill_switch", {})
    explanation = signal.get("explanation", {})
    return {
        "event_id": signal.get("event_id"),
        "post_id": signal.get("post_id"),
        "direction_signal": signal.get("direction_signal"),
        "tradeability_label": signal.get("tradeability_label"),
        "kill_switch_action": kill_switch.get("action"),
        "risk_multiplier": kill_switch.get("risk_multiplier"),
        "whipsaw_risk_level": risk.get("whipsaw_risk_level"),
        "whipsaw_score": risk.get("whipsaw_score"),
        "human_reason": explanation.get("human_readable_reason"),
    }


def _trade_summary(backtest_report: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    filtered = trades["kill_switch_status"].eq(FILTERED_STATUS)
    reduced = trades["kill_switch_status"].eq(REDUCED_STATUS)
    unchanged = trades["kill_switch_status"].eq(UNCHANGED_STATUS)
    overlay_cost = trades["kill_switch_value_usd"] < 0
    unchanged_loser = unchanged & (trades["net_pnl_before_usd"] < 0)
    return {
        "trade_count": int(len(trades)),
        "filtered_trades": int(filtered.sum()),
        "reduced_trades": int(reduced.sum()),
        "unchanged_trades": int(unchanged.sum()),
        "net_pnl_before_usd": backtest_report.get("net_pnl_before_usd"),
        "net_pnl_after_usd": backtest_report.get("net_pnl_after_usd"),
        "kill_switch_value_usd": backtest_report.get("kill_switch_value_usd"),
        "avoided_losing_trade_usd": backtest_report.get("avoided_losing_trade_usd"),
        "missed_winning_trade_usd": backtest_report.get("missed_winning_trade_usd"),
        "false_positive_cost_usd": round(
            float(trades.loc[overlay_cost, "kill_switch_value_usd"].abs().sum()), 4
        ),
        "false_negative_loss_usd": round(
            float(trades.loc[unchanged_loser, "net_pnl_before_usd"].abs().sum()), 4
        ),
    }


def _action_attribution(trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in trades.to_dict("records"):
        rows.append(
            {
                "trade_id": row["trade_id"],
                "entry_ts_utc": row["entry_ts_utc"],
                "kill_switch_status": row["kill_switch_status"],
                "blocking_event_id": row.get("blocking_event_id"),
                "blocking_risk_level": row.get("blocking_risk_level"),
                "net_pnl_before_usd": row["net_pnl_before_usd"],
                "net_pnl_after_usd": row["net_pnl_after_usd"],
                "kill_switch_value_usd": row["kill_switch_value_usd"],
                "attribution_label": _attribution_label(row),
            }
        )
    return rows


def _attribution_label(row: dict[str, Any]) -> str:
    status = row["kill_switch_status"]
    before = float(row["net_pnl_before_usd"])
    value = float(row["kill_switch_value_usd"])
    if status in {FILTERED_STATUS, REDUCED_STATUS} and value > 0:
        return "helpful_risk_reduction"
    if status in {FILTERED_STATUS, REDUCED_STATUS} and value < 0:
        return "false_positive_cost"
    if status == UNCHANGED_STATUS and before < 0:
        return "false_negative_loss"
    return "neutral_or_unaffected"
