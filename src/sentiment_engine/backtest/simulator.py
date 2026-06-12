from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.backtest.account_rules import account_rule_breaches, load_account_rules
from sentiment_engine.config import EngineConfig
from sentiment_engine.models.whipsaw import RISK_HARD, RISK_SOFT
from sentiment_engine.utils.io import write_dataframe, write_json
from sentiment_engine.utils.time import to_utc_series

BLOCK_ACTION = "BLOCK_NEW_ENTRIES"
REDUCE_ACTION = "REDUCE_SIZE"
FILTERED_STATUS = "filtered"
REDUCED_STATUS = "reduced"
UNCHANGED_STATUS = "unchanged"
LONG_SIDE = "long"
SHORT_SIDE = "short"


def run_kill_switch_backtest(
    trades_path: str | Path,
    scored_events: pd.DataFrame,
    config: EngineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trades = _load_trades(trades_path)
    trades["gross_pnl_before_usd"] = trades.apply(
        lambda row: _trade_pnl(row, row["contracts"], config, include_costs=False), axis=1
    )
    trades["net_pnl_before_usd"] = trades.apply(
        lambda row: _trade_pnl(row, row["contracts"], config, include_costs=True), axis=1
    )
    decisions = [
        _apply_kill_switch(row, scored_events, config)
        for row in trades.to_dict("records")
    ]
    decision_frame = pd.DataFrame(decisions)
    trades = pd.concat([trades.reset_index(drop=True), decision_frame], axis=1)
    trades["net_pnl_after_usd"] = trades.apply(
        lambda row: _trade_pnl(row, row["contracts_after"], config, include_costs=True), axis=1
    )
    trades["kill_switch_value_usd"] = trades["net_pnl_after_usd"] - trades["net_pnl_before_usd"]
    rules = load_account_rules(config.backtest["account_rules_config"])
    report = _report(trades, rules)
    write_dataframe(trades, config.paths.report_dir / "backtest_event_audit.csv")
    write_json(config.paths.report_dir / "backtest_report.json", report)
    return trades, report


def _load_trades(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["trade_id", "entry_ts_utc", "exit_ts_utc", "side", "contracts", "entry_price", "exit_price"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Trade CSV missing required columns: {missing}")
    frame["entry_ts_utc"] = to_utc_series(frame["entry_ts_utc"])
    frame["exit_ts_utc"] = to_utc_series(frame["exit_ts_utc"])
    frame["contracts"] = frame["contracts"].astype(int)
    frame["entry_price"] = frame["entry_price"].astype(float)
    frame["exit_price"] = frame["exit_price"].astype(float)
    if (frame["contracts"] <= 0).any():
        raise ValueError("Trade contracts must be positive")
    return frame.sort_values("entry_ts_utc").reset_index(drop=True)


def _trade_pnl(row: pd.Series | dict[str, Any], contracts: int, config: EngineConfig, *, include_costs: bool) -> float:
    if contracts <= 0:
        return 0.0
    side = row["side"]
    price_delta = float(row["exit_price"]) - float(row["entry_price"])
    if side == SHORT_SIDE:
        price_delta = -price_delta
    if side != LONG_SIDE and side != SHORT_SIDE:
        raise ValueError(f"Unsupported trade side: {side}")
    ticks = price_delta / config.instruments.tick_size
    gross = ticks * config.instruments.mnq_tick_value_usd * contracts
    if not include_costs:
        return round(gross, 4)
    round_trip_cost = (
        2 * float(config.backtest["commission_per_contract_usd"])
        + 2
        * float(config.backtest["slippage_ticks_per_side"])
        * config.instruments.mnq_tick_value_usd
    )
    return round(gross - round_trip_cost * contracts, 4)


def _apply_kill_switch(
    trade: dict[str, Any], scored_events: pd.DataFrame, config: EngineConfig
) -> dict[str, Any]:
    entry_ts = pd.Timestamp(trade["entry_ts_utc"])
    active = _active_events(entry_ts, scored_events, config)
    if active.empty:
        return {
            "kill_switch_status": UNCHANGED_STATUS,
            "contracts_after": int(trade["contracts"]),
            "blocking_event_id": None,
            "blocking_risk_level": None,
            "risk_multiplier": 1.0,
        }
    hard = active[active["whipsaw_risk_level"].eq(RISK_HARD)]
    if not hard.empty:
        event = hard.sort_values("whipsaw_score", ascending=False).iloc[0]
        return _decision(FILTERED_STATUS, 0, event, 0.0)
    soft = active[active["whipsaw_risk_level"].eq(RISK_SOFT)]
    if not soft.empty:
        event = soft.sort_values("whipsaw_score", ascending=False).iloc[0]
        if config.live_actions["soft_risk_action"] == BLOCK_ACTION:
            return _decision(FILTERED_STATUS, 0, event, 0.0)
        if config.live_actions["soft_risk_action"] == REDUCE_ACTION:
            contracts_after = int(int(trade["contracts"]) * 0.5)
            return _decision(
                REDUCED_STATUS if contracts_after > 0 else FILTERED_STATUS,
                contracts_after,
                event,
                0.5,
            )
    return {
        "kill_switch_status": UNCHANGED_STATUS,
        "contracts_after": int(trade["contracts"]),
        "blocking_event_id": None,
        "blocking_risk_level": None,
        "risk_multiplier": 1.0,
    }


def _active_events(
    entry_ts: pd.Timestamp, scored_events: pd.DataFrame, config: EngineConfig
) -> pd.DataFrame:
    scored = scored_events.copy()
    scored["signal_effective_ts"] = pd.to_datetime(scored["received_at_utc"], utc=True) + pd.Timedelta(
        seconds=int(config.backtest["latency_seconds"])
    )
    scored["signal_expires_ts"] = scored["signal_effective_ts"] + pd.to_timedelta(
        scored["risk_ttl_seconds"].astype(int), unit="s"
    )
    return scored[(scored["signal_effective_ts"] <= entry_ts) & (scored["signal_expires_ts"] >= entry_ts)]


def _decision(status: str, contracts_after: int, event: pd.Series, risk_multiplier: float) -> dict[str, Any]:
    return {
        "kill_switch_status": status,
        "contracts_after": contracts_after,
        "blocking_event_id": str(event["event_id"]),
        "blocking_risk_level": str(event["whipsaw_risk_level"]),
        "risk_multiplier": risk_multiplier,
    }


def _report(trades: pd.DataFrame, rules) -> dict[str, Any]:
    filtered = trades["kill_switch_status"].eq(FILTERED_STATUS)
    reduced = trades["kill_switch_status"].eq(REDUCED_STATUS)
    before = float(trades["net_pnl_before_usd"].sum())
    after = float(trades["net_pnl_after_usd"].sum())
    avoided_losses = trades[filtered & (trades["net_pnl_before_usd"] < 0)]["net_pnl_before_usd"].abs().sum()
    missed_winners = trades[filtered & (trades["net_pnl_before_usd"] > 0)]["net_pnl_before_usd"].sum()
    return {
        "trade_count": int(len(trades)),
        "filtered_trades": int(filtered.sum()),
        "reduced_trades": int(reduced.sum()),
        "net_pnl_before_usd": round(before, 4),
        "net_pnl_after_usd": round(after, 4),
        "kill_switch_value_usd": round(after - before, 4),
        "avoided_losing_trade_usd": round(float(avoided_losses), 4),
        "missed_winning_trade_usd": round(float(missed_winners), 4),
        "account_breaches_before": account_rule_breaches(trades, "net_pnl_before_usd", rules),
        "account_breaches_after": account_rule_breaches(trades, "net_pnl_after_usd", rules),
        "methodology_notes": [
            "Costs include configured commission and round-trip slippage.",
            "Signal availability is shifted by configured post-to-fill latency.",
            "Fixture backtest is an accounting and integration check, not evidence of edge.",
        ],
    }
