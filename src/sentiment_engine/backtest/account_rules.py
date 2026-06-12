from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sentiment_engine.config import load_yaml


@dataclass(frozen=True)
class AccountRules:
    name: str
    starting_balance_usd: float
    daily_loss_limit_usd: float
    maximum_loss_limit_usd: float
    trailing_drawdown_usd: float
    max_contracts: int
    flatten_on_daily_loss_breach: bool


def load_account_rules(path: str | Path) -> AccountRules:
    raw = load_yaml(path)["account"]
    return AccountRules(
        name=str(raw["name"]),
        starting_balance_usd=float(raw["starting_balance_usd"]),
        daily_loss_limit_usd=float(raw["daily_loss_limit_usd"]),
        maximum_loss_limit_usd=float(raw["maximum_loss_limit_usd"]),
        trailing_drawdown_usd=float(raw["trailing_drawdown_usd"]),
        max_contracts=int(raw["max_contracts"]),
        flatten_on_daily_loss_breach=bool(raw["flatten_on_daily_loss_breach"]),
    )


def account_rule_breaches(trades: pd.DataFrame, pnl_column: str, rules: AccountRules) -> dict[str, int]:
    if trades.empty:
        return {"daily_loss_breaches": 0, "maximum_loss_breaches": 0, "trailing_drawdown_breaches": 0}
    ordered = trades.sort_values("exit_ts_utc").copy()
    ordered["date"] = pd.to_datetime(ordered["exit_ts_utc"], utc=True).dt.date
    daily_pnl = ordered.groupby("date")[pnl_column].sum()
    cumulative = ordered[pnl_column].cumsum()
    equity = rules.starting_balance_usd + cumulative
    running_high = equity.cummax()
    return {
        "daily_loss_breaches": int((daily_pnl <= -rules.daily_loss_limit_usd).sum()),
        "maximum_loss_breaches": int((cumulative <= -rules.maximum_loss_limit_usd).sum()),
        "trailing_drawdown_breaches": int(((running_high - equity) >= rules.trailing_drawdown_usd).sum()),
    }
