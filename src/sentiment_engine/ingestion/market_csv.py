from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.schemas import MarketBar
from sentiment_engine.utils.time import isoformat_z, to_utc_series

MARKET_REQUIRED_COLUMNS = [
    "symbol_root",
    "contract_symbol",
    "continuous_symbol",
    "ts_open_utc",
    "ts_close_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "source_name",
    "is_rth",
    "session_id",
    "is_rollover_period",
    "is_holiday_session",
    "is_valid_bar",
]

BOOLEAN_COLUMNS = ["is_rth", "is_rollover_period", "is_holiday_session", "is_valid_bar"]


def load_market_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in MARKET_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Market CSV missing required columns: {missing}")
    frame = frame[MARKET_REQUIRED_COLUMNS].copy()
    frame["ts_open_utc"] = to_utc_series(frame["ts_open_utc"])
    frame["ts_close_utc"] = to_utc_series(frame["ts_close_utc"])
    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(_to_bool)
    frame["volume"] = frame["volume"].astype(int)
    frame["trade_count"] = frame["trade_count"].astype(int)
    for column in ["open", "high", "low", "close", "vwap"]:
        frame[column] = frame[column].astype(float)
    _validate_rows(frame)
    return frame.sort_values(["ts_open_utc", "contract_symbol"]).drop_duplicates(
        ["ts_open_utc", "contract_symbol"], keep="last"
    )


def audit_market_bars(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[frame["is_valid_bar"]]
    gaps = valid["ts_open_utc"].diff().dropna()
    gap_count = int((gaps > pd.Timedelta(minutes=1)).sum())
    duplicate_bar_keys = frame.duplicated(["ts_open_utc", "contract_symbol"]).sum()
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["volume"] < 0)
    )
    return {
        "row_count": int(len(frame)),
        "valid_rows": int(len(valid)),
        "invalid_rows": int((~frame["is_valid_bar"]).sum()),
        "min_ts_open_utc": isoformat_z(valid["ts_open_utc"].min().to_pydatetime()) if len(valid) else None,
        "max_ts_open_utc": isoformat_z(valid["ts_open_utc"].max().to_pydatetime()) if len(valid) else None,
        "gap_count_gt_1m": gap_count,
        "duplicate_bar_keys": int(duplicate_bar_keys),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "zero_volume_rows": int((frame["volume"] == 0).sum()),
        "symbols": sorted(frame["symbol_root"].dropna().unique().tolist()),
        "contract_symbols": sorted(frame["contract_symbol"].dropna().unique().tolist()),
        "source_names": sorted(frame["source_name"].dropna().unique().tolist()),
    }


def _validate_rows(frame: pd.DataFrame) -> None:
    for row in frame.to_dict("records"):
        MarketBar.model_validate(row)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")
