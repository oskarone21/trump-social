from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

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
OHLC_COLUMNS = ["open", "high", "low", "close"]
MARKET_AUDIT_GROUP_COLUMNS = ["contract_symbol", "session_id"]


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
    coverage = _coverage_audit(valid)
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
        "min_ts_open_utc": (
            isoformat_z(valid["ts_open_utc"].min().to_pydatetime()) if len(valid) else None
        ),
        "max_ts_open_utc": (
            isoformat_z(valid["ts_open_utc"].max().to_pydatetime()) if len(valid) else None
        ),
        "expected_minute_count": coverage["expected_minute_count"],
        "missing_bar_count": coverage["missing_bar_count"],
        "gap_count_gt_1m": coverage["gap_count_gt_1m"],
        "duplicate_bar_keys": int(duplicate_bar_keys),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "stale_bar_rows": coverage["stale_bar_rows"],
        "zero_volume_rows": int((frame["volume"] == 0).sum()),
        "symbols": sorted(frame["symbol_root"].dropna().unique().tolist()),
        "contract_symbols": sorted(frame["contract_symbol"].dropna().unique().tolist()),
        "source_names": sorted(frame["source_name"].dropna().unique().tolist()),
    }


def _coverage_audit(valid: pd.DataFrame) -> dict[str, int]:
    if valid.empty:
        return {
            "expected_minute_count": 0,
            "missing_bar_count": 0,
            "gap_count_gt_1m": 0,
            "stale_bar_rows": 0,
        }
    unique_bars = valid.drop_duplicates(["ts_open_utc", "contract_symbol"]).sort_values(
        MARKET_AUDIT_GROUP_COLUMNS + ["ts_open_utc"]
    )
    expected = 0
    gaps = 0
    stale_rows = 0
    for _group_key, group in unique_bars.groupby(MARKET_AUDIT_GROUP_COLUMNS):
        timestamps = group["ts_open_utc"]
        expected += int((timestamps.max() - timestamps.min()) / pd.Timedelta(minutes=1)) + 1
        gaps += int((timestamps.diff().dropna() > pd.Timedelta(minutes=1)).sum())
        stale_rows += _stale_bar_count(group)
    return {
        "expected_minute_count": int(expected),
        "missing_bar_count": int(expected - len(unique_bars)),
        "gap_count_gt_1m": int(gaps),
        "stale_bar_rows": int(stale_rows),
    }


def _stale_bar_count(group: pd.DataFrame) -> int:
    stale_mask = (
        group[OHLC_COLUMNS].eq(group[OHLC_COLUMNS].shift()).all(axis=1)
        & group["volume"].eq(0)
        & group["trade_count"].eq(0)
    )
    return int(stale_mask.sum())


def _validate_rows(frame: pd.DataFrame) -> None:
    missing = [column for column in MARKET_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Market bars missing required columns: {missing}")
    invalid_symbols = sorted(
        set(frame["symbol_root"].dropna().astype(str).unique()).difference({"NQ", "MNQ"})
    )
    if invalid_symbols:
        raise ValueError(f"Market bars contain invalid symbol_root values: {invalid_symbols}")
    missing_text_columns = [
        column
        for column in ("contract_symbol", "continuous_symbol", "source_name", "session_id")
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any()
    ]
    if missing_text_columns:
        raise ValueError(f"Market bars contain blank identifier columns: {missing_text_columns}")
    invalid_close_times = frame["ts_close_utc"].le(frame["ts_open_utc"])
    if invalid_close_times.any():
        raise ValueError(f"Market bars contain {int(invalid_close_times.sum())} invalid close times")
    valid = frame["is_valid_bar"].astype(bool)
    valid_rows = frame[valid]
    invalid_ohlc = (
        valid_rows["high"].lt(valid_rows[["open", "close"]].max(axis=1))
        | valid_rows["low"].gt(valid_rows[["open", "close"]].min(axis=1))
        | valid_rows["volume"].lt(0)
    )
    if invalid_ohlc.any():
        raise ValueError(f"Market bars contain {int(invalid_ohlc.sum())} invalid valid-bar OHLC rows")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")
