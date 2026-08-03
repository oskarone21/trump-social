from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_engine.ingestion.market_csv import MARKET_REQUIRED_COLUMNS, _to_bool, _validate_rows
from sentiment_engine.utils.time import to_utc_series

SUPPORTED_MARKET_SUFFIXES = {".csv", ".parquet"}
DATABENTO_OHLCV_SOURCE = "databento_glbx_mdp3_ohlcv_1m"
TIMESTAMP_ALIASES = ["ts_open_utc", "ts_event", "timestamp", "datetime", "time"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
BOOLEAN_COLUMNS = ["is_rth", "is_rollover_period", "is_holiday_session", "is_valid_bar"]
NANO_PRICE_THRESHOLD = 1_000_000_000
OHLCV_COLUMN_ALIASES = {
    "open": ["Open"],
    "high": ["High"],
    "low": ["Low"],
    "close": ["Close"],
    "volume": ["Volume"],
}


def load_market_file(
    path: str | Path,
    *,
    source_name: str,
    symbol_root: str,
    contract_symbol: str | None = None,
    continuous_symbol: str | None = None,
) -> pd.DataFrame:
    raw = _read_market_file(path)
    return normalise_market_frame(
        raw,
        source_name=source_name,
        symbol_root=symbol_root,
        contract_symbol=contract_symbol,
        continuous_symbol=continuous_symbol,
    )


def normalise_market_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
    symbol_root: str,
    contract_symbol: str | None = None,
    continuous_symbol: str | None = None,
) -> pd.DataFrame:
    raw = _normalise_column_aliases(_with_timestamp_column(frame))
    if _is_canonical(raw):
        canonical = _canonical_frame(raw)
    else:
        canonical = _normalise_ohlcv(
            raw,
            source_name=source_name,
            symbol_root=symbol_root,
            contract_symbol=contract_symbol,
            continuous_symbol=continuous_symbol,
        )
    _validate_rows(canonical)
    return canonical.sort_values(["ts_open_utc", "contract_symbol"]).drop_duplicates(
        ["ts_open_utc", "contract_symbol"], keep="last"
    )


def _read_market_file(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(
        f"Unsupported market file format {suffix}; expected {SUPPORTED_MARKET_SUFFIXES}"
    )


def _is_canonical(frame: pd.DataFrame) -> bool:
    return all(column in frame.columns for column in MARKET_REQUIRED_COLUMNS)


def _with_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    if _first_present(frame, TIMESTAMP_ALIASES) is not None:
        return frame
    if frame.index.name in TIMESTAMP_ALIASES:
        return frame.reset_index()
    return frame


def _normalise_column_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    casefolded_columns = {str(column).casefold(): column for column in frame.columns}
    for canonical, aliases in OHLCV_COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            source_column = casefolded_columns.get(alias.casefold())
            if source_column is not None:
                rename_map[str(source_column)] = canonical
                break
    if not rename_map:
        return frame
    return frame.rename(columns=rename_map)


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame[MARKET_REQUIRED_COLUMNS].copy()
    canonical["ts_open_utc"] = to_utc_series(canonical["ts_open_utc"])
    canonical["ts_close_utc"] = to_utc_series(canonical["ts_close_utc"])
    for column in BOOLEAN_COLUMNS:
        canonical[column] = canonical[column].map(_to_bool)
    for column in PRICE_COLUMNS:
        canonical[column] = canonical[column].astype(float)
    canonical["volume"] = canonical["volume"].astype(int)
    canonical["trade_count"] = canonical["trade_count"].fillna(0).astype(int)
    canonical["vwap"] = canonical["vwap"].astype(float)
    return canonical


def _normalise_ohlcv(
    frame: pd.DataFrame,
    *,
    source_name: str,
    symbol_root: str,
    contract_symbol: str | None,
    continuous_symbol: str | None,
) -> pd.DataFrame:
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Market file missing OHLC columns: {missing}")
    timestamp_column = _first_present(frame, TIMESTAMP_ALIASES)
    if timestamp_column is None:
        raise ValueError(f"Market file missing timestamp column; accepted {TIMESTAMP_ALIASES}")

    normalized = pd.DataFrame()
    normalized["ts_open_utc"] = to_utc_series(frame[timestamp_column])
    normalized["ts_close_utc"] = normalized["ts_open_utc"] + pd.Timedelta(minutes=1)
    for column in PRICE_COLUMNS:
        normalized[column] = _normalise_price_series(frame[column])
    normalized["volume"] = frame.get("volume", 0)
    normalized["volume"] = normalized["volume"].fillna(0).astype(int)
    normalized["trade_count"] = frame.get("trade_count", frame.get("count", 0))
    normalized["trade_count"] = normalized["trade_count"].fillna(0).astype(int)
    normalized["vwap"] = (
        _normalise_price_series(frame["vwap"]) if "vwap" in frame else _typical_price(normalized)
    )
    normalized["symbol_root"] = symbol_root
    normalized["contract_symbol"] = _contract_symbol(frame, contract_symbol, symbol_root)
    normalized["continuous_symbol"] = continuous_symbol or normalized["contract_symbol"]
    normalized["source_name"] = source_name
    normalized["is_rth"] = frame["is_rth"].map(_to_bool) if "is_rth" in frame else False
    normalized["session_id"] = normalized["ts_open_utc"].dt.strftime("%Y-%m-%d")
    normalized["is_rollover_period"] = (
        frame["is_rollover_period"].map(_to_bool) if "is_rollover_period" in frame else False
    )
    normalized["is_holiday_session"] = (
        frame["is_holiday_session"].map(_to_bool) if "is_holiday_session" in frame else False
    )
    normalized["is_valid_bar"] = _valid_bar_mask(normalized)
    return normalized[MARKET_REQUIRED_COLUMNS]


def _first_present(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _normalise_price(value: object) -> float:
    price = float(value)
    if abs(price) > NANO_PRICE_THRESHOLD:
        return price / NANO_PRICE_THRESHOLD
    return price


def _normalise_price_series(series: pd.Series) -> pd.Series:
    prices = series.astype(float)
    scaled = prices.abs().gt(NANO_PRICE_THRESHOLD)
    if scaled.any():
        prices = prices.copy()
        prices.loc[scaled] = prices.loc[scaled] / NANO_PRICE_THRESHOLD
    return prices


def _typical_price(frame: pd.DataFrame) -> pd.Series:
    return (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0


def _contract_symbol(frame: pd.DataFrame, requested: str | None, symbol_root: str) -> str:
    if requested:
        return requested
    if "symbol" in frame:
        symbols = frame["symbol"].dropna().astype(str).unique().tolist()
        if len(symbols) == 1:
            return symbols[0]
    return symbol_root


def _valid_bar_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[PRICE_COLUMNS].notna().all(axis=1)
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
        & (frame["volume"] >= 0)
    )
