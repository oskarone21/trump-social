from __future__ import annotations

import os
from typing import Any

import pandas as pd

from sentiment_engine.ingestion.market_csv import audit_market_bars
from sentiment_engine.ingestion.market_files import DATABENTO_OHLCV_SOURCE, normalise_market_frame

DATABENTO_API_KEY_ENV = "DATABENTO_API_KEY"
DATABENTO_DATASET_DEFAULT = "GLBX.MDP3"
DATABENTO_SCHEMA_DEFAULT = "ohlcv-1m"
DATABENTO_STYPE_IN_DEFAULT = "continuous"
DATABENTO_STYPE_OUT_DEFAULT = "instrument_id"


def download_databento_ohlcv(
    *,
    start: str,
    end: str | None,
    symbols: list[str],
    symbol_root: str,
    dataset: str = DATABENTO_DATASET_DEFAULT,
    schema: str = DATABENTO_SCHEMA_DEFAULT,
    stype_in: str = DATABENTO_STYPE_IN_DEFAULT,
    stype_out: str = DATABENTO_STYPE_OUT_DEFAULT,
    limit: int | None = None,
    contract_symbol: str | None = None,
    continuous_symbol: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not symbols:
        raise ValueError("At least one Databento symbol is required")

    historical = client if client is not None else _build_historical_client(api_key)
    request = {
        "dataset": dataset,
        "start": start,
        "end": end,
        "symbols": symbols,
        "schema": schema,
        "stype_in": stype_in,
        "stype_out": stype_out,
        "limit": limit,
    }
    store = historical.timeseries.get_range(
        **{key: value for key, value in request.items() if value is not None}
    )
    raw = store.to_df(price_type="float", pretty_ts=True, map_symbols=True, schema=schema)
    bars = normalise_market_frame(
        raw,
        source_name=DATABENTO_OHLCV_SOURCE,
        symbol_root=symbol_root,
        contract_symbol=contract_symbol,
        continuous_symbol=continuous_symbol or _continuous_symbol(symbols),
    )
    audit = {
        "provider": "databento",
        "request": request,
        "raw_row_count": int(len(raw)),
        "market_audit": audit_market_bars(bars),
        "license_note": (
            "Databento market data is licensed data. Do not commit downloaded bars "
            "or redistribute them outside the account's permitted use."
        ),
    }
    return bars, audit


def parse_symbol_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_historical_client(api_key: str | None) -> Any:
    token = api_key or os.environ.get(DATABENTO_API_KEY_ENV)
    if not token:
        raise RuntimeError(
            f"Set {DATABENTO_API_KEY_ENV} before downloading licensed Databento market data"
        )
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            'Install the optional market dependency: pip install -e ".[market]"'
        ) from exc
    return db.Historical(token)


def _continuous_symbol(symbols: list[str]) -> str | None:
    if len(symbols) == 1:
        return symbols[0]
    return None
