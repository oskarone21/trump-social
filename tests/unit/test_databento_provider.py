from __future__ import annotations

import pandas as pd
import pytest

from sentiment_engine.ingestion.databento_provider import (
    DATABENTO_DATASET_DEFAULT,
    DATABENTO_SCHEMA_DEFAULT,
    DATABENTO_STYPE_IN_DEFAULT,
    download_databento_ohlcv,
    parse_symbol_list,
)
from sentiment_engine.ingestion.market_files import DATABENTO_OHLCV_SOURCE
from sentiment_engine.cli import main


class FakeStore:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.to_df_kwargs: dict[str, object] | None = None

    def to_df(self, **kwargs) -> pd.DataFrame:
        self.to_df_kwargs = kwargs
        return self.frame


class FakeTimeseries:
    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.request: dict[str, object] | None = None

    def get_range(self, **kwargs) -> FakeStore:
        self.request = kwargs
        return self.store


class FakeHistoricalClient:
    def __init__(self, store: FakeStore) -> None:
        self.timeseries = FakeTimeseries(store)


def test_download_databento_ohlcv_requests_and_normalises_bars() -> None:
    raw = pd.DataFrame(
        [
            {
                "open": 20_000.0,
                "high": 20_002.0,
                "low": 19_999.0,
                "close": 20_001.0,
                "volume": 10,
                "symbol": "NQH6",
            },
            {
                "open": 20_001.0,
                "high": 20_003.0,
                "low": 20_000.0,
                "close": 20_002.0,
                "volume": 12,
                "symbol": "NQH6",
            },
        ],
        index=pd.DatetimeIndex(
            ["2026-01-02T14:31:00Z", "2026-01-02T14:32:00Z"],
            name="ts_event",
        ),
    )
    store = FakeStore(raw)
    client = FakeHistoricalClient(store)

    bars, audit = download_databento_ohlcv(
        start="2026-01-02T14:30:00Z",
        end="2026-01-02T14:33:00Z",
        symbols=["NQ.c.0"],
        symbol_root="NQ",
        client=client,
    )

    assert client.timeseries.request == {
        "dataset": DATABENTO_DATASET_DEFAULT,
        "start": "2026-01-02T14:30:00Z",
        "end": "2026-01-02T14:33:00Z",
        "symbols": ["NQ.c.0"],
        "schema": DATABENTO_SCHEMA_DEFAULT,
        "stype_in": DATABENTO_STYPE_IN_DEFAULT,
        "stype_out": "instrument_id",
    }
    assert store.to_df_kwargs == {
        "price_type": "float",
        "pretty_ts": True,
        "map_symbols": True,
        "schema": DATABENTO_SCHEMA_DEFAULT,
    }
    assert len(bars) == 2
    assert bars.iloc[0]["ts_open_utc"].isoformat() == "2026-01-02T14:31:00+00:00"
    assert bars.iloc[0]["source_name"] == DATABENTO_OHLCV_SOURCE
    assert bars.iloc[0]["contract_symbol"] == "NQH6"
    assert audit["raw_row_count"] == 2
    assert audit["market_audit"]["valid_rows"] == 2
    assert "DATABENTO_API_KEY" not in str(audit)


def test_parse_symbol_list_splits_cli_value() -> None:
    assert parse_symbol_list("NQ.c.0, MNQ.c.0") == ["NQ.c.0", "MNQ.c.0"]


def test_download_databento_ohlcv_requires_symbols() -> None:
    with pytest.raises(ValueError, match="At least one Databento symbol"):
        download_databento_ohlcv(start="2026-01-02", end=None, symbols=[], symbol_root="NQ")


def test_download_databento_ohlcv_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set DATABENTO_API_KEY"):
        download_databento_ohlcv(
            start="2026-01-02",
            end=None,
            symbols=["NQ.c.0"],
            symbol_root="NQ",
        )


def test_download_databento_cli_exits_cleanly_without_api_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "download-databento-market",
                "--config",
                "configs/research.yaml",
                "--start",
                "2026-01-02",
                "--symbols",
                "NQ.c.0",
            ]
        )

    assert str(exc_info.value) == (
        "Set DATABENTO_API_KEY before downloading licensed Databento market data"
    )
    assert capsys.readouterr().err == ""
