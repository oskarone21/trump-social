from __future__ import annotations

import pandas as pd
import pytest

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import _validate_rows, audit_market_bars
from sentiment_engine.ingestion.market_files import DATABENTO_OHLCV_SOURCE, load_market_file
from sentiment_engine.ingestion.posts_fixture import load_fixture_posts, posts_to_frame
from sentiment_engine.research.archive_events import build_archive_event_dataset


def test_databento_style_ohlcv_export_normalises_to_market_bars(tmp_path) -> None:
    source = tmp_path / "nq_ohlcv.csv"
    pd.DataFrame(
        [
            {
                "ts_event": "2026-01-02T14:31:00Z",
                "symbol": "NQH6",
                "open": 20_000_000_000_000,
                "high": 20_002_000_000_000,
                "low": 19_999_000_000_000,
                "close": 20_001_000_000_000,
                "volume": 10,
            },
            {
                "ts_event": "2026-01-02T14:32:00Z",
                "symbol": "NQH6",
                "open": 20_001_000_000_000,
                "high": 20_003_000_000_000,
                "low": 20_000_000_000_000,
                "close": 20_002_000_000_000,
                "volume": 12,
            },
        ]
    ).to_csv(source, index=False)

    bars = load_market_file(source, source_name=DATABENTO_OHLCV_SOURCE, symbol_root="NQ")
    audit = audit_market_bars(bars)

    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 20000.0
    assert bars.iloc[0]["contract_symbol"] == "NQH6"
    assert audit["valid_rows"] == 2
    assert audit["expected_minute_count"] == 2
    assert audit["missing_bar_count"] == 0
    assert audit["stale_bar_rows"] == 0
    assert audit["source_names"] == [DATABENTO_OHLCV_SOURCE]


def test_capitalized_ohlcv_export_normalises_to_market_bars(tmp_path) -> None:
    source = tmp_path / "nq_ohlcv_capitalized.csv"
    pd.DataFrame(
        [
            {
                "ts_event": "2026-01-02T14:31:00Z",
                "Open": 20000.0,
                "High": 20002.0,
                "Low": 19999.0,
                "Close": 20001.0,
                "Volume": 10,
            },
            {
                "ts_event": "2026-01-02T14:32:00Z",
                "Open": 20001.0,
                "High": 20003.0,
                "Low": 20000.0,
                "Close": 20002.0,
                "Volume": 12,
            },
        ]
    ).to_csv(source, index=False)

    bars = load_market_file(source, source_name="local_nq_1min_clean", symbol_root="NQ")
    audit = audit_market_bars(bars)

    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 20000.0
    assert bars.iloc[0]["volume"] == 10
    assert bars.iloc[0]["contract_symbol"] == "NQ"
    assert audit["valid_rows"] == 2
    assert audit["source_names"] == ["local_nq_1min_clean"]


def test_price_normalisation_handles_mixed_raw_and_scaled_values(tmp_path) -> None:
    source = tmp_path / "nq_ohlcv_mixed_prices.csv"
    pd.DataFrame(
        [
            {
                "ts_event": "2026-01-02T14:31:00Z",
                "open": 20_000_000_000_000,
                "high": 20_002_000_000_000,
                "low": 19_999_000_000_000,
                "close": 20_001_000_000_000,
                "vwap": 20_000_500_000_000,
                "volume": 10,
            },
            {
                "ts_event": "2026-01-02T14:32:00Z",
                "open": 20001.0,
                "high": 20003.0,
                "low": 20000.0,
                "close": 20002.0,
                "vwap": 20001.5,
                "volume": 12,
            },
        ]
    ).to_csv(source, index=False)

    bars = load_market_file(source, source_name=DATABENTO_OHLCV_SOURCE, symbol_root="NQ")

    assert bars["open"].tolist() == [20000.0, 20001.0]
    assert bars["vwap"].tolist() == [20000.5, 20001.5]


def test_market_audit_counts_missing_and_stale_bars() -> None:
    frame = pd.DataFrame(
        [
            _market_row("2026-01-02T14:30:00Z", open_price=20000.0, volume=10),
            _market_row("2026-01-02T14:32:00Z", open_price=20001.0, volume=0),
            _market_row("2026-01-02T14:33:00Z", open_price=20001.0, volume=0),
        ]
    )

    audit = audit_market_bars(frame)

    assert audit["expected_minute_count"] == 4
    assert audit["valid_rows"] == 3
    assert audit["missing_bar_count"] == 1
    assert audit["gap_count_gt_1m"] == 1
    assert audit["stale_bar_rows"] == 1
    assert audit["zero_volume_rows"] == 2


def test_market_validation_rejects_invalid_symbol_root() -> None:
    frame = pd.DataFrame([_market_row("2026-01-02T14:30:00Z", open_price=20000.0, volume=10)])
    frame["symbol_root"] = "ES"

    with pytest.raises(ValueError, match="invalid symbol_root"):
        _validate_rows(frame)


def test_market_validation_rejects_invalid_valid_bar_ohlc() -> None:
    frame = pd.DataFrame([_market_row("2026-01-02T14:30:00Z", open_price=20000.0, volume=10)])
    frame["high"] = frame["open"] - 1.0

    with pytest.raises(ValueError, match="invalid valid-bar OHLC"):
        _validate_rows(frame)


def test_archive_events_build_from_processed_posts_and_market_bars(tmp_path) -> None:
    config = load_config("configs/research.yaml")
    posts_path = tmp_path / "posts.parquet"
    market_path = tmp_path / "market.parquet"
    events_path = tmp_path / "real_events.parquet"
    posts = posts_to_frame(load_fixture_posts(config.paths.posts_fixture))
    posts.to_parquet(posts_path, index=False)
    load_market_file(
        config.paths.market_fixture,
        source_name="fixture_canonical",
        symbol_root="NQ",
    ).to_parquet(market_path, index=False)

    result = build_archive_event_dataset(
        posts_path=posts_path,
        market_path=market_path,
        config=config,
        limit_posts=8,
    )
    result.events.to_parquet(events_path, index=False)

    assert len(result.events) == 8
    assert result.audit["event_count"] == 8
    assert result.audit["skipped_post_count"] == 0
    assert events_path.exists()


def _market_row(timestamp: str, *, open_price: float, volume: int) -> dict[str, object]:
    return {
        "symbol_root": "NQ",
        "contract_symbol": "NQH6",
        "continuous_symbol": "NQ.c.0",
        "ts_open_utc": pd.Timestamp(timestamp),
        "ts_close_utc": pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
        "open": open_price,
        "high": open_price + 1.0,
        "low": open_price - 1.0,
        "close": open_price,
        "volume": volume,
        "trade_count": volume,
        "vwap": open_price,
        "source_name": "fixture",
        "is_rth": True,
        "session_id": "2026-01-02",
        "is_rollover_period": False,
        "is_holiday_session": False,
        "is_valid_bar": True,
    }
