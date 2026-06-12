from __future__ import annotations

import pandas as pd

from sentiment_engine.config import load_config
from sentiment_engine.ingestion.market_csv import audit_market_bars
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
    assert audit["source_names"] == [DATABENTO_OHLCV_SOURCE]


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
