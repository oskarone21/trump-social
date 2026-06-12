from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "data" / "fixtures"
POSTS_PATH = FIXTURE_DIR / "posts_stiles_sample.json"
MARKET_PATH = FIXTURE_DIR / "nq_1m_sample.csv"
CALENDAR_PATH = FIXTURE_DIR / "macro_calendar.csv"
TRADES_PATH = FIXTURE_DIR / "baseline_trades.csv"


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_posts()
    _write_market()
    _write_calendar()
    _write_trades()
    print(f"fixtures written to {FIXTURE_DIR}")


def _write_posts() -> None:
    posts = [
        {
            "id": "fixture-001",
            "created_at": "2026-01-02T14:30:15Z",
            "received_at": "2026-01-02T14:30:17Z",
            "content": "CHINA TARIFFS are coming back. America will win on trade!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-001",
            "media": [],
            "replies_count": 400,
            "reblogs_count": 1200,
            "favourites_count": 6500,
        },
        {
            "id": "fixture-002",
            "created_at": "2026-01-02T14:42:10Z",
            "received_at": "2026-01-02T14:42:13Z",
            "content": "Had a GREAT call with China. A beautiful trade deal is close!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-002",
            "media": [],
            "replies_count": 320,
            "reblogs_count": 1000,
            "favourites_count": 6100,
        },
        {
            "id": "fixture-003",
            "created_at": "2026-01-02T15:10:05Z",
            "received_at": "2026-01-02T15:10:07Z",
            "content": "The Fed should cut rates now. Inflation is beaten, markets should fly!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-003",
            "media": [],
            "replies_count": 290,
            "reblogs_count": 900,
            "favourites_count": 5800,
        },
        {
            "id": "fixture-004",
            "created_at": "2026-01-02T16:05:00Z",
            "received_at": "2026-01-02T16:05:01Z",
            "content": "Fake news media is at it again. So unfair!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-004",
            "media": [],
            "replies_count": 250,
            "reblogs_count": 700,
            "favourites_count": 5200,
        },
        {
            "id": "fixture-005",
            "created_at": "2026-01-05T14:40:11Z",
            "received_at": "2026-01-05T14:40:13Z",
            "content": "American AI and semiconductor companies are the future. Support our tech leaders!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-005",
            "media": [],
            "replies_count": 220,
            "reblogs_count": 780,
            "favourites_count": 5500,
        },
        {
            "id": "fixture-006",
            "created_at": "2026-01-05T14:50:30Z",
            "received_at": "2026-01-05T14:50:35Z",
            "content": "Iran is playing with fire. Energy security and strength are needed immediately!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-006",
            "media": ["https://example.invalid/iran-image.jpg"],
            "replies_count": 500,
            "reblogs_count": 1500,
            "favourites_count": 7200,
        },
        {
            "id": "fixture-007",
            "created_at": "2026-01-05T16:00:05Z",
            "received_at": "2026-01-05T16:00:06Z",
            "content": "Great crowd today. Thank you!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-007",
            "media": [],
            "replies_count": 200,
            "reblogs_count": 600,
            "favourites_count": 5100,
        },
        {
            "id": "fixture-008",
            "created_at": "2026-01-05T16:10:25Z",
            "received_at": "2026-01-05T16:10:27Z",
            "content": "Tax cuts and growth. Lower regulation means Nasdaq companies can boom!",
            "url": "https://truthsocial.com/@realDonaldTrump/fixture-008",
            "media": [],
            "replies_count": 260,
            "reblogs_count": 820,
            "favourites_count": 5600,
        },
    ]
    POSTS_PATH.write_text(json.dumps(posts, indent=2) + "\n", encoding="utf-8")


def _write_market() -> None:
    rows = []
    for day, start_price in (
        (datetime(2026, 1, 2, 14, 20, tzinfo=UTC), 17000.0),
        (datetime(2026, 1, 5, 14, 30, tzinfo=UTC), 17120.0),
    ):
        rows.extend(_market_day(day, start_price))

    with MARKET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _market_day(start: datetime, start_price: float) -> list[dict[str, object]]:
    rows = []
    price = start_price
    for minute in range(170):
        ts_open = start + timedelta(minutes=minute)
        shock = _shock(ts_open)
        drift = 0.20 if ts_open.date().day == 5 else -0.05
        open_price = price
        close_price = round(open_price + drift + shock, 2)
        high = round(max(open_price, close_price) + 0.75, 2)
        low = round(min(open_price, close_price) - 0.75, 2)
        volume = 700 + (minute % 13) * 25 + int(abs(shock) * 30)
        rows.append(
            {
                "symbol_root": "NQ",
                "contract_symbol": "NQH6",
                "continuous_symbol": "NQ.c.0",
                "ts_open_utc": ts_open.isoformat().replace("+00:00", "Z"),
                "ts_close_utc": (ts_open + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                "open": round(open_price, 2),
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
                "trade_count": int(volume / 4),
                "vwap": round((open_price + high + low + close_price) / 4, 2),
                "source_name": "fixture_csv",
                "is_rth": "true",
                "session_id": ts_open.strftime("%Y-%m-%d-RTH"),
                "is_rollover_period": "false",
                "is_holiday_session": "false",
                "is_valid_bar": "true",
            }
        )
        price = close_price
    return rows


def _shock(ts_open: datetime) -> float:
    if datetime(2026, 1, 2, 14, 31, tzinfo=UTC) <= ts_open < datetime(2026, 1, 2, 14, 41, tzinfo=UTC):
        return -0.85
    if datetime(2026, 1, 2, 14, 43, tzinfo=UTC) <= ts_open < datetime(2026, 1, 2, 14, 58, tzinfo=UTC):
        return 0.95
    if datetime(2026, 1, 2, 15, 11, tzinfo=UTC) <= ts_open < datetime(2026, 1, 2, 15, 25, tzinfo=UTC):
        return 0.45
    if datetime(2026, 1, 5, 14, 41, tzinfo=UTC) <= ts_open < datetime(2026, 1, 5, 14, 52, tzinfo=UTC):
        return 0.35
    if datetime(2026, 1, 5, 14, 51, tzinfo=UTC) <= ts_open < datetime(2026, 1, 5, 15, 8, tzinfo=UTC):
        return -0.70
    if datetime(2026, 1, 5, 16, 11, tzinfo=UTC) <= ts_open < datetime(2026, 1, 5, 16, 28, tzinfo=UTC):
        return 0.30
    return 0.0


def _write_calendar() -> None:
    rows = [
        {
            "event_id": "fixture-cpi",
            "event_name": "CPI",
            "event_type": "cpi",
            "scheduled_at_utc": "2026-01-02T13:30:00Z",
            "importance": "high",
        },
        {
            "event_id": "fixture-fed-speaker",
            "event_name": "Fed Speaker",
            "event_type": "fed_speaker",
            "scheduled_at_utc": "2026-01-05T15:00:00Z",
            "importance": "medium",
        },
    ]
    with CALENDAR_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_trades() -> None:
    rows = [
        {
            "trade_id": "trade-001",
            "entry_ts_utc": "2026-01-02T14:33:00Z",
            "exit_ts_utc": "2026-01-02T14:52:00Z",
            "side": "long",
            "contracts": 2,
            "entry_price": 16991.25,
            "exit_price": 16986.25,
        },
        {
            "trade_id": "trade-002",
            "entry_ts_utc": "2026-01-02T15:12:00Z",
            "exit_ts_utc": "2026-01-02T15:28:00Z",
            "side": "long",
            "contracts": 1,
            "entry_price": 17002.25,
            "exit_price": 17009.00,
        },
        {
            "trade_id": "trade-003",
            "entry_ts_utc": "2026-01-05T14:53:00Z",
            "exit_ts_utc": "2026-01-05T15:12:00Z",
            "side": "long",
            "contracts": 2,
            "entry_price": 17131.00,
            "exit_price": 17119.00,
        },
        {
            "trade_id": "trade-004",
            "entry_ts_utc": "2026-01-05T16:12:00Z",
            "exit_ts_utc": "2026-01-05T16:32:00Z",
            "side": "long",
            "contracts": 1,
            "entry_price": 17125.00,
            "exit_price": 17131.50,
        },
    ]
    with TRADES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
