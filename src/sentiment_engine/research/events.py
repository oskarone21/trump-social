from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from sentiment_engine.config import EngineConfig
from sentiment_engine.schemas import EventTargetRecord, PostRecord
from sentiment_engine.utils.hashing import stable_hash

FLAT_DIRECTION_THRESHOLD_TICKS = 4


@dataclass(frozen=True)
class EventBuildResult:
    events: pd.DataFrame
    skipped_posts: list[dict[str, str]]


def build_event_dataset(
    posts: list[PostRecord], market_bars: pd.DataFrame, config: EngineConfig
) -> EventBuildResult:
    valid_bars = (
        market_bars[market_bars["is_valid_bar"]]
        .sort_values("ts_open_utc")
        .reset_index(drop=True)
    )
    records: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for post in posts:
        future_bars = valid_bars[valid_bars["ts_open_utc"] >= pd.Timestamp(post.received_at_utc)]
        if future_bars.empty:
            skipped.append({"post_id": post.post_id, "reason": "no_market_bar_after_received_at"})
            continue
        event_bar = future_bars.iloc[0]
        event_ts = event_bar["ts_open_utc"]
        horizon_values = _horizon_values(valid_bars, event_ts, float(event_bar["open"]), config)
        if horizon_values is None:
            skipped.append({"post_id": post.post_id, "reason": "insufficient_forward_bars"})
            continue
        event_id = stable_hash(f"{post.post_id}|{event_ts.isoformat()}")[:16]
        event_record = EventTargetRecord(
            event_id=event_id,
            post_id=post.post_id,
            received_at_utc=post.received_at_utc,
            aligned_bar_ts_utc=event_ts.to_pydatetime(),
            base_price=float(event_bar["open"]),
            **horizon_values,
        )
        merged = post.model_dump(mode="json")
        merged.update(event_record.model_dump(mode="json"))
        records.append(merged)
    return EventBuildResult(events=pd.DataFrame(records), skipped_posts=skipped)


def _horizon_values(
    bars: pd.DataFrame, event_ts: pd.Timestamp, base_price: float, config: EngineConfig
) -> dict[str, object] | None:
    tick_size = config.instruments.tick_size
    values: dict[str, object] = {}
    for horizon in config.windows.impact_horizons_minutes:
        target_ts = event_ts + pd.Timedelta(minutes=horizon)
        target_bars = bars[bars["ts_open_utc"] >= target_ts]
        if target_bars.empty:
            return None
        target_close = float(target_bars.iloc[0]["close"])
        delta_ticks = round((target_close - base_price) / tick_size, 4)
        values[f"nq_delta_{horizon}m_ticks"] = delta_ticks
        values[f"nq_direction_{horizon}m"] = _direction(delta_ticks)
        window = bars[(bars["ts_open_utc"] >= event_ts) & (bars["ts_open_utc"] <= target_ts)]
        mfe_ticks, mae_ticks = _excursion_ticks(window, base_price, tick_size)
        values[f"max_favourable_excursion_{horizon}m_ticks"] = mfe_ticks
        values[f"max_adverse_excursion_{horizon}m_ticks"] = mae_ticks
        values[f"realised_range_{horizon}m_ticks"] = _range_ticks(window, tick_size)
        values[f"realised_volatility_{horizon}m_ticks"] = _realised_volatility_ticks(
            window, tick_size
        )

    full_window = bars[
        (bars["ts_open_utc"] >= event_ts)
        & (
            bars["ts_open_utc"]
            <= event_ts + pd.Timedelta(minutes=config.windows.whipsaw_evaluation_window_minutes)
        )
    ]
    if full_window.empty:
        return None
    whipsaw = _market_whipsaw_flag(full_window, base_price, tick_size, config)
    values["market_whipsaw_flag"] = whipsaw
    values["tradeability_label"] = _tradeability(values, whipsaw)
    return values


def _direction(delta_ticks: float) -> str:
    if delta_ticks >= FLAT_DIRECTION_THRESHOLD_TICKS:
        return "up"
    if delta_ticks <= -FLAT_DIRECTION_THRESHOLD_TICKS:
        return "down"
    return "flat"


def _range_ticks(window: pd.DataFrame, tick_size: float) -> float:
    if window.empty:
        return 0.0
    return round((float(window["high"].max()) - float(window["low"].min())) / tick_size, 4)


def _excursion_ticks(
    window: pd.DataFrame, base_price: float, tick_size: float
) -> tuple[float, float]:
    if window.empty:
        return 0.0, 0.0
    mfe = max((float(window["high"].max()) - base_price) / tick_size, 0.0)
    mae = max((base_price - float(window["low"].min())) / tick_size, 0.0)
    return round(mfe, 4), round(mae, 4)


def _realised_volatility_ticks(window: pd.DataFrame, tick_size: float) -> float:
    if len(window) < 2:
        return 0.0
    close_changes_ticks = window["close"].astype(float).diff().dropna() / tick_size
    return round(float((close_changes_ticks.pow(2).sum()) ** 0.5), 4)


def _market_whipsaw_flag(
    window: pd.DataFrame, base_price: float, tick_size: float, config: EngineConfig
) -> bool:
    first_10m_end = window.iloc[0]["ts_open_utc"] + timedelta(minutes=10)
    first_10m = window[window["ts_open_utc"] <= first_10m_end]
    up_initial = (float(first_10m["high"].max()) - base_price) / tick_size
    down_initial = (base_price - float(first_10m["low"].min())) / tick_size
    initial_direction = "up" if up_initial >= down_initial else "down"
    initial_move = max(up_initial, down_initial)
    if initial_move < config.thresholds.whipsaw_initial_move_ticks:
        return False
    if initial_direction == "up":
        extreme_index = window["high"].idxmax()
        subsequent = window.loc[extreme_index:]
        extreme_high = float(window.loc[extreme_index, "high"])
        reversal = (extreme_high - float(subsequent["low"].min())) / tick_size
    else:
        extreme_index = window["low"].idxmin()
        subsequent = window.loc[extreme_index:]
        extreme_low = float(window.loc[extreme_index, "low"])
        reversal = (float(subsequent["high"].max()) - extreme_low) / tick_size
    reversal_threshold = max(0.65 * initial_move, config.thresholds.whipsaw_reversal_ticks)
    return reversal >= reversal_threshold


def _tradeability(values: dict[str, object], whipsaw: bool) -> str:
    if whipsaw:
        return "no_trade_whipsaw"
    range_30m = float(values["realised_range_30m_ticks"])
    delta_30m = abs(float(values["nq_delta_30m_ticks"]))
    if range_30m >= 35 and delta_30m < 12:
        return "volatility_only"
    if abs(float(values["nq_delta_15m_ticks"])) >= 8:
        return "tradeable_directional"
    if abs(float(values["nq_delta_30m_ticks"])) <= 4:
        return "no_impact"
    return "ambiguous"
