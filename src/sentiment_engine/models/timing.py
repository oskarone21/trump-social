from __future__ import annotations

import pandas as pd

MINUTES_PER_DAY = 24 * 60


def add_empirical_timing_features(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values("received_at_utc").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(ordered["received_at_utc"], format="mixed", utc=True)
    prior_timestamps = timestamps.shift(1)
    minutes_since_prior = (timestamps - prior_timestamps).dt.total_seconds() / 60.0
    historical_intervals = minutes_since_prior.dropna().tolist()
    ordered["minutes_since_previous_post"] = minutes_since_prior.fillna(MINUTES_PER_DAY)
    for horizon in (15, 30, 60):
        ordered[f"p_next_post_{horizon}m"] = [
            _empirical_probability(historical_intervals[:idx], horizon)
            for idx in range(len(ordered))
        ]
    ordered["expected_minutes_to_next_post"] = [
        _empirical_expected_minutes(historical_intervals[:idx]) for idx in range(len(ordered))
    ]
    return ordered


def _empirical_probability(intervals: list[float], horizon_minutes: int) -> float:
    if not intervals:
        return 0.0
    return round(sum(1 for value in intervals if value <= horizon_minutes) / len(intervals), 6)


def _empirical_expected_minutes(intervals: list[float]) -> float:
    if not intervals:
        return float(MINUTES_PER_DAY)
    return round(float(sum(intervals) / len(intervals)), 4)
