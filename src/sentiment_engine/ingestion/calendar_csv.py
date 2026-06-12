from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_engine.utils.time import to_utc_series

CALENDAR_REQUIRED_COLUMNS = ["event_id", "event_name", "event_type", "scheduled_at_utc", "importance"]


def load_macro_calendar(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in CALENDAR_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Macro calendar CSV missing required columns: {missing}")
    frame = frame[CALENDAR_REQUIRED_COLUMNS].copy()
    frame["scheduled_at_utc"] = to_utc_series(frame["scheduled_at_utc"])
    return frame.sort_values("scheduled_at_utc")
