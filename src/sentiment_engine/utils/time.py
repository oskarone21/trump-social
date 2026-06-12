from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def parse_utc(value: Any) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime-compatible value, got {type(value).__name__}")
    if value.tzinfo is None:
        raise ValueError("Naive datetimes are not accepted; timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


def to_utc_series(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, utc=True, errors="raise", format="mixed")
    if converted.isna().any():
        raise ValueError("Timestamp conversion produced null values")
    return converted


def isoformat_z(value: datetime) -> str:
    return parse_utc(value).isoformat().replace("+00:00", "Z")
