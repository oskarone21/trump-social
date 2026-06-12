from __future__ import annotations

from datetime import datetime

import pytest
import pandas as pd

from sentiment_engine.utils.time import parse_utc, to_utc_series


def test_parse_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="Naive datetimes"):
        parse_utc(datetime(2026, 1, 1, 12, 0))


def test_parse_utc_accepts_z_suffix() -> None:
    parsed = parse_utc("2026-01-01T12:00:00Z")
    assert parsed.tzinfo is not None
    assert parsed.isoformat() == "2026-01-01T12:00:00+00:00"


def test_to_utc_series_accepts_mixed_iso_precision() -> None:
    parsed = to_utc_series(
        pd.Series(["2022-07-17T19:22:14Z", "2026-06-12T13:59:27.160000Z"])
    )

    assert parsed.dt.tz is not None
    assert parsed.iloc[0].isoformat() == "2022-07-17T19:22:14+00:00"
