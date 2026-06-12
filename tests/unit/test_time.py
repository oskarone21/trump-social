from __future__ import annotations

from datetime import datetime

import pytest

from sentiment_engine.utils.time import parse_utc


def test_parse_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="Naive datetimes"):
        parse_utc(datetime(2026, 1, 1, 12, 0))


def test_parse_utc_accepts_z_suffix() -> None:
    parsed = parse_utc("2026-01-01T12:00:00Z")
    assert parsed.tzinfo is not None
    assert parsed.isoformat() == "2026-01-01T12:00:00+00:00"
