from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.utils.hashing import stable_hash


def dataframe_hash(frame: pd.DataFrame) -> str:
    csv_payload = frame.sort_index(axis=1).to_csv(index=False)
    return stable_hash(csv_payload)


def model_metadata(
    *,
    model_name: str,
    model_version: str,
    config_path: str,
    data: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config_path": str(Path(config_path)),
        "data_hash": dataframe_hash(data),
        "row_count": int(len(data)),
        "extra": extra or {},
    }
