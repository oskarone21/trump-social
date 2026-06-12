from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of records")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError(f"Unsupported dataframe output format: {output.suffix}")
