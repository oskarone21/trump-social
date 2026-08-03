from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

TEMP_FILE_PREFIX = ".tmp-"


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of records")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")

    write_with_temporary_path(output, writer)


def write_text(path: str | Path, payload: str) -> None:
    write_with_temporary_path(
        path,
        lambda temporary: temporary.write_text(payload, encoding="utf-8"),
    )


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)

    def writer(temporary: Path) -> None:
        if output.suffix == ".parquet":
            df.to_parquet(temporary, index=False)
        elif output.suffix == ".csv":
            df.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported dataframe output format: {output.suffix}")

    write_with_temporary_path(output, writer)


def write_with_temporary_path(path: str | Path, writer: Callable[[Path], None]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output)
    try:
        writer(temporary)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{TEMP_FILE_PREFIX}{path.stem}-{os.getpid()}-{uuid4().hex}{path.suffix}")
