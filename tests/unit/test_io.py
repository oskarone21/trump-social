from __future__ import annotations

import json

import pandas as pd
import pytest

from sentiment_engine.utils.io import write_dataframe, write_json, write_text, write_with_temporary_path


def test_write_json_replaces_existing_file(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob(".tmp-*"))


def test_write_dataframe_replaces_csv_and_parquet(tmp_path) -> None:
    frame = pd.DataFrame([{"value": 1}, {"value": 2}])
    csv_path = tmp_path / "rows.csv"
    parquet_path = tmp_path / "rows.parquet"

    write_dataframe(frame, csv_path)
    write_dataframe(frame, parquet_path)

    assert pd.read_csv(csv_path)["value"].tolist() == [1, 2]
    assert pd.read_parquet(parquet_path)["value"].tolist() == [1, 2]
    assert not list(tmp_path.glob(".tmp-*"))


def test_write_text_replaces_existing_file(tmp_path) -> None:
    path = tmp_path / "report.md"
    path.write_text("old\n", encoding="utf-8")

    write_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".tmp-*"))


def test_write_with_temporary_path_cleans_up_failed_write(tmp_path) -> None:
    path = tmp_path / "artifact.bin"

    def fail_writer(temporary):
        temporary.write_bytes(b"partial")
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        write_with_temporary_path(path, fail_writer)

    assert not path.exists()
    assert not list(tmp_path.glob(".tmp-*"))
