from __future__ import annotations

import argparse

from scripts.run_archive_backfill import (
    build_freshness_command,
    build_ingest_command,
    should_run_freshness_check,
)


def test_archive_backfill_builds_ingest_and_freshness_commands() -> None:
    args = argparse.Namespace(
        config="configs/research.yaml",
        url="https://example.test/truth_archive.parquet",
        limit=100,
        out="data/processed/archive.parquet",
        stale_after_minutes=60,
        skip_freshness=False,
    )

    assert build_ingest_command(args) == [
        "--config",
        "configs/research.yaml",
        "ingest-archive",
        "--out",
        "data/processed/archive.parquet",
        "--url",
        "https://example.test/truth_archive.parquet",
        "--limit",
        "100",
    ]
    assert build_freshness_command(args) == [
        "--config",
        "configs/research.yaml",
        "check-archive-freshness",
        "--posts",
        "data/processed/archive.parquet",
        "--stale-after-minutes",
        "60",
        "--url",
        "https://example.test/truth_archive.parquet",
    ]


def test_archive_backfill_skips_remote_freshness_for_local_sources() -> None:
    args = argparse.Namespace(skip_freshness=False, url="data/fixtures/posts.json")

    assert should_run_freshness_check(args) is False
