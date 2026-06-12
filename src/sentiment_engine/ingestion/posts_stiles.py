from __future__ import annotations

from pathlib import Path

from sentiment_engine.ingestion.posts_fixture import load_fixture_posts
from sentiment_engine.schemas import PostRecord


def load_stiles_archive(path: str | Path) -> list[PostRecord]:
    return load_fixture_posts(path)
