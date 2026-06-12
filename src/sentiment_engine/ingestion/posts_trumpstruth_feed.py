from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any
from sentiment_engine.schemas import PostRecord
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import urlopen
import re
import xml.etree.ElementTree as ET

from sentiment_engine.ingestion.posts_external_provider import normalise_provider_record
from sentiment_engine.ingestion.posts_fixture import posts_to_frame
from sentiment_engine.utils.hashing import stable_hash
from sentiment_engine.ingestion.posts_fixture import audit_posts

TRUMPSTRUTH_PROVIDER_NAME = "trumpstruth_rss"
TRUMPSTRUTH_SOURCE_NAME = "trumpstruth_feed"
HTTP_PREFIXES = ("http://", "https://")


def load_trumpstruth_feed_posts(
    source: str,
    *,
    source_name: str,
    source_provider: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> list[PostRecord]:
    source_url = _append_date_filters(source, start_date=start_date, end_date=end_date)
    rows = _parse_feed_rows(source_url)
    if limit is not None:
        rows = rows[:limit]
    records = [
        normalise_provider_record(
            row,
            source_name=source_name,
            source_provider=source_provider,
        )
        for row in rows
    ]
    return sorted(_dedupe_rows(records), key=lambda record: (record.created_at_utc, record.post_id))


def load_trumpstruth_feed_rows(
    source: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    source_url = _append_date_filters(source, start_date=start_date, end_date=end_date)
    return _parse_feed_rows(source_url)


def trumpstruth_feed_posts_to_frame(records: list[PostRecord]) -> Any:
    return posts_to_frame(records)


def audit_trumpstruth_feed_posts(
    records: list[PostRecord], *, source: str
) -> dict[str, Any]:
    audit = audit_posts(records)
    audit.update(
        {
            "source": source,
            "source_provider": TRUMPSTRUTH_PROVIDER_NAME,
            "historical_backfill_only": True,
            "source_is_live_capable": False,
            "dl_readiness_note": (
                "This is an archival RSS source; it is suitable for backfill and reference checks "
                "but not designed as the only low-latency live feed."
            ),
        }
    )
    return audit


def _append_date_filters(
    source: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    if not source.startswith(HTTP_PREFIXES):
        return source
    parsed = urlparse(source)
    query = parse_qs(parsed.query or "", keep_blank_values=True)
    if start_date is not None:
        query["start_date"] = [start_date]
    if end_date is not None:
        query["end_date"] = [end_date]
    ordered = sorted((name, values[0]) for name, values in query.items())
    encoded_query = urlencode(ordered, doseq=False)
    return urlunparse(parsed._replace(query=encoded_query))


def _parse_feed_rows(source: str) -> list[dict[str, Any]]:
    payload = _read_feed_payload(source)
    root = ET.fromstring(payload.decode("utf-8"))
    items = root.findall("./channel/item")
    if not items:
        items = root.findall(".//item")
    if not items:
        raise ValueError("Trumpstruth feed contained no <item> entries")
    return [_parse_feed_item(item) for item in items]


def _read_feed_payload(source: str) -> bytes:
    source_text = str(source)
    if source_text.startswith(HTTP_PREFIXES):
        with urlopen(source_text, timeout=30) as response:
            return response.read()
    return Path(source_text).read_bytes()


def _parse_feed_item(item: ET.Element) -> dict[str, Any]:
    link = _find_tag_text(item, "link") or ""
    guid = _find_tag_text(item, "guid") or _extract_id_from_link(link) or _derive_fallback_id(item)
    title = _find_tag_text(item, "title") or ""
    description = _find_tag_text(item, "description") or ""
    description_text = _html_to_text(description)
    created_at = _parse_pub_date(item)
    raw_text = " ".join(part for part in (title.strip(), description_text) if part).strip()
    media_urls = _extract_media_urls(item)
    all_urls = _extract_urls(link, description)
    text_raw = " ".join(part for part in (description_text, title.strip()) if part).strip()
    if not text_raw:
        text_raw = raw_text
    author_id = _normalise_author(item)
    post_type = _normalise_post_type(_find_tag_text(item, "category"), title, text_raw)
    return {
        "post_id": guid,
        "id": guid,
        "created_at": created_at,
        "received_at": created_at,
        "text": text_raw,
        "media_urls": media_urls,
        "media": media_urls,
        "mediaUrls": media_urls,
        "urls": all_urls,
        "url": link,
        "language": "en",
        "post_type": post_type,
        "replies_count": 0,
        "reblogs_count": 0,
        "favourites_count": 0,
        "author_id": author_id,
        "author": {"id": author_id},
        "raw_source": ET.tostring(item, encoding="unicode"),
        "title": title,
    }


def _normalise_author(item: ET.Element) -> str:
    values = (
        _find_tag_text(item, "author"),
        _find_tag_text(item, "creator"),
        _find_tag_text(item, "dc:creator"),
    )
    for value in values:
        if value and value.strip():
            return _clean_author(value)
    return "unknown_author"


def _normalise_post_type(category: str | None, title: str, text_raw: str) -> str:
    raw = (category or "").lower()
    candidate = title.lower() + " " + text_raw.lower()
    if raw:
        if "retruth" in raw or "re-truth" in raw:
            return "retruth"
        if raw in {"reply", "re", "retweet"}:
            return "reply"
        if raw in {"quote", "quoted", "quote"}:
            return "quote"
    if candidate.startswith("rt "):
        return "reply"
    if "retruth" in candidate or "retruthed" in candidate:
        return "retruth"
    if "quote:" in candidate:
        return "quote"
    return "original"


def _parse_pub_date(item: ET.Element) -> str:
    raw_pub = (
        _find_tag_text(item, "pubDate")
        or _find_tag_text(item, "pubdate")
        or _find_tag_text(item, "dc:date")
    )
    if not raw_pub:
        raise ValueError("Feed item missing publication timestamp")
    try:
        dt = parsedate_to_datetime(raw_pub)
    except (TypeError, ValueError):
        dt = None
    if dt is None:
        # Fallback for strict ISO style timestamps.
        dt = datetime.fromisoformat(raw_pub.replace("Z", "+00:00"))
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _find_tag_text(item: ET.Element, tag: str) -> str | None:
    exact = item.findtext(tag)
    if exact is not None and exact.strip():
        return exact.strip()
    for child in item.iter():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name == tag or local_name.split(":")[-1] == tag.split(":")[-1]:
            text = child.text
            if text is not None:
                return text.strip()
    return None


def _extract_media_urls(item: ET.Element) -> list[str]:
    urls: list[str] = []
    for child in item.iter():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name in {"enclosure", "content", "media", "media:content"}:
            url = child.get("url")
            if url:
                urls.append(url.strip())
        if local_name in {"image", "url"} and child.text and child.text.strip().lower().startswith("http"):
            urls.append(child.text.strip())
    # Keep order but dedupe.
    return list(dict.fromkeys(u for u in urls if u))


def _extract_urls(link: str, description: str) -> list[str]:
    raw_urls = []
    raw_urls.extend(_find_all_urls(link))
    raw_urls.extend(_find_all_urls(description))
    return list(dict.fromkeys(u for u in raw_urls if u))


def _find_all_urls(value: str) -> list[str]:
    if not value:
        return []
    html_urls = re.findall(r"https?://[^\s\"'>)]+", value)
    return [u.rstrip(")") for u in html_urls]


def _extract_id_from_link(link: str) -> str | None:
    if not link:
        return None
    match = re.search(r"/posts?/([^/?#]+)$", link)
    if match:
        return match.group(1)
    match = re.search(r"/([^/?#]+)$", link)
    if match:
        return match.group(1)
    return None


def _derive_fallback_id(item: ET.Element) -> str:
    candidate = "|".join(
        [(_find_tag_text(item, "pubDate") or ""), (_find_tag_text(item, "title") or ""), (_find_tag_text(item, "description") or "")]
    )
    return stable_hash(candidate)


def _html_to_text(value: str) -> str:
    if not value:
        return ""
    no_scripts = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", value)
    text = re.sub(r"<[^>]+>", " ", no_scripts)
    text = unescape(text)
    text = " ".join(text.split())
    return text.strip()


def _clean_author(value: str) -> str:
    value = value.strip().strip("@")
    return value or "unknown_author"


def _dedupe_rows(records: list[Any]) -> list[Any]:
    by_key: dict[tuple[str, str], Any] = {}
    for record in records:
        key = (record.post_id, record.content_hash)
        existing = by_key.get(key)
        if existing is None or record.ingested_at_utc >= existing.ingested_at_utc:
            by_key[key] = record
    return list(by_key.values())
