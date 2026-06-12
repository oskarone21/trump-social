from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import urlparse

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://[^\s<>\"]+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_post_text(text: str) -> str:
    unescaped = html.unescape(text or "")
    without_tags = HTML_TAG_RE.sub(" ", unescaped)
    with_url_tokens = URL_RE.sub(" URL ", without_tags)
    normalised = unicodedata.normalize("NFKC", with_url_tokens)
    return WHITESPACE_RE.sub(" ", normalised).strip()


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(html.unescape(text or ""))


def extract_url_domains(urls: list[str]) -> list[str]:
    domains = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc:
            domains.append(parsed.netloc.lower())
    return sorted(set(domains))


def punctuation_features(raw_text: str, clean_text: str) -> dict[str, float | int]:
    alpha_chars = [char for char in clean_text if char.isalpha()]
    all_caps = [char for char in alpha_chars if char.isupper()]
    all_caps_ratio = len(all_caps) / len(alpha_chars) if alpha_chars else 0.0
    return {
        "post_length": len(clean_text),
        "token_count": len(clean_text.split()),
        "all_caps_ratio": round(all_caps_ratio, 6),
        "exclamation_count": raw_text.count("!"),
        "question_mark_count": raw_text.count("?"),
    }
