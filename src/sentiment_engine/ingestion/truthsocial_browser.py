from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from sentiment_engine.ingestion.posts_external_provider import (
    normalise_provider_record,
    truthsocial_provider_posts_to_frame,
)
from sentiment_engine.schemas import PostRecord
from sentiment_engine.utils.io import write_dataframe, write_json
from sentiment_engine.utils.time import isoformat_z

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError
    sync_playwright = None

TRUTHSOCIAL_BROWSER_PROVIDER = "truthsocial_browser_scraper"
TRUTHSOCIAL_BROWSER_SOURCE_NAME = "truthsocial_browser_live"
TRUTHSOCIAL_PROFILE_URL = "https://truthsocial.com/@realDonaldTrump"
TRUTHSOCIAL_ACCOUNT_ID = "107780257626128497"
TRUTHSOCIAL_STATUSES_URL = (
    f"https://truthsocial.com/api/v1/accounts/{TRUTHSOCIAL_ACCOUNT_ID}/statuses"
    "?limit=20&exclude_replies=false"
)
DEFAULT_STORAGE_STATE_PATH = Path("data/interim/truthsocial_browser_storage_state.json")
DEFAULT_CANONICAL_OUT = Path("data/processed/truthsocial_browser_posts.parquet")
DEFAULT_REPORT_OUT = Path("reports/truthsocial_browser_scraper_report.json")
DEFAULT_RAW_DIR = Path("data/raw/truthsocial_browser")
DEFAULT_USERNAME_ENV = "TRUTHSOCIAL_USERNAME"
DEFAULT_PASSWORD_ENV = "TRUTHSOCIAL_PASSWORD"
DEFAULT_TOTP_SECRET_ENV = "TRUTHSOCIAL_TOTP_SECRET"
DEFAULT_POLL_SECONDS = 5
DEFAULT_STALE_AFTER_SECONDS = 10
AUTH_STATUS_AUTHENTICATED = "authenticated"
AUTH_STATUS_FIXTURE = "fixture"
AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED = "challenge_or_login_failed"
AUTH_STATUS_MISSING_CREDENTIALS = "missing_credentials"
BROWSER_STATUS_OK = "ok"
BROWSER_STATUS_FIXTURE = "fixture"
BROWSER_STATUS_FAILED = "failed"
STATUS_PAYLOAD_KEYS = ("posts", "data", "items", "results", "response")
STATUS_URL_MARKER = "/api/v1/accounts/"
STATUS_URL_SUFFIX = "/statuses"
LOGIN_URL = "https://truthsocial.com/login"
HTML_CHALLENGE_MARKERS = ("cloudflare", "checking your browser", "cf-mitigated", "challenge")
USERNAME_SELECTORS = (
    'input[name="username"]',
    'input[name="email"]',
    'input[type="email"]',
    'input[autocomplete="username"]',
)
PASSWORD_SELECTORS = (
    'input[name="password"]',
    'input[type="password"]',
    'input[autocomplete="current-password"]',
)
TOTP_SELECTORS = (
    'input[name="otp_attempt"]',
    'input[name="code"]',
    'input[autocomplete="one-time-code"]',
    'input[type="tel"]',
)
PLAYWRIGHT_FAILURES = (PlaywrightError, PlaywrightTimeoutError)
BROWSER_RECOVERABLE_FAILURES = (*PLAYWRIGHT_FAILURES, RuntimeError, OSError)
POST_NORMALISATION_FAILURES = (TypeError, ValueError, ValidationError)
STATUS_FETCH_FAILURES = (*PLAYWRIGHT_FAILURES, TypeError, ValueError, json.JSONDecodeError)


@dataclass(frozen=True)
class BrowserScraperSettings:
    profile_url: str = TRUTHSOCIAL_PROFILE_URL
    account_id: str = TRUTHSOCIAL_ACCOUNT_ID
    source_name: str = TRUTHSOCIAL_BROWSER_SOURCE_NAME
    source_provider: str = TRUTHSOCIAL_BROWSER_PROVIDER
    storage_state_path: Path = DEFAULT_STORAGE_STATE_PATH
    username_env: str = DEFAULT_USERNAME_ENV
    password_env: str = DEFAULT_PASSWORD_ENV
    totp_secret_env: str = DEFAULT_TOTP_SECRET_ENV
    poll_seconds: int = DEFAULT_POLL_SECONDS
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS
    canonical_out: Path = DEFAULT_CANONICAL_OUT
    report_out: Path = DEFAULT_REPORT_OUT
    raw_dir: Path = DEFAULT_RAW_DIR
    headless: bool = True

    @property
    def statuses_url(self) -> str:
        return (
            f"https://truthsocial.com/api/v1/accounts/{self.account_id}/statuses"
            "?limit=20&exclude_replies=false"
        )


@dataclass(frozen=True)
class ScrapeResult:
    records: list[PostRecord]
    raw_rows: list[dict[str, Any]]
    report: dict[str, Any]
    canonical_out: Path
    raw_out: Path
    report_out: Path


def load_truthsocial_status_rows(source: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    rows = _extract_status_rows(payload)
    if not rows:
        raise ValueError("Truth Social status payload did not contain any posts")
    return [_clean_status_row(row) for row in rows]


def normalise_truthsocial_status_rows(
    rows: list[dict[str, Any]],
    *,
    source_name: str = TRUTHSOCIAL_BROWSER_SOURCE_NAME,
    source_provider: str = TRUTHSOCIAL_BROWSER_PROVIDER,
    received_at_utc: datetime | None = None,
) -> list[PostRecord]:
    received_at = received_at_utc or datetime.now(UTC)
    records = [
        normalise_provider_record(
            _provider_row_from_status(row, received_at),
            source_name=source_name,
            source_provider=source_provider,
        )
        for row in rows
    ]
    return sorted(_dedupe_records(records), key=lambda record: (record.created_at_utc, record.post_id))


def run_truthsocial_fixture_scrape(
    *,
    fixture_path: str | Path,
    settings: BrowserScraperSettings,
    checked_at_utc: datetime | None = None,
) -> ScrapeResult:
    checked_at = checked_at_utc or datetime.now(UTC)
    raw_rows = [
        _provider_row_from_status(row, checked_at)
        for row in load_truthsocial_status_rows(fixture_path)
    ]
    records = normalise_truthsocial_status_rows(
        raw_rows,
        source_name=settings.source_name,
        source_provider=settings.source_provider,
        received_at_utc=checked_at,
    )
    return _persist_scrape_result(
        records=records,
        raw_rows=raw_rows,
        settings=settings,
        auth_status=AUTH_STATUS_FIXTURE,
        browser_status=BROWSER_STATUS_FIXTURE,
        checked_at_utc=checked_at,
        errors=[],
    )


def run_truthsocial_browser_scrape_once(
    *,
    settings: BrowserScraperSettings,
    checked_at_utc: datetime | None = None,
) -> ScrapeResult:
    checked_at = checked_at_utc or datetime.now(UTC)
    try:
        raw_rows, auth_status, browser_status, errors = _collect_with_playwright(settings)
    except BROWSER_RECOVERABLE_FAILURES as exc:
        raw_rows = []
        auth_status = AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED
        browser_status = BROWSER_STATUS_FAILED
        errors = [_safe_error(exc)]

    records: list[PostRecord] = []
    schema_errors: list[str] = []
    if raw_rows:
        try:
            records = normalise_truthsocial_status_rows(
                raw_rows,
                source_name=settings.source_name,
                source_provider=settings.source_provider,
                received_at_utc=checked_at,
            )
        except POST_NORMALISATION_FAILURES as exc:
            schema_errors.append(_safe_error(exc))

    return _persist_scrape_result(
        records=records,
        raw_rows=raw_rows,
        settings=settings,
        auth_status=auth_status,
        browser_status=browser_status,
        checked_at_utc=checked_at,
        errors=[*errors, *schema_errors],
    )


def _collect_with_playwright(
    settings: BrowserScraperSettings,
) -> tuple[list[dict[str, Any]], str, str, list[str]]:
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Run: pip install -e \".[scraper]\" "
            "and python -m playwright install chromium"
        )

    username = os.getenv(settings.username_env, "").strip()
    password = os.getenv(settings.password_env, "").strip()
    if not username or not password:
        return [], AUTH_STATUS_MISSING_CREDENTIALS, BROWSER_STATUS_FAILED, [
            f"Missing {settings.username_env}/{settings.password_env}"
        ]

    captured_rows: list[dict[str, Any]] = []
    discovered_url: str | None = None
    errors: list[str] = []
    storage_state = (
        str(settings.storage_state_path)
        if settings.storage_state_path.exists()
        else None
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=settings.headless)
        context_kwargs: dict[str, Any] = {"storage_state": storage_state} if storage_state else {}
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        def capture_response(response) -> None:
            nonlocal discovered_url
            if not _looks_like_status_url(response.url):
                return
            discovered_url = response.url
            try:
                payload = response.json()
            except STATUS_FETCH_FAILURES as exc:
                errors.append(_safe_error(exc))
                return
            captured_rows.extend(_rows_from_browser_payload(payload))

        page.on("response", capture_response)
        page.goto(settings.profile_url, wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(3_000)

        if _page_has_challenge(page):
            browser.close()
            return [], AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED, BROWSER_STATUS_FAILED, [
                "Truth Social returned a challenge page"
            ]

        if not captured_rows:
            auth_ok, auth_errors = _login(page, context, settings)
            errors.extend(auth_errors)
            if not auth_ok:
                browser.close()
                return [], AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED, BROWSER_STATUS_FAILED, errors
            page.goto(settings.profile_url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(3_000)

        fetch_url = discovered_url or settings.statuses_url
        if not captured_rows:
            payload, fetch_error = _fetch_statuses_in_page(page, fetch_url)
            if payload is not None:
                captured_rows.extend(_rows_from_browser_payload(payload))
            if fetch_error:
                errors.append(fetch_error)

        if captured_rows:
            settings.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(settings.storage_state_path))
            browser.close()
            return (
                [_provider_row_from_status(row, datetime.now(UTC)) for row in captured_rows],
                AUTH_STATUS_AUTHENTICATED,
                BROWSER_STATUS_OK,
                errors,
            )

        browser.close()
        return [], AUTH_STATUS_CHALLENGE_OR_LOGIN_FAILED, BROWSER_STATUS_FAILED, errors


def _login(page, context, settings: BrowserScraperSettings) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(1_000)
        if _page_has_challenge(page):
            return False, ["Truth Social login returned a challenge page"]
        _fill_first(page, USERNAME_SELECTORS, os.getenv(settings.username_env, ""))
        _fill_first(page, PASSWORD_SELECTORS, os.getenv(settings.password_env, ""))
        _click_submit(page)
        page.wait_for_timeout(3_000)
        totp_secret = os.getenv(settings.totp_secret_env, "").strip()
        if totp_secret:
            code = _totp_code(totp_secret)
            if _fill_first(page, TOTP_SELECTORS, code, required=False):
                _click_submit(page)
                page.wait_for_timeout(3_000)
        settings.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(settings.storage_state_path))
        return not _page_has_challenge(page), errors
    except BROWSER_RECOVERABLE_FAILURES as exc:
        return False, [*_safe_error_list(errors), _safe_error(exc)]


def _fill_first(page, selectors: tuple[str, ...], value: str, *, required: bool = True) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if locator.count() > 0:
                locator.fill(value, timeout=5_000)
                return True
        except PLAYWRIGHT_FAILURES:
            continue
    if required:
        raise RuntimeError(f"Could not find login field among selectors: {selectors}")
    return False


def _click_submit(page) -> None:
    selectors = ('button[type="submit"]', 'button:has-text("Log in")', 'button:has-text("Sign in")')
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=5_000)
                return
        except PLAYWRIGHT_FAILURES:
            continue
    page.keyboard.press("Enter")


def _fetch_statuses_in_page(page, url: str) -> tuple[Any | None, str | None]:
    script = """
        async (url) => {
            const response = await fetch(url, {
                credentials: "include",
                headers: {"Accept": "application/json"}
            });
            const text = await response.text();
            return {status: response.status, text};
        }
    """
    try:
        result = page.evaluate(script, url)
        if not (200 <= int(result["status"]) < 300):
            return None, f"status_fetch_http_{result['status']}"
        return json.loads(result["text"]), None
    except STATUS_FETCH_FAILURES as exc:
        return None, _safe_error(exc)


def _persist_scrape_result(
    *,
    records: list[PostRecord],
    raw_rows: list[dict[str, Any]],
    settings: BrowserScraperSettings,
    auth_status: str,
    browser_status: str,
    checked_at_utc: datetime,
    errors: list[str],
) -> ScrapeResult:
    raw_out = _raw_output_path(settings.raw_dir, checked_at_utc)
    if raw_rows:
        _append_raw_jsonl(raw_out, raw_rows)

    existing_records = _read_existing_records(settings.canonical_out)
    merged_records = _dedupe_records([*existing_records, *records])
    write_dataframe(truthsocial_provider_posts_to_frame(merged_records), settings.canonical_out)

    report = build_truthsocial_browser_scraper_report(
        records=merged_records,
        raw_rows=raw_rows,
        settings=settings,
        auth_status=auth_status,
        browser_status=browser_status,
        checked_at_utc=checked_at_utc,
        errors=errors,
    )
    write_json(settings.report_out, report)
    return ScrapeResult(
        records=merged_records,
        raw_rows=raw_rows,
        report=report,
        canonical_out=settings.canonical_out,
        raw_out=raw_out,
        report_out=settings.report_out,
    )


def build_truthsocial_browser_scraper_report(
    *,
    records: list[PostRecord],
    raw_rows: list[dict[str, Any]],
    settings: BrowserScraperSettings,
    auth_status: str,
    browser_status: str,
    checked_at_utc: datetime,
    errors: list[str],
) -> dict[str, Any]:
    latest = max(records, key=lambda record: record.created_at_utc, default=None)
    latest_received = max((record.received_at_utc for record in records), default=None)
    collection_lag_seconds = (
        int((checked_at_utc - latest_received).total_seconds()) if latest_received else None
    )
    latest_post_age_seconds = (
        int((checked_at_utc - latest.created_at_utc).total_seconds()) if latest else None
    )
    duplicate_rows = _raw_duplicate_count(raw_rows)
    schema_drift = bool(errors) and not records
    is_stale = (
        auth_status != AUTH_STATUS_AUTHENTICATED
        and auth_status != AUTH_STATUS_FIXTURE
    ) or browser_status == BROWSER_STATUS_FAILED
    if collection_lag_seconds is not None:
        is_stale = is_stale or collection_lag_seconds > settings.stale_after_seconds
    return {
        "source_name": settings.source_name,
        "source_provider": settings.source_provider,
        "target_profile_url": settings.profile_url,
        "account_id": settings.account_id,
        "checked_at_utc": isoformat_z(checked_at_utc),
        "auth_status": auth_status,
        "browser_status": browser_status,
        "poll_seconds": settings.poll_seconds,
        "stale_after_seconds": settings.stale_after_seconds,
        "raw_rows_seen": len(raw_rows),
        "canonical_rows_written": len(records),
        "duplicate_rows": duplicate_rows,
        "latest_post_id": latest.post_id if latest else None,
        "latest_created_at_utc": isoformat_z(latest.created_at_utc) if latest else None,
        "latest_received_at_utc": isoformat_z(latest_received) if latest_received else None,
        "latest_post_age_seconds": latest_post_age_seconds,
        "collection_lag_seconds": collection_lag_seconds,
        "schema_drift_detected": schema_drift,
        "is_stale": bool(is_stale),
        "output_paths": {
            "canonical_posts": str(settings.canonical_out),
            "raw_jsonl_dir": str(settings.raw_dir),
            "storage_state": str(settings.storage_state_path),
        },
        "errors": _safe_error_list(errors),
        "methodology_notes": [
            "Authenticated browser collection is advisory and may break when Truth Social changes.",
            "No proxy rotation, CAPTCHA bypass, stealth fingerprinting, or anti-bot evasion is used.",
            "Failures must be treated as stale provider data by downstream signal generation.",
        ],
    }


def _extract_status_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in STATUS_PAYLOAD_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            nested = _extract_status_rows(value)
            if nested:
                return nested
    return []


def _rows_from_browser_payload(payload: Any) -> list[dict[str, Any]]:
    return [_clean_status_row(row) for row in _extract_status_rows(payload)]


def _clean_status_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items()}


def _provider_row_from_status(row: dict[str, Any], received_at_utc: datetime) -> dict[str, Any]:
    output = dict(row)
    output.setdefault("received_at", isoformat_z(received_at_utc))
    output.setdefault("collected_at", isoformat_z(received_at_utc))
    if "text" not in output and "content" in output:
        output["text"] = output["content"]
    if "author_id" not in output:
        account = output.get("account")
        if isinstance(account, dict) and account.get("id") is not None:
            output["author_id"] = account["id"]
    if "post_type" not in output:
        output["post_type"] = _status_post_type(output)
    return output


def _status_post_type(row: dict[str, Any]) -> str:
    if row.get("reblog"):
        return "retruth"
    if row.get("quote_id"):
        return "quote"
    if row.get("in_reply_to_id"):
        return "reply"
    return "original"


def _dedupe_records(records: list[PostRecord]) -> list[PostRecord]:
    by_key: dict[tuple[str, str], PostRecord] = {}
    for record in records:
        key = (record.post_id, record.content_hash)
        existing = by_key.get(key)
        if existing is None or record.received_at_utc >= existing.received_at_utc:
            by_key[key] = record
    return sorted(by_key.values(), key=lambda record: (record.created_at_utc, record.post_id))


def _raw_duplicate_count(rows: list[dict[str, Any]]) -> int:
    keys: list[tuple[str, str]] = []
    for row in rows:
        post_id = str(row.get("post_id") or row.get("id") or row.get("postId") or "")
        content = str(row.get("content") or row.get("text") or row.get("body") or "")
        keys.append((post_id, content))
    return len(keys) - len(set(keys))


def _read_existing_records(path: Path) -> list[PostRecord]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    return [PostRecord.model_validate(row) for row in frame.to_dict("records")]


def _append_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _raw_output_path(raw_dir: Path, checked_at_utc: datetime) -> Path:
    return raw_dir / f"{checked_at_utc:%Y%m%d}.jsonl"


def _looks_like_status_url(url: str) -> bool:
    return STATUS_URL_MARKER in url and STATUS_URL_SUFFIX in url


def _page_has_challenge(page) -> bool:
    try:
        body = page.locator("body").inner_text(timeout=3_000).lower()
    except PLAYWRIGHT_FAILURES:
        body = ""
    try:
        title = page.title().lower()
    except PLAYWRIGHT_FAILURES:
        title = ""
    combined = f"{title} {body}"
    return any(marker in combined for marker in HTML_CHALLENGE_MARKERS)


def _totp_code(secret: str, *, timestamp: int | None = None, step: int = 30, digits: int = 6) -> str:
    normalised = secret.replace(" ", "").upper()
    key = base64.b32decode(normalised, casefold=True)
    counter = int((timestamp or int(time.time())) / step).to_bytes(8, "big")
    digest = hmac.new(key, counter, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (
        ((digest[offset] & 0x7F) << 24)
        | ((digest[offset + 1] & 0xFF) << 16)
        | ((digest[offset + 2] & 0xFF) << 8)
        | (digest[offset + 3] & 0xFF)
    ) % (10**digits)
    return str(code).zfill(digits)


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _safe_error_list(errors: list[str]) -> list[str]:
    password = os.getenv(DEFAULT_PASSWORD_ENV, "")
    safe_errors = []
    for error in errors:
        safe = str(error)
        if password:
            safe = safe.replace(password, "[redacted]")
        safe_errors.append(safe)
    return safe_errors
