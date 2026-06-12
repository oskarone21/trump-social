from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_engine.models import taxonomy as tx
from sentiment_engine.utils.time import isoformat_z, parse_utc

LABEL_VERSION_DEFAULT = "human_v1"
TOPIC_SEPARATOR = "|"
MARKET_RELEVANCE_LABELS = {"market_relevant", "not_market_relevant", "uncertain"}
CONTRADICTION_LABELS = {
    "none",
    "self_contradiction",
    "prior_contradiction",
    "policy_reversal",
    "uncertain",
}
HUMAN_LABEL_COLUMNS = [
    "event_id",
    "post_id",
    "received_at_utc",
    "text_clean",
    "has_image",
    "has_video",
    "url_domains",
    "rule_sentiment_label",
    "rule_tradeability_label",
    "rule_topic_labels",
    "human_sentiment_label",
    "human_tradeability_label",
    "human_topic_labels",
    "market_relevance_label",
    "contradiction_label",
    "label_confidence",
    "reviewer_id",
    "reviewed_at_utc",
    "review_notes",
]
REQUIRED_REVIEW_COLUMNS = [
    "event_id",
    "post_id",
    "human_sentiment_label",
    "human_tradeability_label",
    "human_topic_labels",
    "market_relevance_label",
    "contradiction_label",
    "label_confidence",
    "reviewer_id",
    "reviewed_at_utc",
]
SENTIMENT_LABELS = {
    tx.SENTIMENT_BULLISH,
    tx.SENTIMENT_BEARISH,
    tx.SENTIMENT_GEOPOLITICAL,
    tx.SENTIMENT_NEUTRAL,
    tx.SENTIMENT_VOLATILITY_ONLY,
    tx.SENTIMENT_LOW_CONFIDENCE,
}
TRADEABILITY_LABELS = {
    tx.TRADEABILITY_DIRECTIONAL,
    tx.TRADEABILITY_VOLATILITY_ONLY,
    tx.TRADEABILITY_NO_TRADE_WHIPSAW,
    tx.TRADEABILITY_NO_IMPACT,
    tx.TRADEABILITY_AMBIGUOUS,
}
TOPIC_LABELS = set(tx.TOPIC_LABELS)


def build_label_queue(events: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    ordered = events.sort_values("received_at_utc").reset_index(drop=True)
    if limit is not None:
        ordered = ordered.head(limit)
    rows = []
    for row in ordered.to_dict("records"):
        rows.append(
            {
                "event_id": row["event_id"],
                "post_id": row["post_id"],
                "received_at_utc": row["received_at_utc"],
                "text_clean": row["text_clean"],
                "has_image": bool(row.get("has_image", False)),
                "has_video": bool(row.get("has_video", False)),
                "url_domains": _join_labels(row.get("url_domains")),
                "rule_sentiment_label": row.get("rule_sentiment_label", ""),
                "rule_tradeability_label": row.get("rule_tradeability_label", ""),
                "rule_topic_labels": _join_labels(row.get("rule_topic_labels")),
                "human_sentiment_label": "",
                "human_tradeability_label": "",
                "human_topic_labels": "",
                "market_relevance_label": "",
                "contradiction_label": "",
                "label_confidence": "",
                "reviewer_id": "",
                "reviewed_at_utc": "",
                "review_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=HUMAN_LABEL_COLUMNS)


def load_reviewed_labels(
    path: str | Path, *, label_version: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path).fillna("")
    missing_columns = [column for column in REQUIRED_REVIEW_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Reviewed label file missing columns: {missing_columns}")
    normalized = _normalise_reviewed_frame(frame, label_version)
    audit = audit_reviewed_labels(normalized)
    return normalized, audit


def audit_reviewed_labels(frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_event_reviewer = int(frame.duplicated(["event_id", "reviewer_id"]).sum())
    reviewer_counts = frame["reviewer_id"].value_counts().to_dict()
    event_review_counts = frame.groupby("event_id")["reviewer_id"].nunique()
    multi_review_events = int((event_review_counts >= 2).sum())
    return {
        "row_count": int(len(frame)),
        "unique_event_count": int(frame["event_id"].nunique()),
        "reviewer_counts": reviewer_counts,
        "duplicate_event_reviewer_rows": duplicate_event_reviewer,
        "multi_review_event_count": multi_review_events,
        "sentiment_counts": frame["human_sentiment_label"].value_counts().to_dict(),
        "tradeability_counts": frame["human_tradeability_label"].value_counts().to_dict(),
        "market_relevance_counts": frame["market_relevance_label"].value_counts().to_dict(),
        "agreement": _agreement(frame),
        "methodology_notes": [
            "Human labels are versioned separately from weak rule labels and market targets.",
            "Reviewer CSV must not use post-event price movement for live-time sentiment labels.",
        ],
    }


def _normalise_reviewed_frame(frame: pd.DataFrame, label_version: str) -> pd.DataFrame:
    rows = []
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        rows.append(_normalise_reviewed_row(row, row_number, label_version))
    return pd.DataFrame(rows)


def _normalise_reviewed_row(
    row: dict[str, object], row_number: int, label_version: str
) -> dict[str, object]:
    sentiment = _required_string(row, "human_sentiment_label", row_number)
    tradeability = _required_string(row, "human_tradeability_label", row_number)
    market_relevance = _required_string(row, "market_relevance_label", row_number)
    contradiction = _required_string(row, "contradiction_label", row_number)
    topics = _topics(_required_string(row, "human_topic_labels", row_number), row_number)
    confidence = _confidence(row.get("label_confidence"), row_number)
    reviewed_at = parse_utc(_required_string(row, "reviewed_at_utc", row_number))
    reviewer_id = _required_string(row, "reviewer_id", row_number)

    _ensure_allowed(sentiment, SENTIMENT_LABELS, "human_sentiment_label", row_number)
    _ensure_allowed(tradeability, TRADEABILITY_LABELS, "human_tradeability_label", row_number)
    _ensure_allowed(market_relevance, MARKET_RELEVANCE_LABELS, "market_relevance_label", row_number)
    _ensure_allowed(contradiction, CONTRADICTION_LABELS, "contradiction_label", row_number)

    return {
        "label_version": label_version,
        "event_id": _required_string(row, "event_id", row_number),
        "post_id": _required_string(row, "post_id", row_number),
        "human_sentiment_label": sentiment,
        "human_tradeability_label": tradeability,
        "human_topic_labels": topics,
        "market_relevance_label": market_relevance,
        "contradiction_label": contradiction,
        "label_confidence": confidence,
        "reviewer_id": reviewer_id,
        "reviewed_at_utc": isoformat_z(reviewed_at),
        "review_notes": str(row.get("review_notes", "")).strip(),
    }


def _topics(value: str, row_number: int) -> list[str]:
    topics = [item.strip() for item in value.split(TOPIC_SEPARATOR) if item.strip()]
    if not topics:
        raise ValueError(f"Row {row_number}: human_topic_labels cannot be empty")
    invalid = [topic for topic in topics if topic not in TOPIC_LABELS]
    if invalid:
        raise ValueError(f"Row {row_number}: invalid human_topic_labels {invalid}")
    return topics


def _confidence(value: object, row_number: int) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: label_confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Row {row_number}: label_confidence must be between 0 and 1")
    return confidence


def _required_string(row: dict[str, object], column: str, row_number: int) -> str:
    value = str(row.get(column, "")).strip()
    if not value:
        raise ValueError(f"Row {row_number}: {column} is required")
    return value


def _ensure_allowed(value: str, allowed: set[str], column: str, row_number: int) -> None:
    if value not in allowed:
        raise ValueError(f"Row {row_number}: invalid {column} {value!r}")


def _agreement(frame: pd.DataFrame) -> dict[str, Any]:
    multi = frame.groupby("event_id").filter(lambda group: group["reviewer_id"].nunique() >= 2)
    if multi.empty:
        return {"multi_review_event_count": 0}
    return {
        "multi_review_event_count": int(multi["event_id"].nunique()),
        "sentiment_unanimous_rate": _unanimous_rate(multi, "human_sentiment_label"),
        "tradeability_unanimous_rate": _unanimous_rate(multi, "human_tradeability_label"),
        "market_relevance_unanimous_rate": _unanimous_rate(multi, "market_relevance_label"),
    }


def _unanimous_rate(frame: pd.DataFrame, column: str) -> float:
    per_event = frame.groupby("event_id")[column].nunique()
    return round(float((per_event == 1).mean()), 6)


def _join_labels(value: object) -> str:
    if isinstance(value, np.ndarray):
        return TOPIC_SEPARATOR.join(str(item) for item in value.tolist())
    if isinstance(value, list):
        return TOPIC_SEPARATOR.join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)
