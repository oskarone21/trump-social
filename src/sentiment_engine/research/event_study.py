from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

HORIZON_COLUMNS = ["nq_delta_5m_ticks", "nq_delta_15m_ticks", "nq_delta_30m_ticks"]
TARGET_PREFIXES = [
    "nq_delta",
    "realised_range",
    "realised_volatility",
    "max_favourable_excursion",
    "max_adverse_excursion",
]
CONFIDENCE_Z_95 = 1.96
FLAT_MOVE_THRESHOLD_TICKS = 4.0


def build_event_study_report(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"event_count": 0, "horizons": {}, "segments": {}}
    horizons = {}
    for column in HORIZON_COLUMNS:
        horizons[column] = _directional_summary(events[column].astype(float))
    return {
        "event_count": int(len(events)),
        "market_whipsaw_count": int(events["market_whipsaw_flag"].sum()),
        "tradeability_counts": events["tradeability_label"].value_counts().to_dict(),
        "horizons": horizons,
        "target_summary": _target_summary(events),
        "cluster_summary": _cluster_summary(events),
        "macro_summary": _macro_summary(events),
        "segments": _segment_summary(events),
        "rule_sentiment_segments": _categorical_segment_summary(
            events, "rule_sentiment_label"
        ),
        "rule_topic_segments": _topic_segment_summary(events),
    }


def _target_summary(events: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for prefix in TARGET_PREFIXES:
        columns = [column for column in events.columns if column.startswith(f"{prefix}_")]
        output[prefix] = {
            column: _summary_series(events[column].astype(float)) for column in sorted(columns)
        }
    return output


def _segment_summary(events: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for label, group in events.groupby("tradeability_label"):
        output[str(label)] = {
            "count": int(len(group)),
            "mean_30m_ticks": round(float(group["nq_delta_30m_ticks"].mean()), 4),
            "mean_range_30m_ticks": round(float(group["realised_range_30m_ticks"].mean()), 4),
            "mean_realised_volatility_30m_ticks": round(
                float(group["realised_volatility_30m_ticks"].mean()), 4
            ),
        }
    return output


def _categorical_segment_summary(events: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in events:
        return {}
    output = {}
    for label, group in events.dropna(subset=[column]).groupby(column):
        label_text = str(label)
        if not label_text:
            continue
        output[label_text] = _horizon_segment_summary(group)
    return output


def _topic_segment_summary(events: pd.DataFrame) -> dict[str, Any]:
    if "rule_topic_labels" not in events:
        return {}
    rows = []
    for row in events.to_dict("records"):
        for topic in _topic_labels(row.get("rule_topic_labels")):
            item = {column: row[column] for column in HORIZON_COLUMNS}
            item["topic"] = topic
            rows.append(item)
    if not rows:
        return {}
    exploded = pd.DataFrame(rows)
    return {
        str(topic): _horizon_segment_summary(group)
        for topic, group in exploded.groupby("topic")
        if str(topic)
    }


def _horizon_segment_summary(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "count": int(len(group)),
        "horizons": {
            column: _directional_summary(group[column].astype(float))
            for column in HORIZON_COLUMNS
        },
    }


def _cluster_summary(events: pd.DataFrame) -> dict[str, Any]:
    if "event_cluster_id" not in events:
        return {}
    cluster_sizes = events.groupby("event_cluster_id")["event_id"].count()
    return {
        "cluster_count": int(cluster_sizes.count()),
        "isolated_event_count": int(events["is_isolated_event"].sum()),
        "burst_event_count": int(events["is_burst_event"].sum()),
        "max_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "cluster_size_counts": {
            str(size): int(count)
            for size, count in cluster_sizes.value_counts().sort_index().items()
        },
    }


def _macro_summary(events: pd.DataFrame) -> dict[str, Any]:
    if "is_macro_blackout" not in events:
        return {}
    blackout = events[events["is_macro_blackout"]]
    return {
        "macro_blackout_event_count": int(events["is_macro_blackout"].sum()),
        "nearest_macro_event_type_counts": events["nearest_macro_event_type"]
        .fillna("none")
        .value_counts()
        .to_dict(),
        "blackout_event_type_counts": blackout["nearest_macro_event_type"]
        .fillna("none")
        .value_counts()
        .to_dict(),
    }


def _summary_series(series: pd.Series) -> dict[str, Any]:
    series = series.dropna()
    return {
        "count": int(series.count()),
        "mean_ticks": round(float(series.mean()), 4),
        "median_ticks": round(float(series.median()), 4),
        "std_ticks": round(float(series.std(ddof=0)), 4),
        **_mean_confidence_interval(series),
        "p10_ticks": round(float(series.quantile(0.10)), 4),
        "p90_ticks": round(float(series.quantile(0.90)), 4),
    }


def _directional_summary(series: pd.Series) -> dict[str, Any]:
    series = series.dropna()
    summary = _summary_series(series)
    if series.empty:
        summary.update(
            {
                "positive_rate": 0.0,
                "negative_rate": 0.0,
                "flat_rate": 0.0,
                "directional_interpretation": "insufficient_data",
            }
        )
        return summary
    summary.update(
        {
            "positive_rate": round(float((series > FLAT_MOVE_THRESHOLD_TICKS).mean()), 6),
            "negative_rate": round(float((series < -FLAT_MOVE_THRESHOLD_TICKS).mean()), 6),
            "flat_rate": round(
                float(series.between(-FLAT_MOVE_THRESHOLD_TICKS, FLAT_MOVE_THRESHOLD_TICKS).mean()),
                6,
            ),
            "directional_interpretation": _directional_interpretation(summary),
        }
    )
    return summary


def _mean_confidence_interval(series: pd.Series) -> dict[str, Any]:
    count = int(series.count())
    if count == 0:
        return {
            "standard_error_ticks": None,
            "mean_ci_95_low_ticks": None,
            "mean_ci_95_high_ticks": None,
        }
    if count == 1:
        mean = round(float(series.mean()), 4)
        return {
            "standard_error_ticks": 0.0,
            "mean_ci_95_low_ticks": mean,
            "mean_ci_95_high_ticks": mean,
        }
    standard_error = float(series.std(ddof=1) / np.sqrt(count))
    mean = float(series.mean())
    margin = CONFIDENCE_Z_95 * standard_error
    return {
        "standard_error_ticks": round(standard_error, 4),
        "mean_ci_95_low_ticks": round(mean - margin, 4),
        "mean_ci_95_high_ticks": round(mean + margin, 4),
    }


def _directional_interpretation(summary: dict[str, Any]) -> str:
    low = summary.get("mean_ci_95_low_ticks")
    high = summary.get("mean_ci_95_high_ticks")
    mean = float(summary["mean_ticks"])
    if low is None or high is None:
        return "insufficient_data"
    if low > 0:
        return "positive_mean_move"
    if high < 0:
        return "negative_mean_move"
    if abs(mean) <= FLAT_MOVE_THRESHOLD_TICKS:
        return "no_clear_direction"
    return "directionally_uncertain"


def _topic_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    return []
