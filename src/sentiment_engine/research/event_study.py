from __future__ import annotations

from typing import Any

import pandas as pd

HORIZON_COLUMNS = ["nq_delta_5m_ticks", "nq_delta_15m_ticks", "nq_delta_30m_ticks"]
TARGET_PREFIXES = [
    "nq_delta",
    "realised_range",
    "realised_volatility",
    "max_favourable_excursion",
    "max_adverse_excursion",
]


def build_event_study_report(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"event_count": 0, "horizons": {}, "segments": {}}
    horizons = {}
    for column in HORIZON_COLUMNS:
        series = events[column].astype(float)
        horizons[column] = {
            "count": int(series.count()),
            "mean_ticks": round(float(series.mean()), 4),
            "median_ticks": round(float(series.median()), 4),
            "std_ticks": round(float(series.std(ddof=0)), 4),
            "p10_ticks": round(float(series.quantile(0.10)), 4),
            "p90_ticks": round(float(series.quantile(0.90)), 4),
        }
    return {
        "event_count": int(len(events)),
        "market_whipsaw_count": int(events["market_whipsaw_flag"].sum()),
        "tradeability_counts": events["tradeability_label"].value_counts().to_dict(),
        "horizons": horizons,
        "target_summary": _target_summary(events),
        "cluster_summary": _cluster_summary(events),
        "segments": _segment_summary(events),
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


def _summary_series(series: pd.Series) -> dict[str, Any]:
    return {
        "count": int(series.count()),
        "mean_ticks": round(float(series.mean()), 4),
        "median_ticks": round(float(series.median()), 4),
        "p10_ticks": round(float(series.quantile(0.10)), 4),
        "p90_ticks": round(float(series.quantile(0.90)), 4),
    }
