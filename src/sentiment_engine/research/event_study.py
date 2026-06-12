from __future__ import annotations

from typing import Any

import pandas as pd

HORIZON_COLUMNS = ["nq_delta_5m_ticks", "nq_delta_15m_ticks", "nq_delta_30m_ticks"]


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
        "segments": _segment_summary(events),
    }


def _segment_summary(events: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for label, group in events.groupby("tradeability_label"):
        output[str(label)] = {
            "count": int(len(group)),
            "mean_30m_ticks": round(float(group["nq_delta_30m_ticks"].mean()), 4),
            "mean_range_30m_ticks": round(float(group["realised_range_30m_ticks"].mean()), 4),
        }
    return output
