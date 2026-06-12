from __future__ import annotations

import math

import pandas as pd

BURST_WINDOWS_MINUTES = [5, 15, 30, 60, 240]


def add_burst_features(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values("received_at_utc").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(ordered["received_at_utc"], utc=True)
    for window in BURST_WINDOWS_MINUTES:
        counts = []
        topic_entropy = []
        for idx, timestamp in enumerate(timestamps):
            start = timestamp - pd.Timedelta(minutes=window)
            mask = (timestamps < timestamp) & (timestamps >= start)
            prior = ordered.loc[mask]
            counts.append(int(len(prior)))
            topic_entropy.append(_topic_entropy(prior["rule_topic_labels"].tolist()))
        ordered[f"posts_last_{window}m"] = counts
        if window == 60:
            ordered["topic_entropy_60m"] = topic_entropy
    return ordered


def _topic_entropy(topic_lists: list[object]) -> float:
    counts: dict[str, int] = {}
    for topic_list in topic_lists:
        if not isinstance(topic_list, list):
            continue
        for topic in topic_list:
            counts[str(topic)] = counts.get(str(topic), 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(entropy, 6)
