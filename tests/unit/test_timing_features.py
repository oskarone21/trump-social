from __future__ import annotations

import pandas as pd

from sentiment_engine.features.burst_features import add_burst_features
from sentiment_engine.models.timing import add_empirical_timing_features


def test_burst_and_timing_features_accept_mixed_iso_timestamp_precision() -> None:
    events = pd.DataFrame(
        [
            {
                "received_at_utc": "2022-07-17T19:22:13.123000Z",
                "rule_topic_labels": ["trade_policy"],
            },
            {
                "received_at_utc": "2022-07-17T19:22:14Z",
                "rule_topic_labels": ["china_tariffs"],
            },
        ]
    )

    burst = add_burst_features(events)
    timed = add_empirical_timing_features(burst)

    assert timed["posts_last_5m"].tolist() == [0, 1]
    assert "actual_minutes_to_next_post" not in timed.columns
