from __future__ import annotations

from sentiment_engine.schemas import SignalRecord


def signal_metrics(signal: SignalRecord) -> str:
    return "\n".join(
        [
            f"sentiment_engine_whipsaw_score {signal.risk['whipsaw_score']}",
            f"sentiment_engine_feed_lag_ms {signal.data_quality['feed_lag_ms']}",
            f"sentiment_engine_risk_multiplier {signal.kill_switch['risk_multiplier']}",
        ]
    ) + "\n"
