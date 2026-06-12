from __future__ import annotations

from dataclasses import dataclass, field

from sentiment_engine.schemas import SignalRecord


@dataclass
class SignalState:
    latest: SignalRecord | None = None
    by_event_id: dict[str, SignalRecord] = field(default_factory=dict)

    def upsert(self, signal: SignalRecord) -> None:
        self.latest = signal
        self.by_event_id[signal.event_id] = signal

    def get(self, event_id: str) -> SignalRecord | None:
        return self.by_event_id.get(event_id)
