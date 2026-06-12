from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentiment_engine.utils.time import parse_utc

PostType = Literal["original", "reply", "retruth", "quote", "deleted", "edited", "unknown"]
Direction = Literal["up", "down", "flat"]
TradeabilityLabel = Literal[
    "tradeable_directional", "volatility_only", "no_trade_whipsaw", "no_impact", "ambiguous"
]
SentimentLabel = Literal[
    "bullish_market",
    "bearish_market",
    "geopolitical_risk",
    "neutral",
    "volatility_only",
    "low_confidence",
]
DirectionSignal = Literal["BULLISH", "BEARISH", "NEUTRAL", "NO_TRADE"]
WhipsawRiskLevel = Literal["NONE", "WATCH", "SOFT_RISK", "HARD_KILL"]
KillSwitchAction = Literal[
    "ALLOW", "BLOCK_NEW_ENTRIES", "REDUCE_SIZE", "FLATTEN_OPTIONAL", "HARD_FLAT"
]


class PostRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_provider: str
    post_id: str
    author_id: str
    created_at_utc: datetime
    received_at_utc: datetime
    ingested_at_utc: datetime
    text_raw: str
    text_clean: str
    language: str = "en"
    post_type: PostType = "unknown"
    parent_post_id: str | None = None
    quoted_post_id: str | None = None
    urls: list[str] = Field(default_factory=list)
    media_urls: list[str] = Field(default_factory=list)
    has_image: bool = False
    has_video: bool = False
    engagement_metrics_json: dict[str, Any] = Field(default_factory=dict)
    content_hash: str
    raw_json: dict[str, Any]

    @field_validator("created_at_utc", "received_at_utc", "ingested_at_utc", mode="before")
    @classmethod
    def validate_datetime(cls, value: Any) -> datetime:
        return parse_utc(value)

    @model_validator(mode="after")
    def validate_feed_order(self) -> "PostRecord":
        if self.received_at_utc < self.created_at_utc:
            raise ValueError("received_at_utc cannot be earlier than created_at_utc")
        return self


class MarketBar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol_root: Literal["NQ", "MNQ"]
    contract_symbol: str
    continuous_symbol: str
    ts_open_utc: datetime
    ts_close_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int | None = None
    vwap: float | None = None
    source_name: str
    is_rth: bool
    session_id: str
    is_rollover_period: bool
    is_holiday_session: bool
    is_valid_bar: bool

    @field_validator("ts_open_utc", "ts_close_utc", mode="before")
    @classmethod
    def validate_datetime(cls, value: Any) -> datetime:
        return parse_utc(value)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "MarketBar":
        if self.ts_close_utc <= self.ts_open_utc:
            raise ValueError("ts_close_utc must be later than ts_open_utc")
        if not self.is_valid_bar:
            return self
        if self.high < max(self.open, self.close):
            raise ValueError("high cannot be below open/close")
        if self.low > min(self.open, self.close):
            raise ValueError("low cannot be above open/close")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        return self


class EventTargetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    post_id: str
    received_at_utc: datetime
    aligned_bar_ts_utc: datetime
    base_price: float
    nq_delta_5m_ticks: float
    nq_delta_15m_ticks: float
    nq_delta_30m_ticks: float
    nq_direction_5m: Direction
    nq_direction_15m: Direction
    nq_direction_30m: Direction
    max_favourable_excursion_5m_ticks: float
    max_favourable_excursion_15m_ticks: float
    max_favourable_excursion_30m_ticks: float
    max_adverse_excursion_5m_ticks: float
    max_adverse_excursion_15m_ticks: float
    max_adverse_excursion_30m_ticks: float
    realised_range_5m_ticks: float
    realised_range_15m_ticks: float
    realised_range_30m_ticks: float
    realised_volatility_5m_ticks: float
    realised_volatility_15m_ticks: float
    realised_volatility_30m_ticks: float
    is_macro_blackout: bool
    nearest_macro_event_id: str | None = None
    nearest_macro_event_type: str | None = None
    nearest_macro_event_importance: str | None = None
    minutes_to_nearest_macro_event: float | None = None
    market_whipsaw_flag: bool
    tradeability_label: TradeabilityLabel

    @field_validator("received_at_utc", "aligned_bar_ts_utc", mode="before")
    @classmethod
    def validate_datetime(cls, value: Any) -> datetime:
        return parse_utc(value)


class SignalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    post_id: str
    source_provider: str
    created_at_utc: datetime
    received_at_utc: datetime
    generated_at_utc: datetime
    text_clean: str
    sentiment_label: SentimentLabel
    sentiment_confidence: float
    topic_labels: list[str]
    topic_confidence: dict[str, float]
    tradeability_label: TradeabilityLabel
    direction_signal: DirectionSignal
    p_direction: dict[str, float]
    expected_delta_ticks: dict[str, float]
    risk: dict[str, Any]
    kill_switch: dict[str, Any]
    data_quality: dict[str, Any]
    model_versions: dict[str, str]
    explanation: dict[str, Any]

    @field_validator("created_at_utc", "received_at_utc", "generated_at_utc", mode="before")
    @classmethod
    def validate_datetime(cls, value: Any) -> datetime:
        return parse_utc(value)
