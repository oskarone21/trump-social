from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sentiment_engine.constants import (
    MNQ_SYMBOL,
    MNQ_TICK_VALUE_USD,
    NQ_MNQ_TICK_SIZE,
    NQ_SYMBOL,
    NQ_TICK_VALUE_USD,
    UTC,
)


class InstrumentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_symbol: str = NQ_SYMBOL
    execution_symbol: str = MNQ_SYMBOL
    tick_size: float = NQ_MNQ_TICK_SIZE
    nq_tick_value_usd: float = NQ_TICK_VALUE_USD
    mnq_tick_value_usd: float = MNQ_TICK_VALUE_USD

    @field_validator("signal_symbol")
    @classmethod
    def validate_signal_symbol(cls, value: str) -> str:
        if value != NQ_SYMBOL:
            raise ValueError("v1 signal instrument must be NQ")
        return value

    @field_validator("execution_symbol")
    @classmethod
    def validate_execution_symbol(cls, value: str) -> str:
        if value != MNQ_SYMBOL:
            raise ValueError("v1 execution/account simulation instrument must be MNQ")
        return value

    @field_validator("tick_size")
    @classmethod
    def validate_tick_size(cls, value: float) -> float:
        if value != NQ_MNQ_TICK_SIZE:
            raise ValueError("NQ/MNQ tick size must match CME product spec: 0.25")
        return value


class WindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impact_horizons_minutes: list[int] = Field(default_factory=lambda: [5, 15, 30])
    contradiction_window_minutes: int
    whipsaw_evaluation_window_minutes: int
    macro_blackout_before_minutes: int
    macro_blackout_after_minutes: int


class ThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_direction_confidence: float
    min_topic_confidence: float
    soft_whipsaw_threshold: float
    hard_whipsaw_threshold: float
    stale_feed_seconds: int
    stale_market_data_seconds: int
    whipsaw_initial_move_ticks: int
    whipsaw_reversal_ticks: int


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posts_fixture: Path
    market_fixture: Path
    macro_calendar_fixture: Path
    trades_fixture: Path
    processed_dir: Path
    report_dir: Path
    model_dir: Path


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str
    timezone_internal: str = UTC
    seed: int = 42

    @field_validator("timezone_internal")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        if value != UTC:
            raise ValueError("Internal timezone must be UTC")
        return value


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig
    paths: PathConfig
    sources: dict[str, Any]
    instruments: InstrumentConfig
    windows: WindowConfig
    thresholds: ThresholdConfig
    live_actions: dict[str, Any]
    backtest: dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return data


def load_config(path: str | Path) -> EngineConfig:
    config_path = Path(path)
    raw = load_yaml(config_path)
    if "extends" in raw:
        parent = load_yaml(raw["extends"])
        raw = _deep_merge(parent, {key: value for key, value in raw.items() if key != "extends"})
    return EngineConfig.model_validate(raw)


def ensure_output_dirs(config: EngineConfig) -> None:
    for directory in (config.paths.processed_dir, config.paths.report_dir, config.paths.model_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
