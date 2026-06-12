from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.config import load_config
from sentiment_engine.live.signal_engine import latest_signal_from_scores, signal_from_scored_event
from sentiment_engine.live.state import SignalState


def create_app(config_path: str = "configs/live.yaml"):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is not installed. Install with `pip install -e .[api]`.") from exc

    class SimulatePostRequest(BaseModel):
        text: str

    config = load_config(config_path)
    state = SignalState()
    scored_path = config.paths.processed_dir / "whipsaw_scores.parquet"
    if Path(scored_path).exists():
        scored_events = pd.read_parquet(scored_path)
        for row in scored_events.to_dict("records"):
            state.upsert(signal_from_scored_event(row, config))

    app = FastAPI(title="Truth Social NQ Sentiment Engine", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"ready": state.latest is not None, "latest_event_id": state.latest.event_id if state.latest else None}

    @app.get("/signal/latest")
    def latest() -> dict[str, Any]:
        if state.latest is None:
            raise HTTPException(status_code=404, detail="No signal available")
        return state.latest.model_dump(mode="json")

    @app.get("/signal/{event_id}")
    def by_event_id(event_id: str) -> dict[str, Any]:
        signal = state.get(event_id)
        if signal is None:
            raise HTTPException(status_code=404, detail="Signal not found")
        return signal.model_dump(mode="json")

    @app.post("/posts/ingest")
    def ingest_post(payload: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": False, "reason": "fixture service is read-only; use provider adapters for ingestion"}

    @app.post("/simulate/post")
    def simulate_post(payload: SimulatePostRequest) -> dict[str, Any]:
        if state.latest is None:
            raise HTTPException(status_code=404, detail="No fixture signal available")
        simulated = state.latest.model_copy(update={"text_clean": payload.text})
        return simulated.model_dump(mode="json")

    @app.get("/metrics")
    def metrics() -> str:
        latest_score = state.latest.risk["whipsaw_score"] if state.latest else 0.0
        return f"sentiment_engine_latest_whipsaw_score {latest_score}\n"

    @app.websocket("/ws/signals")
    async def websocket_signals(websocket):
        await websocket.accept()
        if state.latest is not None:
            await websocket.send_json(state.latest.model_dump(mode="json"))
        await websocket.close()

    return app


def load_latest_signal(config_path: str = "configs/research.yaml"):
    config = load_config(config_path)
    scored = pd.read_parquet(config.paths.processed_dir / "whipsaw_scores.parquet")
    return latest_signal_from_scores(scored, config)
