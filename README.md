# Trump Truth Social Sentiment Engine

Research-first sentiment and event-risk engine for testing whether Donald Trump's Truth Social posts create tradeable, non-tradeable, or volatility-only short-horizon conditions in NQ/MNQ futures.

The v1 posture is advisory and shadow-mode by default. It does not place orders and it does not claim live-trading readiness without licensed market data, a permitted live post provider, paper/shadow deployment, and out-of-sample validation.

## What Is Implemented First

- Fixture-backed source validation so the pipeline runs without paid credentials.
- Pluggable post and market-data ingestion boundaries.
- UTC-only timestamp handling and leakage-safe event target construction.
- Rule and TF-IDF baselines before any transformer work.
- Whipsaw, contradiction, stale-feed, and kill-switch semantics.
- Static dashboard output for visual verification.

## Quick Start

```bash
python -m sentiment_engine --help
python scripts/generate_fixtures.py
python -m sentiment_engine run-full --config configs/research.yaml
python -m pytest -q
```

If the package is not installed, run commands from the repo root with:

```bash
PYTHONPATH=src python -m sentiment_engine run-full --config configs/research.yaml
```

## Main Commands

```bash
PYTHONPATH=src python -m sentiment_engine ingest-posts --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine ingest-archive --config configs/research.yaml --url https://ix.cnn.io/data/truth-social/truth_archive.parquet --limit 100
PYTHONPATH=src python -m sentiment_engine ingest-market --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine build-events --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine train-classifier --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine score-whipsaw --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine tune-whipsaw --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine backtest --config configs/research.yaml
PYTHONPATH=src python -m sentiment_engine dashboard --config configs/research.yaml
python scripts/run_full_pipeline.py configs/research.yaml
```

The full run writes `reports/dashboard.html`, `reports/latest_signal.json`, classifier/event-study reports, whipsaw evaluation, and the kill-switch backtest audit.

For a plain-English status of implemented models, results, metrics, tuned parameters, Optuna, and data-cleanliness checks, see [docs/implementation_audit.md](docs/implementation_audit.md).

For the source and deep-learning readiness path, see [docs/dl_data_strategy.md](docs/dl_data_strategy.md).

## API

The FastAPI service is optional because local research and CI should not require web-server dependencies:

```bash
pip install -e ".[api]"
PYTHONPATH=src uvicorn "sentiment_engine.live.service:create_app" --factory
```

Endpoints include `/health`, `/ready`, `/signal/latest`, `/signal/{event_id}`, `/posts/ingest`, `/simulate/post`, `/metrics`, and `/ws/signals`.

## Safety Assumptions

- `live_advisory` is the default mode.
- Live flattening is disabled by default.
- Missing or stale post/market data maps to `BLOCK_NEW_ENTRIES`, not `ALLOW`.
- Research targets are kept out of live feature construction.
- Fixture results are engineering checks, not evidence of economic edge.

## Data Sources

See [docs/source_audit.md](docs/source_audit.md) for verified source status, data risks, and licensing constraints.
