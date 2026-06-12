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

## Safety Assumptions

- `live_advisory` is the default mode.
- Live flattening is disabled by default.
- Missing or stale post/market data maps to `BLOCK_NEW_ENTRIES`, not `ALLOW`.
- Research targets are kept out of live feature construction.
- Fixture results are engineering checks, not evidence of economic edge.

## Data Sources

See [docs/source_audit.md](docs/source_audit.md) for verified source status, data risks, and licensing constraints.
