# Source Audit

Verification date: 2026-06-12.

## Summary

The executable v1 uses local fixtures by default. That is deliberate: current and historical Truth Social post sources differ in freshness, schema, licensing, and latency, while NQ/MNQ market data requires licensed vendor or broker export for real research.

## Post Sources

| Source | Verified status | Use in this repo |
|---|---|---|
| `stiles/trump-truth-social-archive` | The README states the GitHub Actions workflow was disabled on 2025-10-26 and that the repo output will no longer be updated there. It also points to an archive updated every five minutes at `https://ix.cnn.io/data/truth-social/truth_archive.json`, with `.csv` and `.parquet` variants. | Historical/backfill adapter only. Never treated as the sole live source. |
| `kashish-s/TruthSocial_2024ElectionInitiative` | The README describes a Kaggle-hosted election dataset with posts from February 2022 through October 2024. | Research-only candidate source after licensing and schema review. Not low-latency. |
| Paid/live providers | No official stable public Truth Social trading-grade API is assumed. Provider terms, latency, schema, and permitted use must be verified before use. | Adapter interface, heartbeat checks, schema validation, and stale-feed safe mode. |
| Local fixture adapter | Deterministic records owned by this repo. | Default CI/local execution path. |

## Market Sources

| Source | Verified status | Use in this repo |
|---|---|---|
| CME product specifications | CME lists NQ as `$20 x Nasdaq-100 index` with a `0.25` index-point minimum tick. CME lists MNQ as `$2 x Nasdaq-100 index` with a `0.25` index-point minimum tick. | Instrument config validation: NQ tick value is `$5`, MNQ tick value is `$0.50`. |
| Broker/Databento/exported OHLCV | Required for serious NQ/MNQ research. Licensing, continuous-contract logic, session handling, roll policy, and timestamp quality must be checked per source. | CSV/Parquet import first. API adapters later. |
| Economic calendar CSV | Needed to reduce false attribution around CPI, FOMC, NFP, Fed speakers, auctions, and earnings windows. | Local CSV fixture/import first. |

## Data-Quality Gates

- All timestamps must be timezone-aware UTC after ingestion.
- Post target alignment uses the first market bar opening at or after `received_at_utc`.
- Market bars crossing contract roll gaps, maintenance breaks, holidays, or invalid sessions are excluded from target calculation.
- Dedupe is deterministic by `post_id` and `content_hash`.
- Feed latency is retained as `received_at_utc - created_at_utc`.
- Real-provider credentials and secrets must come from the environment, never committed YAML.

## Implementation Implication

The repo should be considered complete for v1 engineering only when the fixture path, schema validation, event construction, baseline reporting, whipsaw logic, backtest accounting, stale-feed handling, and dashboard all execute. It should not be considered live-blocking ready until real licensed data and a permitted live provider pass shadow-mode validation.
