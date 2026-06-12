# Source Audit

Verification date: 2026-06-12.

## Summary

The executable v1 uses local fixtures by default. That is deliberate: current and historical Truth Social post sources differ in freshness, schema, licensing, and latency, while NQ/MNQ market data requires licensed vendor or broker export for real research.

## Post Sources

| Source | Verified status | Use in this repo |
|---|---|---|
| `stiles/trump-truth-social-archive` | The README states the GitHub Actions workflow was disabled on 2025-10-26 and that the repo output will no longer be updated there. It also points to an archive updated every five minutes at `https://ix.cnn.io/data/truth-social/truth_archive.json`, with `.csv` and `.parquet` variants. | Historical/backfill adapter only. Never treated as the sole live source. |
| `socialcrawl` Truth Social API mirrors | Requires approved API credentials; typical header is `x-api-key` and endpoint shape is provider-specific. | Optional live path only through `ingest-provider-posts --api-key` or `--api-key-env`; records must pass schema validation and dedupe logic. |
| ScrapeCreators Truth Social endpoint | Publicly documented unofficial endpoint shape is `/v1/truthsocial/user/posts`, commonly with `user_id` and optional query filters. | Optional advisory source via `ingest-provider-posts` if account contract permits ingestion and terms allow use. |
| CNN `ix.cnn.io` Truth archive | Verified by local HEAD request on 2026-06-12: JSON and parquet endpoints returned HTTP 200, last modified 2026-06-12. Full parquet ingest on 2026-06-12 loaded 33,899 valid rows with 0 duplicate post IDs. | Implemented as the preferred historical/backfill adapter via `ingest-archive`. |
| `trumpstruth.org` RSS feed | Public RSS endpoint supports date filters (`start_date`, `end_date`) and has been used as an independent archival path for backfill/reference checks. | Implemented via `ingest-trumpstruth-feed`; treat as secondary source with schema drift checks. |
| `kashish-s/TruthSocial_2024ElectionInitiative` | The README describes a Kaggle-hosted election dataset with posts from February 2022 through October 2024. | Research-only candidate source after licensing and schema review. Not low-latency. |
| Paid/live providers | No official stable public Truth Social trading-grade API is assumed. Provider terms, latency, schema, and permitted use must be verified before use. | `ingest-provider-posts` normalizes generic third-party exports (JSON/CSV/Parquet) for live-test or approved-advisory use. |
| Local fixture adapter | Deterministic records owned by this repo. | Default CI/local execution path. |

## Market Sources

| Source | Verified status | Use in this repo |
|---|---|---|
| CME product specifications | CME lists NQ as `$20 x Nasdaq-100 index` with a `0.25` index-point minimum tick. CME lists MNQ as `$2 x Nasdaq-100 index` with a `0.25` index-point minimum tick. | Instrument config validation: NQ tick value is `$5`, MNQ tick value is `$0.50`. |
| Databento `GLBX.MDP3` exported or API OHLCV | Databento documents `GLBX.MDP3` as CME Globex MDP 3.0 futures/options coverage and documents OHLCV aggregate bars at 1-minute intervals with `ts_event` as the bar start timestamp. The Historical API exposes `timeseries.get_range`, and `DBNStore.to_df()` returns a pandas DataFrame. | Implemented via `ingest-market-file` for CSV/parquet exports and `download-databento-market` for licensed API access. Requires `DATABENTO_API_KEY` and permitted-use review. |
| Broker/exported OHLCV | Required fallback for serious NQ/MNQ research if Databento is not used. Licensing, continuous-contract logic, session handling, roll policy, and timestamp quality must be checked per source. | Supported when the file has generic timestamp/open/high/low/close/volume columns. API adapters later. |
| Economic calendar CSV | Needed to reduce false attribution around CPI, FOMC, NFP, Fed speakers, auctions, and earnings windows. | Local CSV fixture/import first. |

## Data-Quality Gates

- All timestamps must be timezone-aware UTC after ingestion.
- Post target alignment uses the first market bar opening at or after `received_at_utc`.
- External market files are normalized to canonical `MarketBar` rows before event construction.
- Databento API downloads are normalized to canonical `MarketBar` rows and audited before event construction.
- Market bars crossing contract roll gaps, maintenance breaks, holidays, or invalid sessions are excluded from target calculation.
- `check-provider-freshness` writes provider-level health metadata (`http_ok`, endpoint method, schema drift, local lag) to `reports/provider_posts_freshness_report.json`.
- Dedupe is deterministic by `post_id` and `content_hash`.
- `check-archive-freshness` writes HTTP metadata, local row count, duplicate post IDs, duplicate content hashes, empty text rows, media-only rows, and max post timestamp to `reports/archive_freshness_report.json`.
- Empty-text and media-only posts are counted explicitly because they are valid records but weak text-model training rows.
- Latest full CNN archive ingest counted 6,392 empty-text rows and 5,830 media-only rows; text-only models must filter or separately encode them.
- Feed latency is retained as `received_at_utc - created_at_utc`.
- Real-provider credentials and secrets must come from the environment, never committed YAML.

## Implementation Implication

The repo should be considered complete for v1 engineering only when the fixture path, schema validation, event construction, baseline reporting, whipsaw logic, backtest accounting, stale-feed handling, and dashboard all execute. It should not be considered live-blocking ready until real licensed data and a permitted live provider pass shadow-mode validation.
