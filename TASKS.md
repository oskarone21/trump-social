# Truth Social NQ/MNQ Sentiment Engine Tasks

Last updated: 2026-06-12.

This file is the implementation roadmap for taking the current fixture-backed vertical slice to a full research-grade and eventually live-advisory Truth Social prediction/sentiment engine with ML and DL models.

## Current State

Implemented:

- Local fixture pipeline for posts, NQ 1-minute bars, labels, event targets, whipsaw scoring, Optuna tuning, backtest, dashboard, and signal contract.
- CNN `ix.cnn.io` Truth Social archive adapter via `ingest-archive`.
- Real archive smoke ingest verified on 2026-06-12 with 25 recent posts.
- Full CNN archive ingest verified on 2026-06-12 with 33,899 posts.
- TF-IDF + logistic-regression baseline.
- LightGBM classifier smoke baseline when optional `ml` dependencies are installed.
- PyTorch TF-IDF MLP smoke baseline when optional `dl` dependencies are installed.
- Rule-based sentiment/topic/tradeability classifier.
- Weighted whipsaw detector and Optuna tuning report.
- Static dashboard and optional FastAPI service.
- External NQ/MNQ market-file adapter for Databento-style or generic 1-minute OHLCV exports.
- Archive-to-market event builder for processed CNN posts plus canonical market bars.

Not yet implemented:

- Licensed historical NQ/MNQ market-data file has not been supplied at the scale required for research.
- Full real event dataset cannot be produced until licensed bars covering the archive period are provided.
- Human-reviewed sentiment/topic/tradeability labels.
- Production-grade LightGBM walk-forward baseline on real archive + market data.
- FinBERT, DeBERTa/DistilBERT, sentence-transformer, cross-encoder/NLI, or LSTM models.
- Statistical interpretation of ML/DL results on real out-of-sample data.
- Live provider integration with an SLA/contract and heartbeat.

## Source Findings

### Truth Social Posts

- `stiles/trump-truth-social-archive` is useful as documentation and history, but its GitHub workflow stopped after 2025-10-26.
- The current practical backfill source is CNN's archive:
  - `https://ix.cnn.io/data/truth-social/truth_archive.json`
  - `https://ix.cnn.io/data/truth-social/truth_archive.csv`
  - `https://ix.cnn.io/data/truth-social/truth_archive.parquet`
- Local HEAD checks on 2026-06-12 returned HTTP 200 for CNN JSON and parquet endpoints.
- Direct Truth Social profile/API probes returned Cloudflare 403 challenges:
  - `https://truthsocial.com/@realDonaldTrump`
  - `https://truthsocial.com/api/v1/accounts/107780257626128497/statuses`
  - `https://truthsocial.com/api/v1/accounts/lookup?acct=realDonaldTrump`
- Conclusion: do not build the trading pipeline around direct scraping of `truthsocial.com`. Use the CNN archive for backfill and a contracted/live provider for low-latency operation.

### Market Data

- CME confirms NQ is `$20 x Nasdaq-100 index` with minimum tick `0.25`; tick value is `$5.00`.
- CME confirms MNQ is `$2 x Nasdaq-100 index` with minimum tick `0.25`; tick value is `$0.50`.
- Required research data is licensed NQ/MNQ intraday OHLCV or tick/order-book data.
- Candidate vendors:
  - Databento `GLBX.MDP3` for CME Globex data. This is the selected first integration path.
  - CME DataMine.
  - Broker export from a platform used for execution/account simulation.
- Free delayed/proxy data is acceptable only for smoke tests, not for research conclusions.

## Definition of Done

The full engine is done only when all of these are true:

- Real post data is ingested and audited from the CNN archive or a permitted provider.
- Licensed NQ/MNQ market data is ingested, audited, and aligned to post receive timestamps.
- Event targets are generated for 5m/15m/30m horizons without leakage.
- A human-reviewed labeled dataset exists.
- Naive, rules, TF-IDF, LightGBM, and DL models are trained and compared.
- Model selection uses purged chronological validation and final untouched test data.
- Results are interpreted with uncertainty, costs, opportunity cost, and regime stability.
- The dashboard shows data quality, model metrics, backtest attribution, and latest advisory signal.
- The full script runs end to end on real data, not only fixtures.
- Live blocking remains disabled until shadow-mode evidence exists.

## Phase 1: Real Post Backfill

- [x] Implement CNN archive ingestion adapter.
- [x] Support JSON/CSV/parquet local and remote sources.
- [x] Normalize records into `PostRecord`.
- [x] Count duplicates, empty-text rows, and media-only rows.
- [x] Smoke ingest live CNN parquet archive.
- [x] Ingest the full CNN archive into the configured local processed path.
- [ ] Add partitioned output by source/date.
- [ ] Add content-hash and post-id dedupe report for the full archive.
- [ ] Add archive freshness monitor using HTTP `Last-Modified`, `ETag`, and max post timestamp.
- [ ] Add source-license note and permitted-use confirmation before using for live trading.

Acceptance:

- Full archive ingest completes.
- `reports/cnn_archive_ingestion_audit.json` reports row count, date range, duplicates, empty text, media-only rows, and source metadata.
- The audit is reproducible from a pinned archive snapshot or immutable raw file.

## Phase 2: Market Data Acquisition

- [x] Select primary provider path: Databento `GLBX.MDP3` or equivalent broker export.
- [ ] Document license and permitted research/live use.
- [x] Implement local export adapter in `src/sentiment_engine/ingestion/market_files.py`.
- [x] Support NQ and MNQ 1-minute OHLCV first.
- [ ] Add optional tick/order-book support for spread, slippage, and latency research.
- [ ] Implement active-contract selection and continuous-contract mapping.
- [ ] Store raw contract bars and adjusted continuous bars separately.
- [ ] Implement CME session calendar, holidays, DST, maintenance breaks, and rollover flags.
- [ ] Add market-data audit:
  - [ ] expected minute count
  - [x] valid minute count
  - [ ] missing bar count
  - [ ] stale/zero-volume bars
  - [x] zero-volume bars
  - [x] duplicate timestamps
  - [x] invalid OHLC rows
  - [ ] rollover periods
  - [ ] holiday/short sessions

Acceptance:

- At least 2022-present NQ 1-minute bars are ingested.
- Coverage is at least 99% for valid sessions.
- All timestamps are timezone-aware UTC.
- No event target crosses a roll discontinuity unless explicitly adjusted.

## Phase 3: Event Dataset

- [ ] Join all Trump posts to first valid NQ bar at or after `received_at_utc`.
- [x] Implement archive-to-market event builder for processed posts and canonical bars.
- [ ] Retain feed latency.
- [ ] Generate 5m/15m/30m targets.
- [ ] Generate MFE/MAE/range/realized-volatility targets.
- [ ] Generate market-whipsaw labels.
- [ ] Implement event clustering and de-overlap.
- [ ] Build isolated-post and burst-post subsets.
- [ ] Add macro-event blackout flags.
- [ ] Add matched control windows by:
  - [ ] day of week
  - [ ] session bucket
  - [ ] volatility regime
  - [ ] pre-event trend regime
  - [ ] macro proximity
- [ ] Produce event-study report on real data.

Acceptance:

- Event dataset is reproducible from raw posts, market bars, and config.
- No live feature uses post-event data.
- Event-study report includes raw and abnormal moves, confidence intervals, and placebo windows.

## Phase 4: Labeling System

- [ ] Expand `src/sentiment_engine/labels/label_guidelines.md`.
- [ ] Build labeling queue for human review.
- [ ] Add label schema:
  - [ ] market sentiment
  - [ ] topics
  - [ ] tradeability
  - [ ] contradiction/stance pair labels
  - [ ] market relevance
  - [ ] confidence/uncertainty
- [ ] Add adjudication workflow with reviewer IDs and timestamps.
- [ ] Track inter-annotator agreement.
- [ ] Seed at least 500 human-reviewed events before transformer fine-tuning.
- [ ] Preserve weak labels separately from human labels.

Acceptance:

- Label audit report exists.
- Human labels are versioned.
- Weak labels are not mixed with final validation labels.

## Phase 5: Baselines Before DL

- [x] Naive majority baseline.
- [x] Rule baseline.
- [x] TF-IDF + logistic regression baseline.
- [x] LightGBM fixture smoke baseline with leakage-safe text/context features.
- [ ] TF-IDF + linear SVM baseline.
- [ ] LightGBM classifier on real text + market context features.
- [ ] LightGBM regressor/quantile model for move size and range.
- [ ] Calibrated probability outputs.
- [ ] Abstention thresholds.
- [ ] SHAP or permutation feature importance.

Metrics:

- Accuracy.
- Macro F1.
- Per-class precision/recall/F1.
- Confusion matrix.
- Expected calibration error.
- Brier score.
- Actionable-signal precision after abstention.
- Whipsaw recall for `SOFT_RISK`.
- Whipsaw precision for `HARD_KILL`.

Acceptance:

- Baselines use chronological train/validation/test splits.
- No random headline split is used.
- Final test set remains untouched during model selection.

## Phase 6: Deep-Learning Models

- [ ] Add optional `dl` dependency group:
  - [x] `torch`
  - [x] `transformers`
  - [x] `datasets`
  - [x] `evaluate`
  - [x] `sentence-transformers`
  - [x] `accelerate`
- [x] Implement PyTorch TF-IDF MLP fixture smoke baseline and report.
- [ ] Implement FinBERT inference baseline.
- [ ] Implement DistilBERT/DeBERTa classifier fine-tuning.
- [ ] Implement sentence-transformer embeddings for clustering/retrieval.
- [ ] Implement cross-encoder/NLI contradiction scorer.
- [ ] Add model registry:
  - [ ] model version
  - [ ] data hash
  - [ ] label version
  - [ ] config hash
  - [ ] training seed
  - [ ] train/validation/test windows
- [ ] Add calibration layer.
- [ ] Add confidence-based abstention.
- [ ] Compare DL against naive/rules/TF-IDF/LightGBM.

Acceptance:

- DL model is trained only on human-reviewed or explicitly weak-supervised labels.
- DL beats TF-IDF and LightGBM out of sample before promotion.
- Calibration improves actionable precision.
- Model artefacts are reproducible.

## Phase 7: Hyperparameter Tuning

- [x] Add Optuna tuning for whipsaw weights and thresholds.
- [ ] Add Optuna tuning for TF-IDF/logistic-regression hyperparameters.
- [ ] Add Optuna tuning for LightGBM.
- [ ] Add Optuna pruning for DL fine-tuning only after dataset size justifies it.
- [ ] Add parameter stability report across walk-forward folds.

Do not tune on the final test period.

Acceptance:

- Every tuning run records objective, trials, seed, parameter ranges, best params, train metrics, validation metrics, and holdout metrics.
- Tuned params are not automatically promoted without stability checks.

## Phase 8: Backtesting and Interpretation

- [x] Fixture kill-switch overlay backtest with costs and latency.
- [ ] Real baseline strategy overlay.
- [ ] Event-level audit log for real data.
- [ ] Cost sensitivity grid.
- [ ] Slippage sensitivity grid.
- [ ] Latency sensitivity grid.
- [ ] Opportunity-cost analysis.
- [ ] False-positive and false-negative dollar attribution.
- [ ] Monthly/quarterly stability table.
- [ ] Regime segmentation:
  - [ ] RTH open
  - [ ] RTH midday
  - [ ] RTH close
  - [ ] overnight Asia
  - [ ] overnight Europe
  - [ ] macro blackout
  - [ ] high/low volatility
- [ ] Bootstrap confidence intervals.
- [ ] Minimum sample-size gates.
- [ ] Multiple-testing/data-mining risk note.

Acceptance:

- Interpretation report answers:
  - [ ] Does the signal reduce drawdown?
  - [ ] Does it reduce MAE?
  - [ ] Does it improve net expectancy after costs?
  - [ ] What does it miss?
  - [ ] What does it falsely block?
  - [ ] Is the effect stable across time?
  - [ ] Is the result robust to costs/latency?

## Phase 9: Live Provider and Monitoring

- [ ] Select live provider:
  - [ ] paid Truth Social provider
  - [ ] CNN archive polling for low-frequency advisory only
  - [ ] manually supplied fallback
- [ ] Verify provider terms.
- [ ] Implement heartbeat.
- [ ] Implement stale-feed detection.
- [ ] Implement schema-drift detection.
- [ ] Implement provider failover.
- [ ] Add Redis/WebSocket output if needed.
- [ ] Add Prometheus metrics.
- [ ] Add alerting for:
  - [ ] stale post feed
  - [ ] stale market data
  - [ ] model load failure
  - [ ] schema mismatch
  - [ ] latency breach

Acceptance:

- Stale feed always produces conservative `BLOCK_NEW_ENTRIES`.
- Live flattening remains disabled by default.
- Latest signal can be reconstructed from raw input, features, model versions, and thresholds.

## Phase 10: Dashboard and Result Interpretation

- [x] Static dashboard for fixture verification.
- [ ] Add real-data dashboard sections:
  - [ ] data coverage
  - [ ] archive freshness
  - [ ] label coverage
  - [ ] classifier metrics
  - [ ] DL metrics
  - [ ] calibration curves
  - [ ] event-study distributions
  - [ ] backtest before/after
  - [ ] missed winners
  - [ ] avoided losers
  - [ ] regime stability
- [ ] Add model comparison table.
- [ ] Add event drilldown.
- [ ] Add signal explanation panel.

Acceptance:

- A non-technical reviewer can see what the model predicted, why, and whether it helped or hurt after costs.

## Phase 11: Scripts and CI

- [x] `scripts/run_full_pipeline.py` for fixture path.
- [ ] `scripts/run_archive_backfill.py`.
- [x] `scripts/run_real_event_build.py`.
- [ ] `scripts/run_model_training.py`.
- [ ] `scripts/run_walk_forward.py`.
- [ ] `scripts/run_shadow_report.py`.
- [ ] Add CI test matrix:
  - [ ] unit tests
  - [ ] integration fixture pipeline
  - [ ] schema validation
  - [ ] no-leakage tests
  - [ ] dashboard smoke

Acceptance:

- A single command can run fixture CI.
- A separate explicit command can run real-data research after credentials/data paths are configured.

## Immediate Next Implementation Tasks

1. Ingest the full CNN archive to a raw immutable local snapshot.
2. Add full archive audit and freshness monitor.
3. Supply licensed NQ/MNQ 1-minute bars covering the archive period.
4. Run `ingest-market-file` and `build-archive-events` on that licensed data.
5. Generate real event targets.
6. Produce first real event-study report.
7. Add label-review workflow.
8. Train LightGBM baseline on real event data.
9. Add FinBERT inference baseline.
10. Fine-tune DeBERTa/DistilBERT only after human labels exist.

## Current Verification Commands

```bash
PYTHONPATH=src python -m sentiment_engine --config configs/research.yaml ingest-archive --url https://ix.cnn.io/data/truth-social/truth_archive.parquet --limit 25
PYTHONPATH=src python -m sentiment_engine --config configs/research.yaml ingest-market-file --input data/fixtures/nq_1m_sample.csv --source-name fixture_canonical --symbol-root NQ --out data/processed/external_market_bars.parquet
python scripts/run_real_event_build.py --posts data/processed/posts.parquet --market data/processed/external_market_bars.parquet --out data/processed/real_events.parquet --limit-posts 8
python scripts/run_full_pipeline.py configs/research.yaml
PYTHONPATH=src python -m pytest -q
```

Latest known verification:

- Archive smoke ingest: 25 rows, 25 valid, 0 duplicates, 4 empty-text/media-only.
- Full archive ingest: 33,899 rows, 33,899 valid, 0 duplicates, 6,392 empty-text rows, 5,830 media-only rows.
- Market-file ingest path: 340 canonical fixture bars ingested from CSV.
- Archive event builder path: 8 events, 0 skipped posts using processed posts plus canonical bars.
- Full fixture pipeline: completed.
- Tests: 12 passed.
- Browser dashboard check via localhost: title rendered, 10 metrics, 8 event rows.
