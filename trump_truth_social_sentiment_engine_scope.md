# Trump Truth Social Sentiment Engine — Implementation Scope for AI Coding Agent

## 1. Project Aim

Build a production-grade research and live-inference system that detects whether Donald Trump's Truth Social posts create tradeable or non-tradeable short-term market conditions for NASDAQ futures (`NQ` / `MNQ`).

The primary trading use case is **risk filtering and whipsaw avoidance**, not standalone automated trade entry. The system should classify posts, estimate short-horizon market impact, detect contradictory posting behaviour, and emit a low-latency signal that can be consumed by a futures trading platform or execution layer.

The v1 output must answer:

1. Is this post likely relevant to NASDAQ futures price action?
2. Is the expected immediate impact bullish, bearish, neutral, or volatility-only?
3. Is this post part of a contradictory burst that makes the market non-tradeable?
4. Should the trading system allow entries, block new entries, reduce size, or trigger a configurable kill-switch?
5. Can the system demonstrate backtested value after slippage, commissions, latency, and account-rule constraints?

---

## 2. Reviewed Plan — Bottlenecks, Improvements, and Trading-Focused Changes

### 2.1 Highest-Risk Bottlenecks

| Bottleneck | Why It Matters | Required Mitigation |
|---|---|---|
| Historical source staleness | The proposed `stiles/trump-truth-social-archive` repository is useful historically but no longer suitable as the sole ongoing source because its workflow was disabled after 26 Oct 2025. | Implement pluggable ingestion adapters: historical archive, manual/backfill archive, paid live provider, and local fixture adapter. Treat data acquisition as a first-class risk. |
| No official Truth Social API dependency | Live access may rely on third-party scrapers/APIs, which can break, lag, change schema, or violate provider constraints. | Define a provider interface, heartbeat checks, failover, schema validation, rate-limit handling, and stale-feed no-trade behaviour. |
| Sparse labelled data | Trump posts are limited in number, and truly market-moving posts are rarer. A large transformer fine-tune can overfit quickly. | Start with strong baselines, weak labels, human adjudication, active learning, temporal validation, and calibrated confidence thresholds. |
| Causality and confounding | Posts may coincide with CPI, FOMC, jobs data, earnings, geopolitical headlines, or market open/close volatility. | Use event-study controls, matched non-event windows, macro-news blackout flags, placebo tests, and walk-forward validation. |
| Overlapping event windows | Multiple posts in 5–30 minutes contaminate target labels and make attribution ambiguous. | Implement event clustering, de-overlap logic, and separate targets for isolated posts versus post bursts. |
| Directional accuracy alone is weak | A 58% hit rate can still lose money after spread, slippage, latency, and adverse selection. | Add expectancy, drawdown reduction, MAE reduction, no-trade opportunity cost, calibration, and account-rule survival metrics. |
| Whipsaw definition is underspecified | Textual contradiction is not identical to market whipsaw. A contradictory post may not move NQ, and a whipsaw can happen without explicit contradiction. | Define both `TEXT_CONTRADICTION` and `MARKET_WHIPSAW`. The kill-switch should be driven by a combined text + market-path risk score. |
| Precision/recall objective conflict | Original plan says false positives are low cost but sets precision as the main whipsaw objective. If false negatives are expensive, recall and dollar-loss capture matter more. | Use two-tier gating: high-recall `SOFT_RISK` and higher-precision `HARD_KILL`. Optimise cost-weighted trading utility, not F1 alone. |
| Feature leakage risk | VIX, volume profile, realised volatility, or market bars may accidentally include information after the post timestamp. | Enforce point-in-time feature generation. All features must be available at or before `post_received_at_utc`. |
| Contract rollover and sessions | NQ/MNQ futures require correct contract mapping, session handling, holidays, maintenance breaks, and DST handling. | Use UTC internally, store source timezone, implement continuous contract logic, and exclude invalid bars. |
| Live execution ambiguity | “Kill-switch” can mean block new entries, flatten existing positions, reduce size, or suppress discretionary alerts. | Emit explicit action semantics: `ALLOW`, `BLOCK_NEW_ENTRIES`, `REDUCE_SIZE`, `FLATTEN_OPTIONAL`, `HARD_FLAT`. Default to advisory/paper mode. |

### 2.2 Improvements to the Original Design

1. **Prioritise whipsaw and tail-risk filtering before directional alpha.** The literature and trading hypothesis both point more strongly to jump volatility than stable drift. The first profitable use case is likely avoiding bad fills and reducing adverse excursion.
2. **Add a tradeability classifier.** Classify each event as `TRADEABLE_DIRECTIONAL`, `VOLATILITY_ONLY`, `NO_TRADE_WHIPSAW`, or `NO_IMPACT`, not only bullish/bearish/neutral.
3. **Use market-defined whipsaw labels.** A whipsaw event should require an initial directional move followed by a meaningful reversal, measured in ticks, ATR, or pre-event volatility units.
4. **Replace pure cosine contradiction with an NLI/stance layer.** Cosine similarity alone is weak for contradiction. Use a sentence-pair model or cross-encoder for `entails`, `contradicts`, `unrelated`, plus topic/entity matching.
5. **Use purged walk-forward validation.** Random train/test splits will leak regime information and overstate performance.
6. **Add confidence-based abstention.** The system should be allowed to output `NO_SIGNAL_LOW_CONFIDENCE`; this is essential for trading robustness.
7. **Separate research-time labels from live-time features.** Research can compute post-event moves; live inference cannot. Enforce this in code with separate modules.
8. **Use dollar-value metrics.** Report tick value, commissions, slippage, avoided losses, missed gains, and account-rule breaches.
9. **Add feed reliability criteria.** A live trading filter that silently misses posts is dangerous. Missing/stale feed must trigger conservative behaviour.
10. **Use model versioning and reproducible artefacts.** Every live signal must store model version, feature version, data source, and inference timestamp.

### 2.3 Additional Trading Goals

Add these to v1 or v1.5:

- **Tail-risk alarm:** detect posts likely to expand 5–30 minute realised range even if direction is uncertain.
- **Position-sizing scalar:** output a `risk_multiplier` from `0.0` to `1.0` so the trading system can reduce size instead of only going flat.
- **Post-burst regime detector:** identify when posting frequency itself becomes a volatility signal.
- **Macro-event guardrail:** suppress attribution and/or live trading during high-impact scheduled events unless explicitly overridden.
- **Opportunity-cost measurement:** track missed favourable moves during kill-switch periods.
- **Model confidence calibration:** only allow directional signals above a calibrated probability threshold.
- **Paper-trading deployment gate:** live kill-switch integration must run in shadow mode before it can affect real positions.

---

## 3. Project Scope

### 3.1 In Scope for v1

The AI coding agent must implement:

1. Historical post ingestion from one or more archive sources.
2. Live post ingestion through a provider adapter interface.
3. NQ/MNQ 1-minute OHLCV ingestion with UTC alignment.
4. Optional contextual market-data ingestion: VIX/VXN, realised volatility, volume, session profile, economic calendar flags.
5. Event-aligned dataset builder with 5m, 15m, and 30m post-event targets.
6. NLP preprocessing pipeline for Truth Social text.
7. Sentiment/topic/tradeability classifier.
8. Contradiction and whipsaw detector.
9. Impact model for expected price delta and/or probability of directional move.
10. Backtesting module focused on kill-switch and risk-filter value.
11. FastAPI live inference service.
12. Signal output contract suitable for downstream trading-system integration.
13. Dashboard or CLI status monitor for live state, current risk flag, and feed/model health.
14. Test suite, fixtures, reproducibility controls, and clear developer documentation.

### 3.2 Out of Scope for v1

- Fully automated order placement.
- Options, equities, FX, crypto, or instruments other than NQ/MNQ.
- Long-horizon portfolio allocation.
- Sentiment analysis of all political figures.
- Unsupervised scraping that bypasses access controls.
- Guaranteed causal claims without appropriate event-study controls.
- Model training on private paid datasets unless the user supplies credentials and licensing permission.

### 3.3 Recommended v1 Product Positioning

The v1 system should be positioned as:

> A real-time political-post risk filter for NQ/MNQ futures that identifies volatility, contradiction, and whipsaw risk, and optionally emits directional signals only when confidence and market conditions are favourable.

---

## 4. Definitions

### 4.1 Core Labels

| Label | Meaning |
|---|---|
| `bullish_market` | Post is expected to support risk-on NASDAQ price action. |
| `bearish_market` | Post is expected to pressure NASDAQ price action. |
| `geopolitical_risk` | Post is expected to increase uncertainty, volatility, oil/geopolitical risk, or safe-haven behaviour. |
| `neutral` | Post has no clear market implication. |
| `volatility_only` | Post may move range/volatility but direction is uncertain. |
| `low_confidence` | Model abstains because confidence is below threshold. |

### 4.2 Topic Labels

Initial topic taxonomy:

- `trade_policy`
- `china_tariffs`
- `fed_monetary`
- `inflation_rates`
- `iran_energy`
- `middle_east_geopolitics`
- `equities_direct`
- `technology_ai_semiconductors`
- `tax_fiscal_policy`
- `domestic_political`
- `legal_personal`
- `media_attack`
- `campaign_political`
- `other`

Notes:

- Keep the original topic categories but split overly broad market-sensitive topics.
- Allow multi-label topics because one post can mention tariffs, China, and equities simultaneously.
- Store topic confidence per label.

### 4.3 Signal Labels

| Signal | Meaning |
|---|---|
| `BULLISH` | Directional long bias allowed if confidence and risk filters pass. |
| `BEARISH` | Directional short bias allowed if confidence and risk filters pass. |
| `NEUTRAL` | No directional signal. |
| `VOLATILITY_RISK` | Elevated range/jump risk; directional confidence insufficient. |
| `WHIPSAW_RISK` | Contradiction/burst/reversal risk; block or reduce trading. |
| `NO_TRADE_LOW_CONFIDENCE` | Insufficient model confidence. |
| `NO_TRADE_DATA_STALE` | Feed or market data stale/missing. |
| `NO_TRADE_MACRO_BLACKOUT` | Scheduled macro-risk window. |

### 4.4 Whipsaw Definitions

Implement two related but separate concepts.

#### Text Contradiction

A pair or cluster of posts is a text contradiction when:

1. Posts occur within `contradiction_window_minutes`, default `60`.
2. Posts share at least one market-relevant topic or entity.
3. Stance/sentiment changes direction materially.
4. An NLI/stance model or calibrated contradiction scorer exceeds threshold.

#### Market Whipsaw

An event is a market whipsaw when:

1. An initial move exceeds a threshold after the first relevant post.
2. Price then reverses by a threshold amount inside the configured window.
3. The reversal is large enough to invalidate an ordinary stop/technical setup.

Default market label proposal:

```text
initial_move = max_abs_price_change_in_first_10m
reversal = opposite_direction_move_after_initial_extreme_until_30m
market_whipsaw = initial_move >= max(20 ticks, 0.35 * pre_event_ATR_30m)
                 and reversal >= max(0.65 * initial_move, 15 ticks)
```

These defaults must be configurable and validated empirically.

---

## 5. Data Requirements

### 5.1 Source Registry

| Source | Purpose | Status / Risk | Implementation Requirement |
|---|---|---|---|
| `stiles/trump-truth-social-archive` | Historical posts | Useful for historical data, but not sufficient for live/current archive after workflow disablement. | Implement as historical adapter only. |
| Presidency/UCSB Truth Social archive | Additional historical/backfill validation | Not designed for low-latency trading. | Optional backfill adapter. |
| ScrapeCreators, TweetStream, Apify, or similar provider | Live Truth Social ingestion | Third-party dependency; schema and access may change. | Implement provider interface and failover strategy. |
| Databento / broker export | NQ/MNQ OHLCV and optional tick data | Paid data/licensing required. | Implement CSV/Parquet import first; API adapter second. |
| CBOE / paid vendor / broker feed | VIX/VXN or volatility context | May be delayed or licensed. | Optional feature source with fallback. |
| Economic calendar provider | CPI, FOMC, NFP, Fed speakers, auctions | Needed to reduce false causal attribution. | Implement simple CSV calendar import in v1. |
| Trading account configuration | Topstep rules, account size, DLL, MLL, scaling plan | Rules change; do not hardcode permanently. | Load from versioned YAML config. |

### 5.2 Data Quality Requirements

All ingested posts must store:

- `source_name`
- `source_provider`
- `post_id`
- `author_id`
- `created_at_utc`
- `received_at_utc`
- `ingested_at_utc`
- `text_raw`
- `text_clean`
- `language`
- `post_type`: `original`, `reply`, `retruth`, `quote`, `deleted`, `edited`, `unknown`
- `parent_post_id`
- `quoted_post_id`
- `urls`
- `media_urls`
- `has_image`
- `has_video`
- `engagement_metrics_json`
- `content_hash`
- `raw_json`

All market bars must store:

- `symbol_root`: `NQ` or `MNQ`
- `contract_symbol`
- `continuous_symbol`
- `ts_open_utc`
- `ts_close_utc`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `trade_count` if available
- `vwap` if available
- `source_name`
- `is_rth`
- `session_id`
- `is_rollover_period`
- `is_holiday_session`
- `is_valid_bar`

### 5.3 Time Handling Requirements

- Store all timestamps internally as timezone-aware UTC.
- Preserve original source timezone in raw metadata.
- Use CME session calendar for RTH/ETH/overnight classification.
- Handle DST transitions explicitly.
- Exclude market maintenance breaks and invalid bars from target computation.
- Align post targets to the first market bar that begins after `received_at_utc`, not merely the post's published timestamp.
- Track feed latency as `received_at_utc - created_at_utc`.

### 5.4 Contract Handling Requirements

- Use `NQ` as the primary signal instrument.
- Use `MNQ` for execution/account simulation where tick value matters.
- Implement active-contract selection by volume/open interest or user-specified roll date.
- Store both raw contract bars and continuous adjusted bars.
- Never compute intraday targets across contract roll discontinuities unless explicitly adjusted.

---

## 6. Target Variables

### 6.1 Directional Targets

For each post/event cluster:

- `nq_delta_5m_ticks`
- `nq_delta_15m_ticks`
- `nq_delta_30m_ticks`
- `nq_direction_5m`: `up`, `down`, `flat`
- `nq_direction_15m`: `up`, `down`, `flat`
- `nq_direction_30m`: `up`, `down`, `flat`

Use tick-normalised values and optionally dollar values:

```text
NQ tick size = 0.25 index points
NQ tick value = $5.00
MNQ tick value = $0.50
```

### 6.2 Risk and Tradeability Targets

- `max_favourable_excursion_30m_ticks`
- `max_adverse_excursion_30m_ticks`
- `realised_range_5m_ticks`
- `realised_range_15m_ticks`
- `realised_range_30m_ticks`
- `realised_volatility_30m`
- `jump_flag_5m`
- `market_whipsaw_flag`
- `tradeability_label`: `tradeable_directional`, `volatility_only`, `no_trade_whipsaw`, `no_impact`, `ambiguous`

### 6.3 Abnormal-Return Targets

Add abnormal move targets to separate ordinary market movement from post-linked reaction:

```text
abnormal_delta_horizon = post_event_delta_horizon - expected_delta_matched_control
```

Matched controls should use same:

- day of week
- session bucket
- volatility regime
- pre-event trend regime
- macro-event proximity bucket

---

## 7. Feature Requirements

### 7.1 Text Features

- Clean post text with HTML stripping, Unicode normalisation, whitespace normalisation, URL placeholder replacement, ticker/entity preservation, emoji preservation where useful.
- Retain all-caps intensity, exclamation count, question marks, and punctuation features because they may carry signal.
- Extract:
  - post length
  - token count
  - all-caps ratio
  - exclamation count
  - question mark count
  - URL domains
  - mentioned tickers/entities
  - geopolitical entities
  - policy keywords
  - media/link flags
  - repost/reply/original flag

### 7.2 Market Context Features

All features must be point-in-time safe.

- Session bucket: `premarket_us`, `rth_open`, `rth_midday`, `rth_close`, `postmarket`, `overnight_asia`, `overnight_europe`, `maintenance_or_closed`.
- Minutes to/from RTH open and close.
- Pre-event NQ return over 5m, 15m, 30m, 60m.
- Pre-event realised volatility over 15m, 30m, 60m.
- Pre-event range and volume percentile.
- Current spread and top-of-book imbalance if tick/order-book data is available.
- VIX/VXN level and change if point-in-time data is available.
- Volume profile location: above/below VWAP, prior high/low, prior close, overnight high/low.
- Economic calendar proximity flags:
  - `macro_event_within_15m`
  - `macro_event_within_60m`
  - `fomc_day`
  - `cpi_day`
  - `nfp_day`
  - `major_earnings_window`

### 7.3 Post Burst Features

- Number of Trump posts in last 5m, 15m, 30m, 60m, 240m.
- Number of market-relevant posts in each window.
- Topic entropy in rolling window.
- Max contradiction score in rolling window.
- Time since previous post.
- Estimated probability of another post within 15m/30m/60m.

---

## 8. Model Requirements

### 8.1 Baseline Models Required Before Transformers

The coding agent must implement simple baselines first:

1. Keyword/topic rules.
2. Logistic regression or linear SVM using TF-IDF.
3. LightGBM on engineered text + market features.
4. Naive no-trade baseline.
5. Matched-control event-study baseline.

Do not treat a transformer model as successful unless it beats these baselines in out-of-sample walk-forward tests.

### 8.2 NLP Classifier

Implement a multi-task classifier that predicts:

- sentiment/market implication label
- topic labels
- tradeability label
- confidence/calibration score

Candidate models:

- FinBERT baseline.
- DistilBERT or DeBERTa classifier if FinBERT underperforms on political/social-media language.
- Sentence-transformer embeddings for retrieval and clustering.
- NLI/cross-encoder model for contradiction scoring.

Requirements:

- Use time-based train/validation/test split.
- Store model artefacts with version and training data hash.
- Output calibrated probabilities, not only class labels.
- Support abstention when confidence is below threshold.
- Include per-class precision, recall, F1, confusion matrix, and calibration report.

### 8.3 Impact Model

Implement two impact models:

1. **Directional classifier**: predicts probability of up/down/flat move at 5m, 15m, 30m.
2. **Regression/quantile model**: predicts expected delta and move distribution.

Recommended model:

- LightGBM classifier/regressor for v1.
- Quantile regression for move distribution and tail risk.
- SHAP feature importance report.

Avoid LSTM/sequence models in v1 unless there is sufficient labelled data and a clear baseline improvement.

### 8.4 Whipsaw Detector

Implement a combined whipsaw score:

```text
whipsaw_score = weighted_sum(
    text_contradiction_score,
    same_topic_score,
    post_burst_score,
    direction_flip_score,
    volatility_regime_score,
    initial_market_reaction_score
)
```

The detector must support:

- rolling time window, default 60 minutes
- configurable topic-match threshold
- configurable contradiction threshold
- separate soft and hard thresholds
- TTL on emitted risk flag
- explanation fields showing which posts caused the flag

Output risk levels:

- `NONE`
- `WATCH`
- `SOFT_RISK`
- `HARD_KILL`

### 8.5 Post Timing Model

Start with a non-parametric empirical hazard model before implementing Hawkes:

1. Empirical time-to-next-post distribution by hour-of-day and day-of-week.
2. Rolling post intensity features.
3. Optional Hawkes process model if post clustering is strong and improves live risk filtering.

Output:

- `p_next_post_15m`
- `p_next_post_30m`
- `p_next_post_60m`
- `expected_minutes_to_next_post`

---

## 9. Backtesting and Research Requirements

### 9.1 Event Study

Implement an event-study module that reports:

- raw delta distribution by horizon
- abnormal delta distribution by horizon
- realised range distribution
- MAE/MFE distribution
- volatility/jump response
- topic/sentiment segmented results
- session segmented results
- isolated-post versus burst-post results
- pre/post placebo windows
- matched non-event controls
- confidence intervals via bootstrap

### 9.2 Validation Rules

Use:

- walk-forward time split
- purged/embargoed cross-validation for overlapping windows
- no random split as headline metric
- no using post-event data in live features
- model selection based only on validation periods
- final test period untouched until final evaluation

### 9.3 Trading Simulation

The backtester must support:

- NQ and MNQ tick values.
- Commission and exchange fees.
- Slippage assumptions in ticks.
- Latency from post creation to signal to fill.
- Spread and no-fill modelling where data allows.
- Existing strategy overlay: apply kill-switch to a baseline strategy's entries/exits.
- Account-specific risk rules loaded from YAML.
- Daily loss limit, maximum loss limit, trailing drawdown, and scaling-plan constraints as configurable rules.
- RTH-only and ETH-enabled modes.
- Macro blackout overlays.

### 9.4 Required Backtest Reports

Produce:

- signal attribution report
- confusion matrix for whipsaw and tradeability
- PnL before/after kill-switch
- max drawdown before/after kill-switch
- average MAE before/after kill-switch
- avoided losing trades
- missed winning trades
- false positive opportunity cost
- false negative loss cost
- rule-breach count before/after
- monthly/quarterly stability table
- event-level audit log

---

## 10. Live Pipeline Requirements

### 10.1 Architecture

```text
Truth Social provider adapter
    -> raw post validator
    -> dedupe/idempotency layer
    -> post normaliser
    -> event store
    -> feature builder
    -> NLP classifier
    -> contradiction/whipsaw detector
    -> impact model
    -> signal composer
    -> API/WebSocket/Redis output
    -> dashboard + audit log
```

### 10.2 Live Service Requirements

- Python 3.11+.
- FastAPI for HTTP/WebSocket signal output.
- Async ingestion where provider supports streaming.
- Polling fallback where streaming is unavailable.
- Pydantic models for all input/output contracts.
- PostgreSQL for production event log.
- DuckDB/Parquet for research and local development.
- Redis optional for low-latency signal state and pub/sub.
- Prometheus-compatible metrics endpoint.
- Streamlit dashboard optional for v1; CLI monitor is acceptable.

### 10.3 Signal Output Contract

Every signal must use this JSON-compatible schema:

```json
{
  "event_id": "string",
  "post_id": "string",
  "source_provider": "string",
  "created_at_utc": "2026-01-01T12:00:00Z",
  "received_at_utc": "2026-01-01T12:00:01Z",
  "generated_at_utc": "2026-01-01T12:00:02Z",
  "text_clean": "string",
  "sentiment_label": "bullish_market|bearish_market|geopolitical_risk|neutral|volatility_only|low_confidence",
  "sentiment_confidence": 0.0,
  "topic_labels": ["trade_policy"],
  "topic_confidence": {"trade_policy": 0.0},
  "tradeability_label": "tradeable_directional|volatility_only|no_trade_whipsaw|no_impact|ambiguous",
  "direction_signal": "BULLISH|BEARISH|NEUTRAL|NO_TRADE",
  "p_direction": {
    "up_5m": 0.0,
    "down_5m": 0.0,
    "flat_5m": 0.0,
    "up_15m": 0.0,
    "down_15m": 0.0,
    "flat_15m": 0.0,
    "up_30m": 0.0,
    "down_30m": 0.0,
    "flat_30m": 0.0
  },
  "expected_delta_ticks": {
    "5m": 0.0,
    "15m": 0.0,
    "30m": 0.0
  },
  "risk": {
    "whipsaw_risk_level": "NONE|WATCH|SOFT_RISK|HARD_KILL",
    "whipsaw_score": 0.0,
    "volatility_risk_score": 0.0,
    "p_next_post_15m": 0.0,
    "p_next_post_30m": 0.0,
    "p_next_post_60m": 0.0
  },
  "kill_switch": {
    "action": "ALLOW|BLOCK_NEW_ENTRIES|REDUCE_SIZE|FLATTEN_OPTIONAL|HARD_FLAT",
    "risk_multiplier": 1.0,
    "ttl_seconds": 0,
    "reason_codes": ["string"]
  },
  "data_quality": {
    "feed_lag_ms": 0,
    "market_data_lag_ms": 0,
    "market_data_stale": false,
    "features_complete": true
  },
  "model_versions": {
    "classifier": "string",
    "impact_model": "string",
    "whipsaw_model": "string",
    "feature_set": "string"
  },
  "explanation": {
    "top_features": ["string"],
    "contradicting_post_ids": ["string"],
    "human_readable_reason": "string"
  }
}
```

### 10.4 Kill-Switch Semantics

Default mapping:

| Condition | Action | TTL |
|---|---|---|
| Feed stale or provider error | `BLOCK_NEW_ENTRIES` | until healthy |
| Market data stale | `BLOCK_NEW_ENTRIES` | until healthy |
| Macro blackout | `BLOCK_NEW_ENTRIES` | configured window |
| `WATCH` | `ALLOW` with warning | 15m |
| `SOFT_RISK` | `REDUCE_SIZE` or `BLOCK_NEW_ENTRIES` | 30m |
| `HARD_KILL` | `BLOCK_NEW_ENTRIES`; optional flatten | 60m default |
| High-confidence directional signal and no risk flags | `ALLOW` | 15m |

Do not force live flattening in v1 unless explicitly enabled by configuration.

---

## 11. Functional Requirements

### 11.1 CLI Commands

Implement CLI commands similar to:

```bash
python -m sentiment_engine ingest-posts --source stiles --out data/raw/posts.parquet
python -m sentiment_engine ingest-market --source csv --symbol NQ --in data/raw/nq.csv
python -m sentiment_engine build-events --config configs/research.yaml
python -m sentiment_engine label-assist --config configs/labels.yaml
python -m sentiment_engine train-classifier --config configs/model.yaml
python -m sentiment_engine train-impact --config configs/impact.yaml
python -m sentiment_engine evaluate --config configs/eval.yaml
python -m sentiment_engine backtest --config configs/backtest_topstep.yaml
python -m sentiment_engine serve --config configs/live.yaml
python -m sentiment_engine dashboard --config configs/live.yaml
```

### 11.2 API Endpoints

Minimum FastAPI endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness. |
| `/ready` | GET | Readiness including feed/model/market-data status. |
| `/signal/latest` | GET | Latest signal state. |
| `/signal/{event_id}` | GET | Signal details and explanation. |
| `/posts/ingest` | POST | Manual post ingestion for testing/fallback. |
| `/simulate/post` | POST | Run inference on supplied text without storing as live event. |
| `/metrics` | GET | Prometheus-compatible metrics. |
| `/ws/signals` | WebSocket | Live signal stream. |

### 11.3 Configuration

Use YAML configs for:

- source adapters
- model paths
- feature set version
- thresholds
- whipsaw windows
- macro blackout windows
- account rules
- latency assumptions
- execution assumptions
- environment mode: `research`, `paper`, `live_advisory`, `live_blocking`

Secrets must come from environment variables or secret manager, never from committed YAML files.

---

## 12. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Latency | P95 post-received-to-signal latency below 2 seconds in live mode. |
| Reliability | Missing/stale feed must trigger conservative no-trade state. |
| Reproducibility | All training runs must store seed, data hash, feature version, model version, and config. |
| Auditability | Every signal must be reconstructable from stored raw post, features, model versions, and thresholds. |
| Maintainability | Typed Python, Pydantic schemas, modular adapters, tests, clear README. |
| Scalability | Single-node deployment sufficient for v1; async ingestion and batch research pipeline must not block each other. |
| Security | API keys in environment variables; no secrets in logs; provider payloads stored safely. |
| Compliance | Respect data-provider licensing and platform terms. |
| Fault tolerance | On model load failure, stale market data, provider outage, or schema mismatch, emit safe `NO_TRADE_DATA_STALE` or equivalent. |

---

## 13. Project Requirements by Component

### 13.1 Repository Structure

Recommended layout:

```text
sentiment-engine/
  README.md
  pyproject.toml
  .env.example
  configs/
    research.yaml
    model.yaml
    impact.yaml
    whipsaw.yaml
    backtest_topstep.yaml
    live.yaml
  data/
    raw/
    interim/
    processed/
    fixtures/
  notebooks/
    01_event_study.ipynb
    02_label_review.ipynb
    03_backtest_review.ipynb
  src/sentiment_engine/
    __init__.py
    cli.py
    config.py
    schemas.py
    ingestion/
      posts_base.py
      posts_stiles.py
      posts_provider.py
      posts_fixture.py
      market_csv.py
      market_databento.py
      calendar_csv.py
    preprocessing/
      text.py
      time.py
      contracts.py
    features/
      text_features.py
      market_features.py
      burst_features.py
      feature_store.py
    labels/
      event_targets.py
      whipsaw_labels.py
      label_guidelines.md
    models/
      classifier.py
      impact.py
      contradiction.py
      timing.py
      calibration.py
    backtest/
      event_study.py
      simulator.py
      account_rules.py
      reports.py
    live/
      service.py
      signal_engine.py
      state.py
      dashboard.py
    monitoring/
      metrics.py
      logging.py
    utils/
      io.py
      hashing.py
      validation.py
  tests/
    unit/
    integration/
    fixtures/
```

### 13.2 Data Storage

Research mode:

- Parquet files partitioned by source/date.
- DuckDB for analysis queries.

Production mode:

- PostgreSQL tables:
  - `posts_raw`
  - `posts_clean`
  - `market_bars`
  - `event_features`
  - `event_targets`
  - `model_predictions`
  - `signals`
  - `backtest_runs`
  - `provider_health`

### 13.3 Tests

Required tests:

- Post deduplication.
- Timestamp conversion and DST handling.
- Market bar alignment to post timestamp.
- No target leakage in feature generation.
- Event-window de-overlap logic.
- Whipsaw-label calculation using synthetic price paths.
- Contradiction detector thresholding with fixtures.
- Signal JSON schema validation.
- Kill-switch state machine.
- Backtest account-rule simulation.
- Provider stale-feed fallback.
- Reproducible model-training smoke test on tiny fixtures.

---

## 14. Success Criteria

### 14.1 Data Success Criteria

| Criterion | Target |
|---|---|
| Historical post ingest success | ≥ 99% valid rows after schema validation. |
| Duplicate post handling | 100% deterministic dedupe by `post_id` and `content_hash`. |
| Timestamp alignment | 100% timezone-aware UTC fields; no naive datetimes accepted. |
| Market bar coverage | ≥ 99% of expected valid 1-minute bars for evaluated sessions. |
| Event-target generation | 100% reproducible from raw posts + market bars + config. |
| Live feed health | Stale provider detected and reflected in signal state within 10 seconds. |

### 14.2 NLP Success Criteria

| Criterion | Target |
|---|---|
| Sentiment/market implication macro F1 | ≥ 0.78 on temporally held-out test set. |
| Topic classifier macro F1 | ≥ 0.75 on temporally held-out test set. |
| Tradeability classifier macro F1 | ≥ 0.70 on temporally held-out test set. |
| Calibration error | Expected calibration error ≤ 0.10. |
| Abstention behaviour | Low-confidence abstention improves precision of actionable signals versus non-abstaining model. |
| Baseline comparison | Transformer or LightGBM hybrid must beat keyword/TF-IDF baseline by statistically meaningful margin. |

### 14.3 Impact Model Success Criteria

| Criterion | Target |
|---|---|
| Directional accuracy, all events | ≥ 55% and statistically above matched baseline. |
| Directional accuracy, high-confidence events | ≥ 58% after costs and latency assumptions. |
| Regression MAE | Better than naive same-session/time-of-day baseline. |
| Tail-risk classification | Identifies top-quartile realised range events with AUPRC above baseline. |
| Feature utility | VIX/VXN/session/volume features must show measurable validation lift or be removed. |
| Stability | No single topic or month contributes more than 50% of measured model edge. |

### 14.4 Whipsaw Detector Success Criteria

Use both classification and trading-value metrics.

| Criterion | Target |
|---|---|
| `SOFT_RISK` whipsaw recall | ≥ 0.85 on held-out period. |
| `HARD_KILL` whipsaw precision | ≥ 0.75 on held-out period. |
| False-negative loss capture | Avoids or flags ≥ 75% of historical whipsaw-dollar loss exposure in backtest. |
| Opportunity-cost cap | Kill-switch suppresses ≤ 20% of otherwise eligible trading minutes unless performance gain justifies more. |
| Historical known-event recall | Flags ≥ 85% of manually curated known whipsaw events. |
| Threshold robustness | Selected threshold remains within 10% performance degradation across walk-forward folds. |

### 14.5 Trading Success Criteria

| Criterion | Target |
|---|---|
| Kill-switch PnL attribution | Net positive after commissions, slippage, and latency. |
| Max drawdown impact | Reduces max drawdown of baseline strategy by ≥ 10% without destroying expectancy. |
| MAE impact | Reduces average adverse excursion on filtered trades by ≥ 15%. |
| Rule-breach impact | Reduces simulated daily-loss / max-loss breaches versus baseline. |
| Missed-winner control | Missed profitable trades from false positives are reported and remain below configured opportunity-cost limit. |
| Shadow-mode gate | At least 30 live/paper trading days with stable latency, no schema failures, and positive or neutral risk-filter attribution before live blocking mode. |

### 14.6 Live System Success Criteria

| Criterion | Target |
|---|---|
| Post received-to-signal latency | P95 < 2 seconds, P99 < 5 seconds. |
| Signal availability | ≥ 99% during configured trading hours, excluding provider outages. |
| Safe degradation | 100% of provider/model/market-data failures result in conservative signal state. |
| Audit completeness | 100% of live signals store raw input, features, model versions, thresholds, and output. |
| Dashboard freshness | Latest signal state visible within 2 seconds of generation. |
| Config safety | Live flattening disabled by default; paper mode default for new deployments. |

---

## 15. Implementation Milestones

### Milestone 1 — Research Data Foundation

Deliver:

- Post ingestion adapters for historical archive and fixture data.
- Market OHLCV CSV/Parquet ingestion.
- Time alignment and contract handling.
- Event dataset builder.
- Basic event-study report.

Acceptance:

- Can ingest historical posts and NQ bars.
- Can produce event targets for 5m/15m/30m.
- Can reproduce event dataset from config.

### Milestone 2 — Labeling and Baselines

Deliver:

- Label guidelines.
- Seed labelled dataset.
- Keyword/TF-IDF/LightGBM baselines.
- Initial tradeability and whipsaw labels.

Acceptance:

- Baseline metrics generated with temporal split.
- Label audit report exists.
- Synthetic whipsaw labels pass unit tests.

### Milestone 3 — NLP and Impact Models

Deliver:

- Fine-tuned or embedded classifier.
- Impact classifier/regressor.
- Calibration report.
- SHAP/feature-importance report.

Acceptance:

- Models beat baselines on validation.
- Models save versioned artefacts.
- Inference returns complete signal fields on fixture posts.

### Milestone 4 — Whipsaw and Timing Models

Deliver:

- Contradiction detector.
- Rolling post-state engine.
- Empirical post-timing model.
- Combined whipsaw risk score.

Acceptance:

- Risk levels emitted with explanations.
- Thresholds configurable.
- Held-out whipsaw report generated.

### Milestone 5 — Backtesting

Deliver:

- Kill-switch overlay simulator.
- Topstep-style account-rule config.
- Backtest attribution report.
- Event audit log.

Acceptance:

- Reports PnL before/after kill-switch.
- Reports avoided losses and missed winners.
- Uses slippage, commissions, latency, and account rules.

### Milestone 6 — Live Inference Service

Deliver:

- FastAPI service.
- Provider adapter interface.
- Signal WebSocket and latest-signal endpoint.
- Health/readiness/metrics endpoints.
- Dashboard or CLI monitor.

Acceptance:

- Fixture live feed produces valid signals.
- Stale provider triggers safe no-trade state.
- P95 local inference latency below 2 seconds on target hardware.

### Milestone 7 — Paper / Shadow Deployment

Deliver:

- Paper-mode deployment config.
- Live audit logging.
- Daily performance report.
- Shadow-mode comparison against baseline strategy.

Acceptance:

- 30 paper/live-observation days completed before enabling live blocking.
- No unhandled schema/provider/model errors.
- Risk-filter attribution is positive or operationally useful.

---

## 16. AI Coding Agent Instructions

When implementing this project:

1. Build vertical slices; do not start with a large model before data alignment works.
2. Prefer clear, testable modules over notebooks-only workflows.
3. Use type hints and Pydantic schemas for all external boundaries.
4. Use deterministic seeds and store config/data/model hashes.
5. Never mix research-only target generation into live inference features.
6. Implement safe failure modes before live provider integration.
7. Use fixture data for all tests; tests must not require paid APIs.
8. Do not hardcode account rules, tick values, provider URLs, API keys, or thresholds in business logic.
9. Keep all thresholds in config files.
10. Log enough detail to reconstruct every signal.
11. Make live trading actions advisory by default; live blocking/flattening requires explicit config.
12. Add TODO markers only when accompanied by a failing or skipped test that defines expected behaviour.

---

## 17. Minimum Viable v1 Deliverable

A successful v1 does **not** need to prove fully automated directional alpha. It must prove that the system can reliably:

1. Ingest and align posts with NQ market data.
2. Build a leakage-safe event dataset.
3. Classify posts into market-relevant labels with measured accuracy.
4. Detect contradictory/burst posting behaviour.
5. Define and backtest market whipsaw events.
6. Emit a real-time risk signal with safe degradation.
7. Show whether using the signal as a kill-switch improves risk-adjusted trading outcomes.

The v1 decision gate is:

```text
Ship to live advisory / shadow mode if:
  data pipeline is reliable,
  live signal latency is below target,
  whipsaw detector improves risk metrics out-of-sample,
  backtest attribution remains positive after realistic costs,
  and all safe-failure tests pass.

Do not ship to live blocking mode until shadow-mode evidence is collected.
```

---

## 18. Suggested Default Configuration

```yaml
project:
  mode: research
  timezone_internal: UTC

instruments:
  signal_symbol: NQ
  execution_symbol: MNQ
  tick_size: 0.25
  nq_tick_value_usd: 5.0
  mnq_tick_value_usd: 0.5

windows:
  impact_horizons_minutes: [5, 15, 30]
  contradiction_window_minutes: 60
  whipsaw_evaluation_window_minutes: 30
  macro_blackout_before_minutes: 15
  macro_blackout_after_minutes: 15

thresholds:
  min_direction_confidence: 0.58
  min_topic_confidence: 0.60
  soft_whipsaw_threshold: 0.55
  hard_whipsaw_threshold: 0.75
  stale_feed_seconds: 10
  stale_market_data_seconds: 10

live_actions:
  default_mode: live_advisory
  allow_live_flatten: false
  soft_risk_action: REDUCE_SIZE
  hard_risk_action: BLOCK_NEW_ENTRIES
  hard_risk_ttl_seconds: 3600

backtest:
  commission_per_contract_usd: 0.0
  slippage_ticks_per_side: 1
  latency_seconds: 2
  account_rules_config: configs/backtest_topstep.yaml
```

---

## 19. Final Acceptance Checklist

Before calling the project complete, verify:

- [ ] Historical post ingestion works from at least one source.
- [ ] Live provider adapter works with fixtures and one real provider.
- [ ] NQ/MNQ data ingestion works from CSV/Parquet.
- [ ] All timestamps are timezone-aware UTC.
- [ ] Event target generation is reproducible.
- [ ] Feature generation is point-in-time safe.
- [ ] Baseline models are implemented and reported.
- [ ] NLP model evaluation uses temporal splits.
- [ ] Impact model beats naive baseline where claimed.
- [ ] Whipsaw labels are clearly defined and unit-tested.
- [ ] Whipsaw detector reports recall, precision, opportunity cost, and dollar-loss capture.
- [ ] Backtest includes slippage, commission, latency, and account rules.
- [ ] Live service emits valid signal JSON.
- [ ] Stale data/provider failures produce safe no-trade state.
- [ ] Every signal is auditable by model/config/data version.
- [ ] README explains setup, configs, commands, and safety assumptions.
- [ ] Shadow-mode deployment is completed before live blocking mode.
