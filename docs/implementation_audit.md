# Implementation Audit

Verification date: 2026-06-12.

This audit answers what is actually implemented in the current repo. It is based on the executable fixture pipeline, not on claims from the original scope.

## Bottom Line

- The repo implements a fixture-backed research and live-advisory vertical slice.
- It implements a PyTorch TF-IDF MLP smoke baseline, but it does not implement production-grade transformer models.
- It does not prove a tradable edge.
- It implements three executable ML/DL baseline paths: TF-IDF logistic regression, optional LightGBM, and optional PyTorch TF-IDF MLP.
- It implements rule-based sentiment/topic labels, deterministic event targets, weighted whipsaw scoring, Optuna tuning reports, a kill-switch backtest, a signal JSON contract, an optional FastAPI service, and a static dashboard.
- All current results are from 8 fixture posts, 340 fixture NQ one-minute bars, and 4 fixture baseline trades.
- It can now ingest the CNN-hosted Trump Truth Social archive for historical/backfill posts via `ingest-archive`.

## Implemented Models

| Component | Implemented | Type | File |
|---|---:|---|---|
| Keyword/topic classifier | Yes | Rule-based NLP baseline | `src/sentiment_engine/models/rules.py` |
| Tradeability baseline | Yes | TF-IDF + logistic regression | `src/sentiment_engine/models/baselines.py` |
| LightGBM tradeability baseline | Yes | Optional LightGBM text/context baseline | `src/sentiment_engine/models/baselines.py` |
| Neural tradeability smoke baseline | Yes | Optional PyTorch MLP over TF-IDF/context features | `src/sentiment_engine/models/baselines.py` |
| Naive baseline | Yes | Majority-class baseline | `src/sentiment_engine/models/baselines.py` |
| Whipsaw detector | Yes | Weighted transparent scoring model | `src/sentiment_engine/models/whipsaw.py` |
| Timing model | Yes | Empirical hazard/count features | `src/sentiment_engine/models/timing.py` |
| Optuna tuning | Yes | TPE optimization of whipsaw weights/thresholds | `src/sentiment_engine/models/tuning.py` |
| CNN archive adapter | Yes | Real post backfill ingestion | `src/sentiment_engine/ingestion/posts_cnn_archive.py` |
| Transformer / FinBERT / DeBERTa | No | Not implemented | Real labels and larger data required first |
| LSTM / deep sequence model | No | Not implemented | Not justified for v1 fixtures |

## Results From Latest Full Run

Command:

```bash
python scripts/run_full_pipeline.py configs/research.yaml
```

Output summary:

```text
ingested 8 posts
ingested 340 market bars
built 8 events
event study written for 8 events
labeled 8 events
classifier baselines evaluated on 3 temporal holdout rows
scored whipsaw risk for 8 events
whipsaw tuning complete: trials=60, holdout_soft_recall=1.0
backtest complete: before=-56.38, after=-1.96
dashboard written to reports/dashboard.html
full pipeline completed
```

Test command:

```bash
PYTHONPATH=src python -m pytest -q
```

Output summary:

```text
10 passed, 1 warning
```

## Real Archive Smoke Test

Command:

```bash
PYTHONPATH=src python -m sentiment_engine --config configs/research.yaml ingest-archive --url https://ix.cnn.io/data/truth-social/truth_archive.parquet --limit 25
```

Result:

- 25 archive posts ingested.
- 25 valid rows.
- 0 duplicate post IDs.
- 4 empty-text rows.
- 4 media-only rows.
- Date range in smoke sample: `2026-06-10T13:19:46.612000Z` to `2026-06-12T13:59:27.160000Z`.

This proves the current archive schema can be normalized by the repo. It does not prove market-label quality or live-trading suitability.

## Full Archive Backfill Test

Command:

```bash
PYTHONPATH=src python -m sentiment_engine --config configs/research.yaml ingest-archive --url https://ix.cnn.io/data/truth-social/truth_archive.parquet
```

Result:

- 33,899 archive posts ingested.
- 33,899 valid rows.
- 0 duplicate post IDs.
- 6,392 empty-text rows.
- 5,830 media-only rows.
- Date range: `2022-02-14T15:54:32.528000Z` to `2026-06-12T13:59:27.160000Z`.

This verifies the post backfill path at full archive scale. It still does not solve the required NQ/MNQ market-data join or human-label problem.

## Classifier Results

Source report: `reports/classifier_baseline_report.json`.

Dataset split:

- Rows: 8
- Train rows: 5
- Temporal holdout rows: 3
- Train end: `2026-01-05T14:40:13Z`
- Test start: `2026-01-05T14:50:35Z`
- Split method: time-ordered holdout, not random split.

Metrics:

| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| Naive majority baseline | 0.3333 | 0.2500 | Predicts majority class only |
| Rule tradeability baseline | 0.3333 | 0.1667 | Text/topic heuristics |
| TF-IDF + logistic regression | 0.6667 | 0.6667 | ML baseline; tiny holdout |
| LightGBM text/context | 0.6667 | 0.6667 | Optional ML baseline; excludes post-event target columns |
| PyTorch TF-IDF MLP | 0.3333 | 0.2500 | DL smoke baseline, not a transformer |

The macro F1 result is not statistically meaningful because the fixture holdout has only 3 rows. It is an integration check proving the evaluation path works.

Additional model reports:

- `reports/lightgbm_baseline_report.json`
- `reports/neural_baseline_report.json`

Both reports include explicit methodology notes warning that fixture metrics are not evidence of economic edge.

## Whipsaw Results

Source report: `reports/whipsaw_report.json`.

Risk counts:

- `HARD_KILL`: 1
- `SOFT_RISK`: 2
- `WATCH`: 3
- `NONE`: 2

Metrics against fixture market-whipsaw labels:

| Gate | Precision | Recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|
| Soft risk or harder | 0.6667 | 0.6667 | 2 | 1 | 4 | 1 |
| Hard kill only | 1.0000 | 0.3333 | 1 | 0 | 5 | 2 |

Mean score:

- Actual whipsaw events: `0.531111`
- Non-whipsaw events: `0.275333`

## Backtest Results

Source report: `reports/backtest_report.json`.

The fixture kill-switch overlay uses:

- MNQ tick value: `$0.50`
- Tick size: `0.25`
- Commission per contract: `$0.74`
- Slippage: `1` tick per side
- Signal latency: `2` seconds
- Topstep-style account rules from `configs/backtest_topstep.yaml`

Results:

| Metric | Value |
|---|---:|
| Trades | 4 |
| Filtered trades | 2 |
| Reduced trades | 1 |
| Net PnL before | `-56.38` |
| Net PnL after | `-1.96` |
| Kill-switch value | `54.42` |
| Avoided losing-trade exposure | `52.96` |
| Missed winning-trade exposure | `11.02` |
| Daily-loss breaches before/after | `0 / 0` |
| Max-loss breaches before/after | `0 / 0` |
| Trailing-drawdown breaches before/after | `0 / 0` |

These results are fixture accounting checks only. They are not evidence of economic edge.

## Optuna Tuning

Optuna is now declared as a project dependency in `pyproject.toml` and used by:

```bash
PYTHONPATH=src python -m sentiment_engine tune-whipsaw --config configs/research.yaml
```

Source report: `reports/whipsaw_tuning_report.json`.

Tuned parameters:

- `headline_risk` weight
- `market_relevance` weight
- `text_contradiction` weight
- `post_burst` weight
- `direction_flip` weight
- `same_topic` weight
- `volatility_regime` weight
- `soft_whipsaw_threshold`
- `hard_whipsaw_threshold`

Latest tuning run:

- Optimizer: Optuna TPE
- Trials: 60
- Seed: 42
- Train rows: 5
- Holdout rows: 3
- Best train objective: `0.475`

Best reported parameters:

```json
{
  "weights": {
    "headline_risk": 0.278198,
    "market_relevance": 0.204675,
    "text_contradiction": 0.063113,
    "post_burst": 0.051895,
    "direction_flip": 0.063616,
    "same_topic": 0.234568,
    "volatility_regime": 0.103936
  },
  "soft_whipsaw_threshold": 0.325563,
  "hard_whipsaw_threshold": 0.418492
}
```

Holdout metrics for the tuned fixture parameters:

- Soft risk precision: `1.0`
- Soft risk recall: `1.0`
- Hard kill precision: `1.0`
- Hard kill recall: `1.0`

Important: tuned parameters are not written back into live config because the sample is too small. They are reported for review only.

## Recorded Metrics

Data quality:

- Post row count
- Valid post rows
- Duplicate post IDs
- Min/max post timestamp
- Max feed lag
- Market bar row count
- Valid/invalid market rows
- Market-bar gaps greater than one minute
- Symbols present
- Skipped event-build posts

Event study:

- Event count
- Market-whipsaw count
- Tradeability label counts
- 5m/15m/30m delta count, mean, median, standard deviation, p10, p90
- Segment-level mean 30m ticks and realised range

Classifier:

- Train/test rows
- Temporal split timestamps
- Accuracy
- Macro F1
- Precision/recall/F1 by class
- Confusion matrix

Whipsaw:

- Risk-level counts
- Soft/hard precision and recall
- TP/FP/TN/FN
- Mean whipsaw score by actual class

Optuna:

- Trials
- Seed
- Objective value
- Best weights
- Best thresholds
- Train metrics
- Chronological holdout metrics
- Default-parameter holdout metrics

Backtest:

- Trades
- Filtered/reduced trades
- Net PnL before/after
- Kill-switch value
- Avoided losing exposure
- Missed winning exposure
- Account-rule breach counts

Live signal:

- Direction signal
- Kill-switch action
- Risk multiplier
- TTL
- Whipsaw score
- Next-post probabilities
- Data-quality flags
- Model versions
- Explanation fields

## Hyperparameters

Current fixed classifier hyperparameters:

- `TfidfVectorizer(max_features=250, ngram_range=(1, 2))`
- `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`
- `LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, min_data_in_leaf=1, n_jobs=1, random_state=42)`
- PyTorch MLP: TF-IDF/context input, `16` hidden units, `40` epochs, Adam learning rate `0.03`
- Temporal holdout fraction: `0.35`
- Seed: `42`

Current fixed whipsaw defaults:

- Soft threshold: `0.55`
- Hard threshold: `0.75`
- Default weights:
  - `headline_risk`: `0.35`
  - `market_relevance`: `0.30`
  - `text_contradiction`: `0.15`
  - `post_burst`: `0.10`
  - `direction_flip`: `0.07`
  - `same_topic`: `0.03`
  - `volatility_regime`: `0.00`

Tuned whipsaw hyperparameters are recorded separately in `reports/whipsaw_tuning_report.json` and are not automatically promoted.

## Data Cleanliness

Current fixture data passes these checks:

- 8 of 8 posts valid.
- 0 duplicate post IDs.
- 340 of 340 market bars valid.
- 0 invalid market rows.
- 0 skipped event-build posts.
- All internal timestamps are timezone-aware UTC.
- Post targets align to the first market bar after `received_at_utc`.
- Keyword matching uses word boundaries for single-token keywords, which prevents false matches like `ai` inside `again`.
- Real archive smoke ingest also records empty-text/media-only counts, which must be excluded or separately handled for text-only DL training.

Known limitation:

- The one reported market gap greater than one minute is the intentional gap between the two fixture trading days, not an intraday missing-bar error.
- Real data cleanliness has not been proven. The source audit says real historical/live sources still require licensing, schema, latency, and coverage checks before research claims.

## What Is Still Not Done

- No real Truth Social provider integration has been validated.
- No real NQ/MNQ licensed historical data has been loaded.
- No human-labeled dataset exists.
- No transformer, FinBERT, DeBERTa, LSTM, or production-grade deep-learning model is trained.
- LightGBM and PyTorch MLP are fixture smoke baselines only.
- No statistical significance claim can be made from fixtures.
- No Optuna result should be promoted without real walk-forward validation.
- No live blocking should be enabled; current posture remains advisory/shadow-mode.
