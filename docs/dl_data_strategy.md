# Data and DL Strategy

Verification date: 2026-06-12.

## Problem

The implementation now includes a PyTorch TF-IDF MLP smoke baseline so the DL training/reporting path can be executed, plus a FinBERT inference baseline for financial-tone scoring when model weights are available. It still does not fine-tune FinBERT, DeBERTa, LSTM, or other production-grade deep-learning models because the repo only has local fixtures. That remains the right constraint: a transformer trained on eight synthetic rows would be fake progress.

## Practical Source Path

The best current route is:

1. Use the CNN-hosted Truth Social archive for Trump post backfill.
2. Join those posts to licensed NQ/MNQ one-minute market data.
3. Generate event targets and market-whipsaw labels.
4. Use weak labels plus human review to create a real labeled dataset.
5. Train DL models only after baselines and label quality are measurable.

Verified source:

- `stiles/trump-truth-social-archive` says its GitHub workflow was disabled on 2025-10-26.
- The same README points to `https://ix.cnn.io/data/truth-social/truth_archive.json`, with `.csv` and `.parquet` variants.
- A local HEAD check on 2026-06-12 returned HTTP 200 for both JSON and parquet.
- JSON size was about 18.5 MB; parquet size was about 6.2 MB.
- The archive was last modified on 2026-06-12.

## Implemented Now

The repo now has a real archive adapter:

```bash
PYTHONPATH=src python -m sentiment_engine ingest-archive \
  --config configs/research.yaml \
  --url https://ix.cnn.io/data/truth-social/truth_archive.parquet \
  --limit 100
```

It writes:

- `data/processed/cnn_archive_posts.parquet`
- `reports/cnn_archive_ingestion_audit.json`

The adapter also supports local JSON/CSV/parquet files for repeatable tests.

The post audit records empty-text and media-only rows. Those rows are valid archive records, but they should not be blindly used for text-only DL training.

Full archive verification on 2026-06-12 ingested 33,899 valid rows with 0 duplicate post IDs. The audit found 6,392 empty-text rows and 5,830 media-only rows, so the training dataset must filter text-empty rows or route them through separate media/link features.

The market-data side now has executable CSV/parquet export and Databento API paths:

```bash
PYTHONPATH=src python -m sentiment_engine ingest-market-file \
  --config configs/research.yaml \
  --input path/to/licensed_nq_ohlcv.parquet \
  --source-name databento_glbx_mdp3_ohlcv_1m \
  --symbol-root NQ
```

```bash
pip install -e ".[market]"
export DATABENTO_API_KEY="..."
PYTHONPATH=src python -m sentiment_engine download-databento-market \
  --config configs/research.yaml \
  --start 2022-02-14T00:00:00Z \
  --end 2026-06-12T23:59:59Z \
  --symbols NQ.c.0 \
  --symbol-root NQ
```

Use this with Databento `GLBX.MDP3` `ohlcv-1m` exports/API access or equivalent broker OHLCV files, then run `build-archive-events`. This still requires licensed market data that covers the archive period.

The label side now has an executable review path:

```bash
PYTHONPATH=src python -m sentiment_engine export-label-queue \
  --config configs/research.yaml \
  --events data/processed/real_events.parquet \
  --out data/interim/label_queue.csv

PYTHONPATH=src python -m sentiment_engine import-reviewed-labels \
  --config configs/research.yaml \
  --input path/to/reviewed_labels.csv \
  --label-version human_v1
```

The review queue excludes post-event price-target columns. Reviewed labels are kept in `data/processed/human_labels.parquet`, separate from weak rule labels and market targets.

The repo also writes visible ML/DL smoke reports during the fixture pipeline:

- `reports/classifier_baseline_report.json`
- `reports/lightgbm_baseline_report.json`
- `reports/neural_baseline_report.json`
- `reports/finbert_inference_report.json` when `scripts/run_finbert_inference.py` is run

The classifier report includes naive, rules, TF-IDF logistic regression, TF-IDF linear SVM, optional LightGBM, and optional PyTorch TF-IDF MLP rows. Probabilistic baselines report log loss, multiclass Brier score, expected calibration error, and confidence-threshold abstention diagnostics. The neural report is a PyTorch MLP over TF-IDF/context features. FinBERT inference is reported separately because it is a financial-tone score, not a tradeability model trained on this event set. These paths prove deterministic training/evaluation/reporting mechanics; they are not substitutes for a human-labeled validation set.

FinBERT inference verification:

```bash
python scripts/run_finbert_inference.py --config configs/research.yaml --limit 2
```

The verified local run used `ProsusAI/finbert` and scored 2 fixture rows. Both rows were classified as positive financial tone. After the model files are cached, add `--local-files-only` for deterministic offline reruns. This proves model loading, tokenization, scoring, and report writing; it does not prove predictive power.

## DL Readiness Gate

Do not train or trust a DL model until these are true:

- At least several hundred event-labeled posts after joining to market bars.
- A human-reviewed validation set for sentiment/topic/tradeability.
- Temporal train/validation/test split.
- Baselines already measured: naive, rules, TF-IDF logistic regression, and preferably LightGBM.
- Probability quality is measured before any abstention threshold is promoted.
- Label leakage prevented: no post-event fields in live-time text features.
- Results beat baselines on out-of-sample periods.
- Calibration and abstention improve actionable precision.

## Candidate Models Once Data Exists

| Model | Use | Gate |
|---|---|---|
| FinBERT | Financial-tone baseline | Only if it beats TF-IDF on held-out events |
| DeBERTa/DistilBERT | Political/social-media language classifier | Needs human labels and calibration |
| Sentence-transformer | Retrieval, clustering, similar-post lookup | Useful before full fine-tuning |
| Cross-encoder NLI | Contradiction/stance scoring | Needs pair/cluster labels |
| LightGBM | Text + market-context baseline | Should be trained before transformer promotion |

## Live Source Path

The CNN archive is useful for backfill and near-current monitoring, but it should not be treated as a trading-grade live feed without latency checks. For live advisory mode, use a provider adapter with:

- heartbeat and stale-feed detection
- schema validation
- explicit provider terms
- local audit log
- conservative `BLOCK_NEW_ENTRIES` on feed failure
