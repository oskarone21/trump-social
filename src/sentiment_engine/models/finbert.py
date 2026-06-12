from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from sentiment_engine.utils.io import write_dataframe, write_json

DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REPORT_NAME = "finbert_inference_report.json"
FINBERT_SCORES_NAME = "finbert_scores.parquet"
TEXT_COLUMN = "text_clean"
EVENT_ID_COLUMN = "event_id"
POST_ID_COLUMN = "post_id"
RECEIVED_AT_COLUMN = "received_at_utc"
POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"
NEUTRAL_LABEL = "neutral"
FINBERT_LABELS = [POSITIVE_LABEL, NEGATIVE_LABEL, NEUTRAL_LABEL]
FINBERT_TONE_LABEL_MAP = {
    "LABEL_0": NEUTRAL_LABEL,
    "LABEL_1": POSITIVE_LABEL,
    "LABEL_2": NEGATIVE_LABEL,
}


class TextClassifier(Protocol):
    def __call__(self, texts: list[str], **kwargs: Any) -> list[Any]:
        pass


def score_finbert_sentiment(
    events: pd.DataFrame,
    *,
    report_dir: Path,
    scores_path: Path | None = None,
    model_name: str = DEFAULT_FINBERT_MODEL,
    batch_size: int = 8,
    local_files_only: bool = False,
    classifier: TextClassifier | None = None,
) -> dict[str, Any]:
    if TEXT_COLUMN not in events:
        raise ValueError(f"FinBERT inference requires a {TEXT_COLUMN} column")

    report_dir.mkdir(parents=True, exist_ok=True)
    scores_path = scores_path or report_dir / FINBERT_SCORES_NAME
    scoring_frame = events.copy()
    text = scoring_frame[TEXT_COLUMN].fillna("").astype(str)
    scoring_frame = scoring_frame[text.str.strip() != ""].reset_index(drop=True)
    skipped_empty = int(len(events) - len(scoring_frame))
    if scoring_frame.empty:
        report = _skipped_report(
            reason="no_non_empty_text",
            remediation="Filter or enrich media-only posts before text-model inference.",
            model_name=model_name,
            event_rows=len(events),
            skipped_empty_text_rows=skipped_empty,
        )
        write_json(report_dir / FINBERT_REPORT_NAME, report)
        return report

    try:
        scorer = classifier or _load_finbert_pipeline(
            model_name, local_files_only=local_files_only
        )
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        report = _skipped_report(
            reason="finbert_model_unavailable",
            remediation=(
                "Install the optional dl dependency group and make the FinBERT model "
                "weights available locally or via permitted Hugging Face access."
            ),
            model_name=model_name,
            event_rows=len(events),
            skipped_empty_text_rows=skipped_empty,
            detail=str(exc),
        )
        write_json(report_dir / FINBERT_REPORT_NAME, report)
        return report

    texts = scoring_frame[TEXT_COLUMN].astype(str).tolist()
    raw_predictions = scorer(texts, batch_size=batch_size, truncation=True, top_k=None)
    score_rows = [
        _score_row(scoring_frame.iloc[index], raw_prediction)
        for index, raw_prediction in enumerate(raw_predictions)
    ]
    scores = pd.DataFrame(score_rows)
    write_dataframe(scores, scores_path)
    label_counts = scores["finbert_label"].value_counts().to_dict()
    report = {
        "status": "scored",
        "model_name": model_name,
        "mode": "inference_only",
        "local_files_only": bool(local_files_only),
        "event_rows": int(len(events)),
        "scored_rows": int(len(scores)),
        "skipped_empty_text_rows": skipped_empty,
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        "mean_scores": {
            label: round(float(scores[f"finbert_{label}_score"].mean()), 6)
            for label in FINBERT_LABELS
        },
        "scores_path": str(scores_path),
        "methodology_notes": [
            "FinBERT inference is a zero/few-shot financial-tone baseline, not a model "
            "fine-tuned on Trump Truth Social posts.",
            "Scores are not calibrated for NQ/MNQ tradeability and must not be used as "
            "live trading thresholds without temporal validation and human labels.",
            "Empty-text and media-only posts are skipped for text-only transformer scoring.",
        ],
    }
    write_json(report_dir / FINBERT_REPORT_NAME, report)
    return report


def _load_finbert_pipeline(model_name: str, *, local_files_only: bool) -> TextClassifier:
    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model.eval()

    class FinbertClassifier:
        def __call__(self, texts: list[str], **kwargs: Any) -> list[Any]:
            batch_size = int(kwargs.get("batch_size", 8))
            predictions: list[Any] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=bool(kwargs.get("truncation", True)),
                    return_tensors="pt",
                )
                with torch.no_grad():
                    probabilities = torch.softmax(model(**encoded).logits, dim=1)
                for row in probabilities:
                    predictions.append(
                        [
                            {
                                "label": model.config.id2label.get(index, f"LABEL_{index}"),
                                "score": float(score),
                            }
                            for index, score in enumerate(row.tolist())
                        ]
                    )
            return predictions

    return FinbertClassifier()


def _score_row(event: pd.Series, raw_prediction: Any) -> dict[str, Any]:
    scores = _normalise_scores(raw_prediction)
    top_label = max(FINBERT_LABELS, key=lambda label: scores[label])
    row = {
        EVENT_ID_COLUMN: event.get(EVENT_ID_COLUMN),
        POST_ID_COLUMN: event.get(POST_ID_COLUMN),
        RECEIVED_AT_COLUMN: event.get(RECEIVED_AT_COLUMN),
        TEXT_COLUMN: event.get(TEXT_COLUMN),
        "finbert_label": top_label,
        "finbert_score": round(float(scores[top_label]), 6),
    }
    for label in FINBERT_LABELS:
        row[f"finbert_{label}_score"] = round(float(scores[label]), 6)
    return row


def _normalise_scores(raw_prediction: Any) -> dict[str, float]:
    entries = _prediction_entries(raw_prediction)
    scores = {label: 0.0 for label in FINBERT_LABELS}
    for entry in entries:
        label = _normalise_label(str(entry["label"]))
        if label in scores:
            scores[label] = float(entry["score"])
    return scores


def _prediction_entries(raw_prediction: Any) -> list[dict[str, Any]]:
    if isinstance(raw_prediction, dict):
        return [raw_prediction]
    if isinstance(raw_prediction, list) and raw_prediction and isinstance(raw_prediction[0], dict):
        return raw_prediction
    return list(raw_prediction)


def _normalise_label(label: str) -> str:
    mapped = FINBERT_TONE_LABEL_MAP.get(label, label)
    return mapped.lower()


def _skipped_report(
    *,
    reason: str,
    remediation: str,
    model_name: str,
    event_rows: int,
    skipped_empty_text_rows: int,
    detail: str | None = None,
) -> dict[str, Any]:
    report = {
        "status": "skipped",
        "reason": reason,
        "model_name": model_name,
        "event_rows": int(event_rows),
        "skipped_empty_text_rows": int(skipped_empty_text_rows),
        "remediation": remediation,
        "methodology_notes": [
            "A skipped FinBERT report is explicit so missing model weights are not "
            "mistaken for trained transformer results."
        ],
    }
    if detail:
        report["detail"] = detail
    return report
