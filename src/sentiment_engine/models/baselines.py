from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from sentiment_engine.features.text_features import enrich_with_text_features
from sentiment_engine.models.metadata import model_metadata
from sentiment_engine.utils.io import write_dataframe, write_json

TARGET_COLUMN = "tradeability_label"
TEXT_COLUMN = "text_clean"
TEMPORAL_COLUMN = "received_at_utc"
TFIDF_MODEL_VERSION = "tfidf_logreg_v1"


def build_labeled_events(events: pd.DataFrame) -> pd.DataFrame:
    return enrich_with_text_features(events)


def train_tradeability_baselines(
    labeled_events: pd.DataFrame,
    *,
    report_dir: Path,
    model_dir: Path,
    config_path: str,
    temporal_test_fraction: float = 0.35,
    seed: int = 42,
    tfidf_max_features: int = 250,
) -> dict[str, Any]:
    ordered = labeled_events.sort_values(TEMPORAL_COLUMN).reset_index(drop=True)
    split_index = max(1, int(len(ordered) * (1.0 - temporal_test_fraction)))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    if test.empty:
        raise ValueError("Temporal split produced an empty test set")

    naive_predictions = _naive_predictions(train[TARGET_COLUMN], len(test))
    rule_predictions = test["rule_tradeability_label"].tolist()
    tfidf_model, tfidf_predictions = _fit_tfidf(train, test, seed, tfidf_max_features)

    target_labels = sorted(ordered[TARGET_COLUMN].unique().tolist())
    report = {
        "row_count": int(len(ordered)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split": {
            "train_end_utc": str(train[TEMPORAL_COLUMN].max()),
            "test_start_utc": str(test[TEMPORAL_COLUMN].min()),
            "method": "time_ordered_holdout",
        },
        "target_labels": target_labels,
        "naive": _metrics(test[TARGET_COLUMN].tolist(), naive_predictions),
        "rules": _metrics(test[TARGET_COLUMN].tolist(), rule_predictions),
        "tfidf_logreg": _metrics(test[TARGET_COLUMN].tolist(), tfidf_predictions),
        "methodology_notes": [
            "Fixture metrics are engineering smoke checks, not statistical evidence.",
            "Rows are split in timestamp order; no random headline split is reported.",
            "Labels are weak research labels until human adjudication exists.",
        ],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "classifier_baseline_report.json", report)
    write_dataframe(labeled_events, report_dir / "labeled_events_snapshot.csv")
    if tfidf_model is not None:
        joblib.dump(tfidf_model, model_dir / "tfidf_tradeability_baseline.joblib")
    write_json(
        model_dir / "tfidf_tradeability_baseline.metadata.json",
        model_metadata(
            model_name="tradeability_baseline",
            model_version=TFIDF_MODEL_VERSION,
            config_path=config_path,
            data=labeled_events,
            extra={"target": TARGET_COLUMN, "temporal_test_fraction": temporal_test_fraction},
        ),
    )
    return report


def _fit_tfidf(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, tfidf_max_features: int
) -> tuple[Pipeline | None, list[str]]:
    unique_labels = train[TARGET_COLUMN].nunique()
    if unique_labels < 2:
        return None, _naive_predictions(train[TARGET_COLUMN], len(test))
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    model.fit(train[TEXT_COLUMN].astype(str), train[TARGET_COLUMN].astype(str))
    predictions = model.predict(test[TEXT_COLUMN].astype(str)).tolist()
    return model, predictions


def _naive_predictions(labels: pd.Series, count: int) -> list[str]:
    if labels.empty:
        raise ValueError("Cannot build naive baseline without training labels")
    majority = labels.value_counts().index[0]
    return [str(majority)] * count


def _metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true).union(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": {
            "labels": labels,
            "rows": matrix.tolist(),
        },
    }
