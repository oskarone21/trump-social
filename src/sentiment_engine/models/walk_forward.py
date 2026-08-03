from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_engine.models.baselines import (
    TARGET_COLUMN,
    TEMPORAL_COLUMN,
    _fit_tfidf_logreg,
    _fit_tfidf_svm,
    _metrics,
    _naive_predictions,
)
from sentiment_engine.utils.io import write_json

WALK_FORWARD_REPORT_NAME = "walk_forward_report.json"
DEFAULT_MIN_TRAIN_ROWS = 3
DEFAULT_TEST_WINDOW_ROWS = 1
DEFAULT_STEP_ROWS = 1
DEFAULT_EMBARGO_ROWS = 1
DEFAULT_EMBARGO_MINUTES = 30


def evaluate_walk_forward_classifiers(
    labeled_events: pd.DataFrame,
    *,
    report_dir: Path,
    seed: int = 42,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    test_window_rows: int = DEFAULT_TEST_WINDOW_ROWS,
    step_rows: int = DEFAULT_STEP_ROWS,
    embargo_rows: int = DEFAULT_EMBARGO_ROWS,
    embargo_minutes: int = DEFAULT_EMBARGO_MINUTES,
    tfidf_max_features: int = 250,
) -> dict[str, Any]:
    ordered = labeled_events.sort_values(TEMPORAL_COLUMN).reset_index(drop=True)
    folds = _walk_forward_folds(
        ordered,
        min_train_rows=min_train_rows,
        test_window_rows=test_window_rows,
        step_rows=step_rows,
        embargo_rows=embargo_rows,
        embargo_minutes=embargo_minutes,
    )
    model_predictions = {
        "naive": [],
        "rules": [],
        "tfidf_logreg": [],
        "tfidf_linear_svm": [],
    }
    target_labels = sorted(ordered[TARGET_COLUMN].astype(str).unique().tolist())
    fold_reports = []
    for fold_number, train, test in folds:
        y_true = test[TARGET_COLUMN].astype(str).tolist()
        fold_report = {
            "fold_number": fold_number,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start_utc": str(train[TEMPORAL_COLUMN].min()),
            "train_end_utc": str(train[TEMPORAL_COLUMN].max()),
            "test_start_utc": str(test[TEMPORAL_COLUMN].min()),
            "test_end_utc": str(test[TEMPORAL_COLUMN].max()),
        }
        predictions = {
            "naive": _naive_predictions(train[TARGET_COLUMN], len(test)),
            "rules": test["rule_tradeability_label"].astype(str).tolist(),
            "tfidf_logreg": _fit_tfidf_logreg(
                train, test, seed, tfidf_max_features
            )[1],
            "tfidf_linear_svm": _fit_tfidf_svm(train, test, seed, tfidf_max_features)[1],
        }
        for model_name, y_pred in predictions.items():
            model_predictions[model_name].extend(
                {"actual": actual, "predicted": predicted}
                for actual, predicted in zip(y_true, y_pred)
            )
            fold_report[model_name] = _metrics(y_true, y_pred, labels=target_labels)
        fold_reports.append(fold_report)

    report = {
        "status": "evaluated" if folds else "skipped",
        "row_count": int(len(ordered)),
        "fold_count": int(len(folds)),
        "split_method": "expanding_walk_forward_with_row_embargo",
        "min_train_rows": int(min_train_rows),
        "test_window_rows": int(test_window_rows),
        "step_rows": int(step_rows),
        "embargo_rows": int(embargo_rows),
        "embargo_minutes": int(embargo_minutes),
        "target_labels": target_labels,
        "model_summary": _model_summary(model_predictions, fold_reports, target_labels),
        "folds": fold_reports,
        "methodology_notes": [
            "Rows are sorted by received_at_utc before splitting.",
            "Training rows are purged if their maximum target horizon can overlap the test fold.",
            "Fixture fold metrics are validation plumbing checks, not evidence of edge.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / WALK_FORWARD_REPORT_NAME, report)
    return report


def _walk_forward_folds(
    ordered: pd.DataFrame,
    *,
    min_train_rows: int,
    test_window_rows: int,
    step_rows: int,
    embargo_rows: int,
    embargo_minutes: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    folds = []
    fold_number = 1
    train_end = min_train_rows
    timestamps = pd.to_datetime(ordered[TEMPORAL_COLUMN], format="mixed", utc=True)
    embargo_delta = pd.Timedelta(minutes=embargo_minutes)
    while train_end + embargo_rows < len(ordered):
        test_start = train_end + embargo_rows
        test_end = min(test_start + test_window_rows, len(ordered))
        if test_end <= test_start:
            break
        train_cutoff = timestamps.iloc[test_start] - embargo_delta
        train_mask = timestamps.iloc[:train_end] <= train_cutoff
        train = ordered.iloc[:train_end].loc[train_mask.to_numpy()].copy()
        if len(train) < min_train_rows:
            train_end += step_rows
            continue
        folds.append(
            (
                fold_number,
                train,
                ordered.iloc[test_start:test_end].copy(),
            )
        )
        fold_number += 1
        train_end += step_rows
    return folds


def _model_summary(
    model_predictions: dict[str, list[dict[str, str]]],
    fold_reports: list[dict[str, Any]],
    target_labels: list[str],
) -> dict[str, Any]:
    summary = {}
    for model_name, rows in model_predictions.items():
        if not rows:
            summary[model_name] = {"status": "skipped", "reason": "no_fold_predictions"}
            continue
        y_true = [row["actual"] for row in rows]
        y_pred = [row["predicted"] for row in rows]
        aggregate = _metrics(y_true, y_pred, labels=target_labels)
        per_fold_f1 = [
            fold[model_name]["macro_f1"]
            for fold in fold_reports
            if model_name in fold and fold[model_name].get("macro_f1") is not None
        ]
        summary[model_name] = {
            "status": "evaluated",
            "prediction_count": int(len(rows)),
            "aggregate": aggregate,
            "macro_f1_by_fold": per_fold_f1,
            "macro_f1_mean": _round_or_none(np.mean(per_fold_f1)) if per_fold_f1 else None,
            "macro_f1_std": _round_or_none(np.std(per_fold_f1)) if per_fold_f1 else None,
            "macro_f1_min": min(per_fold_f1) if per_fold_f1 else None,
            "macro_f1_max": max(per_fold_f1) if per_fold_f1 else None,
        }
    return summary


def _round_or_none(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
