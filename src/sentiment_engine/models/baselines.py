from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sentiment_engine.features.text_features import enrich_with_text_features
from sentiment_engine.models.metadata import model_metadata
from sentiment_engine.utils.io import write_dataframe, write_json

TARGET_COLUMN = "tradeability_label"
TEXT_COLUMN = "text_clean"
TEMPORAL_COLUMN = "received_at_utc"
TFIDF_MODEL_VERSION = "tfidf_logreg_v1"
SVM_MODEL_VERSION = "tfidf_linear_svm_v1"
LIGHTGBM_MODEL_VERSION = "lightgbm_text_context_v1"
NEURAL_MODEL_VERSION = "neural_tfidf_mlp_v1"
CLASSIFIER_REPORT_NAME = "classifier_baseline_report.json"
LIGHTGBM_REPORT_NAME = "lightgbm_baseline_report.json"
NEURAL_REPORT_NAME = "neural_baseline_report.json"
TFIDF_MODEL_NAME = "tfidf_tradeability_baseline.joblib"
SVM_MODEL_NAME = "tfidf_linear_svm_tradeability_baseline.joblib"
LIGHTGBM_MODEL_NAME = "lightgbm_tradeability_baseline.joblib"
NEURAL_MODEL_NAME = "neural_tfidf_mlp.pt"
CONTEXT_FEATURE_COLUMNS = [
    "post_length",
    "token_count",
    "all_caps_ratio",
    "exclamation_count",
    "question_mark_count",
    "has_image",
    "has_video",
    "rule_sentiment_confidence",
]
NEURAL_HIDDEN_UNITS = 16
NEURAL_EPOCHS = 40
NEURAL_LEARNING_RATE = 0.03
ABSTENTION_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


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
    tfidf_model, tfidf_predictions, tfidf_probabilities, tfidf_probability_labels = (
        _fit_tfidf_logreg(train, test, seed, tfidf_max_features)
    )
    svm_model, svm_predictions, svm_probabilities, svm_probability_labels = _fit_tfidf_svm(
        train, test, seed, tfidf_max_features
    )
    lightgbm_report = train_lightgbm_baseline(
        train,
        test,
        report_dir=report_dir,
        model_dir=model_dir,
        config_path=config_path,
        seed=seed,
        tfidf_max_features=tfidf_max_features,
        all_labeled_events=ordered,
    )
    neural_report = train_neural_text_baseline(
        train,
        test,
        report_dir=report_dir,
        model_dir=model_dir,
        config_path=config_path,
        seed=seed,
        tfidf_max_features=tfidf_max_features,
        all_labeled_events=ordered,
    )

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
        "tfidf_logreg": _metrics(
            test[TARGET_COLUMN].tolist(),
            tfidf_predictions,
            y_proba=tfidf_probabilities,
            proba_labels=tfidf_probability_labels,
        ),
        "tfidf_linear_svm": _metrics(
            test[TARGET_COLUMN].tolist(),
            svm_predictions,
            y_proba=svm_probabilities,
            proba_labels=svm_probability_labels,
        ),
        "lightgbm": lightgbm_report,
        "neural_tfidf_mlp": neural_report,
        "methodology_notes": [
            "Fixture metrics are engineering smoke checks, not statistical evidence.",
            "Rows are split in timestamp order; no random headline split is reported.",
            "Labels are weak research labels until human adjudication exists.",
            "LightGBM and neural MLP reports are included only when optional dependencies "
            "are installed.",
            "Probability metrics are reported only for models that expose class "
            "probabilities on the holdout set.",
        ],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / CLASSIFIER_REPORT_NAME, report)
    write_dataframe(labeled_events, report_dir / "labeled_events_snapshot.csv")
    if tfidf_model is not None:
        joblib.dump(tfidf_model, model_dir / TFIDF_MODEL_NAME)
    if svm_model is not None:
        joblib.dump(svm_model, model_dir / SVM_MODEL_NAME)
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
    write_json(
        model_dir / "tfidf_linear_svm_tradeability_baseline.metadata.json",
        model_metadata(
            model_name="tradeability_linear_svm_baseline",
            model_version=SVM_MODEL_VERSION,
            config_path=config_path,
            data=labeled_events,
            extra={"target": TARGET_COLUMN, "temporal_test_fraction": temporal_test_fraction},
        ),
    )
    return report


def train_lightgbm_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    report_dir: Path,
    model_dir: Path,
    config_path: str,
    seed: int,
    tfidf_max_features: int,
    all_labeled_events: pd.DataFrame,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError:
        report = _skipped_report("lightgbm_missing", "Install the optional ml dependency group.")
        write_json(report_dir / LIGHTGBM_REPORT_NAME, report)
        return report

    if train[TARGET_COLUMN].nunique() < 2:
        report = _skipped_report(
            "insufficient_training_classes", "Need at least two training labels."
        )
        write_json(report_dir / LIGHTGBM_REPORT_NAME, report)
        return report

    label_encoder = _label_encoder(all_labeled_events)
    train_features, test_features, vectorizer = _text_context_matrices(
        train, test, tfidf_max_features
    )
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        min_data_in_leaf=1,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(train_features, label_encoder.transform(train[TARGET_COLUMN].astype(str)))
    predicted_indexes = model.predict(test_features)
    predictions = label_encoder.inverse_transform(predicted_indexes).tolist()
    probabilities = model.predict_proba(test_features)
    report = {
        "status": "trained",
        "model_version": LIGHTGBM_MODEL_VERSION,
        "feature_view": "tfidf_text_plus_live_time_text_context",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "metrics": _metrics(
            test[TARGET_COLUMN].astype(str).tolist(),
            predictions,
            y_proba=probabilities,
            proba_labels=label_encoder.classes_.tolist(),
        ),
        "methodology_notes": [
            "This baseline excludes post-event target columns to avoid look-ahead leakage.",
            "Fixture metrics are a smoke check only; real comparison needs chronological "
            "validation.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / LIGHTGBM_REPORT_NAME, report)
    joblib.dump(
        {"model": model, "label_encoder": label_encoder, "vectorizer": vectorizer},
        model_dir / LIGHTGBM_MODEL_NAME,
    )
    write_json(
        model_dir / "lightgbm_tradeability_baseline.metadata.json",
        model_metadata(
            model_name="tradeability_lightgbm_baseline",
            model_version=LIGHTGBM_MODEL_VERSION,
            config_path=config_path,
            data=all_labeled_events,
            extra={"target": TARGET_COLUMN, "feature_columns": CONTEXT_FEATURE_COLUMNS},
        ),
    )
    return report


def train_neural_text_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    report_dir: Path,
    model_dir: Path,
    config_path: str,
    seed: int,
    tfidf_max_features: int,
    all_labeled_events: pd.DataFrame,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError:
        report = _skipped_report("torch_missing", "Install the optional dl dependency group.")
        write_json(report_dir / NEURAL_REPORT_NAME, report)
        return report

    if train[TARGET_COLUMN].nunique() < 2:
        report = _skipped_report(
            "insufficient_training_classes", "Need at least two training labels."
        )
        write_json(report_dir / NEURAL_REPORT_NAME, report)
        return report

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    label_encoder = _label_encoder(all_labeled_events)
    train_features, test_features, vectorizer = _text_context_matrices(
        train, test, tfidf_max_features
    )
    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(
        label_encoder.transform(train[TARGET_COLUMN].astype(str)), dtype=torch.long
    )
    x_test = torch.tensor(test_features, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(train_features.shape[1], NEURAL_HIDDEN_UNITS),
        nn.ReLU(),
        nn.Linear(NEURAL_HIDDEN_UNITS, len(label_encoder.classes_)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(NEURAL_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        probabilities = torch.softmax(logits, dim=1).numpy()
        predicted_indexes = torch.argmax(logits, dim=1).numpy()
    predictions = label_encoder.inverse_transform(predicted_indexes).tolist()
    report = {
        "status": "trained",
        "model_version": NEURAL_MODEL_VERSION,
        "architecture": {
            "input": "tfidf_text_plus_live_time_text_context",
            "hidden_units": NEURAL_HIDDEN_UNITS,
            "output_classes": label_encoder.classes_.tolist(),
        },
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "epochs": NEURAL_EPOCHS,
        "learning_rate": NEURAL_LEARNING_RATE,
        "metrics": _metrics(
            test[TARGET_COLUMN].astype(str).tolist(),
            predictions,
            y_proba=probabilities,
            proba_labels=label_encoder.classes_.tolist(),
        ),
        "methodology_notes": [
            "This is a PyTorch MLP smoke baseline, not a trained FinBERT or DeBERTa model.",
            "It uses a temporal holdout and excludes post-event target columns.",
            "Do not interpret fixture performance as economic or statistical evidence.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / NEURAL_REPORT_NAME, report)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_classes": label_encoder.classes_.tolist(),
            "vectorizer_vocabulary": vectorizer.vocabulary_,
            "context_feature_columns": CONTEXT_FEATURE_COLUMNS,
        },
        model_dir / NEURAL_MODEL_NAME,
    )
    write_json(
        model_dir / "neural_tfidf_mlp.metadata.json",
        model_metadata(
            model_name="tradeability_neural_tfidf_mlp",
            model_version=NEURAL_MODEL_VERSION,
            config_path=config_path,
            data=all_labeled_events,
            extra={
                "target": TARGET_COLUMN,
                "feature_columns": CONTEXT_FEATURE_COLUMNS,
                "epochs": NEURAL_EPOCHS,
                "seed": seed,
            },
        ),
    )
    return report


def _fit_tfidf_logreg(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, tfidf_max_features: int
) -> tuple[Pipeline | None, list[str], np.ndarray | None, list[str] | None]:
    unique_labels = train[TARGET_COLUMN].nunique()
    if unique_labels < 2:
        return None, _naive_predictions(train[TARGET_COLUMN], len(test)), None, None
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
    probabilities = model.predict_proba(test[TEXT_COLUMN].astype(str))
    probability_labels = model.named_steps["classifier"].classes_.tolist()
    return model, predictions, probabilities, probability_labels


def _fit_tfidf_svm(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, tfidf_max_features: int
) -> tuple[Pipeline | None, list[str], np.ndarray | None, list[str] | None]:
    unique_labels = train[TARGET_COLUMN].nunique()
    if unique_labels < 2:
        return None, _naive_predictions(train[TARGET_COLUMN], len(test)), None, None
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(train[TEXT_COLUMN].astype(str), train[TARGET_COLUMN].astype(str))
    predictions = model.predict(test[TEXT_COLUMN].astype(str)).tolist()
    probabilities = model.predict_proba(test[TEXT_COLUMN].astype(str))
    probability_labels = model.named_steps["classifier"].classes_.tolist()
    return model, predictions, probabilities, probability_labels


def _text_context_matrices(
    train: pd.DataFrame, test: pd.DataFrame, tfidf_max_features: int
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
    train_text = vectorizer.fit_transform(train[TEXT_COLUMN].astype(str)).toarray()
    test_text = vectorizer.transform(test[TEXT_COLUMN].astype(str)).toarray()
    train_context = _context_matrix(train)
    test_context = _context_matrix(test)
    return (
        np.hstack([train_text, train_context]).astype(np.float32),
        np.hstack([test_text, test_context]).astype(np.float32),
        vectorizer,
    )


def _context_matrix(frame: pd.DataFrame) -> np.ndarray:
    context = frame[CONTEXT_FEATURE_COLUMNS].copy()
    for column in ("has_image", "has_video"):
        context[column] = context[column].astype(int)
    return context.fillna(0).astype(float).to_numpy()


def _label_encoder(labeled_events: pd.DataFrame) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(sorted(labeled_events[TARGET_COLUMN].astype(str).unique().tolist()))
    return encoder


def _skipped_report(reason: str, remediation: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "remediation": remediation,
        "methodology_notes": [
            "Skipping is explicit so missing optional model dependencies do not silently "
            "look trained."
        ],
    }


def _naive_predictions(labels: pd.Series, count: int) -> list[str]:
    if labels.empty:
        raise ValueError("Cannot build naive baseline without training labels")
    majority = labels.value_counts().index[0]
    return [str(majority)] * count


def _metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    y_proba: np.ndarray | None = None,
    proba_labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = sorted(set(y_true).union(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            6,
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": {
            "labels": labels,
            "rows": matrix.tolist(),
        },
    }
    if y_proba is not None and proba_labels is not None:
        metrics["probability_metrics"] = _probability_metrics(
            y_true, y_pred, y_proba, proba_labels
        )
    return metrics


def _probability_metrics(
    y_true: list[str], y_pred: list[str], y_proba: np.ndarray, proba_labels: list[str]
) -> dict[str, Any]:
    aligned, labels = _align_probabilities(y_true, y_proba, proba_labels)
    true_indexes = np.array([labels.index(label) for label in y_true])
    one_hot = np.zeros_like(aligned)
    one_hot[np.arange(len(y_true)), true_indexes] = 1.0
    label_indexes = {label: index for index, label in enumerate(labels)}
    predicted_indexes = np.array([label_indexes.get(label, -1) for label in y_pred])
    confidences = np.array(
        [
            aligned[index, predicted_index] if predicted_index >= 0 else 0.0
            for index, predicted_index in enumerate(predicted_indexes)
        ]
    )
    correctness = (predicted_indexes == true_indexes).astype(float)
    brier_score = float(np.mean(np.sum((aligned - one_hot) ** 2, axis=1)))
    calibration_error = _expected_calibration_error(confidences, correctness)
    return {
        "multiclass_brier_score": round(brier_score, 6),
        "negative_log_loss": round(float(log_loss(y_true, aligned, labels=labels)), 6),
        "expected_calibration_error": round(float(calibration_error), 6),
        "class_labels": labels,
        "abstention_curve": _abstention_curve(y_true, y_pred, aligned, labels),
        "threshold_note": (
            "Confidence-threshold rows are holdout diagnostics. Promote thresholds only "
            "from validation folds with enough human-reviewed labels."
        ),
    }


def _align_probabilities(
    y_true: list[str], y_proba: np.ndarray, proba_labels: list[str]
) -> tuple[np.ndarray, list[str]]:
    labels = sorted(set(y_true).union(proba_labels))
    aligned = np.zeros((len(y_true), len(labels)), dtype=float)
    source_indexes = {label: index for index, label in enumerate(proba_labels)}
    for target_index, label in enumerate(labels):
        source_index = source_indexes.get(label)
        if source_index is not None:
            aligned[:, target_index] = y_proba[:, source_index]
    row_sums = aligned.sum(axis=1)
    missing = row_sums <= 0
    if missing.any():
        aligned[missing, :] = 1.0 / len(labels)
        row_sums = aligned.sum(axis=1)
    return aligned / row_sums[:, None], labels


def _expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10
) -> float:
    if len(confidences) == 0:
        return 0.0
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        upper = lower + 1.0 / n_bins
        if upper >= 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        confidence_gap = abs(float(confidences[mask].mean()) - float(correctness[mask].mean()))
        ece += float(mask.mean()) * confidence_gap
    return ece


def _abstention_curve(
    y_true: list[str], y_pred: list[str], aligned_probabilities: np.ndarray, labels: list[str]
) -> list[dict[str, Any]]:
    predicted_labels = list(y_pred)
    label_indexes = {label: index for index, label in enumerate(labels)}
    confidences = np.array(
        [
            aligned_probabilities[index, label_indexes[predicted]]
            if predicted in label_indexes
            else 0.0
            for index, predicted in enumerate(predicted_labels)
        ]
    )
    rows = []
    for threshold in ABSTENTION_THRESHOLDS:
        retained = confidences >= threshold
        retained_true = [label for index, label in enumerate(y_true) if retained[index]]
        retained_pred = [
            label for index, label in enumerate(predicted_labels) if retained[index]
        ]
        rows.append(
            {
                "confidence_threshold": threshold,
                "retained_count": int(retained.sum()),
                "abstained_count": int((~retained).sum()),
                "coverage": round(float(retained.mean()), 6),
                "accuracy": _retained_accuracy(retained_true, retained_pred),
                "macro_f1": _retained_macro_f1(retained_true, retained_pred),
                "predicted_label_counts": _label_counts(retained_pred),
                "true_label_counts": _label_counts(retained_true),
            }
        )
    return rows


def _retained_accuracy(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true:
        return None
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    return round(correct / len(y_true), 6)


def _retained_macro_f1(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true:
        return None
    labels = sorted(set(y_true).union(y_pred))
    return round(
        float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        6,
    )


def _label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
