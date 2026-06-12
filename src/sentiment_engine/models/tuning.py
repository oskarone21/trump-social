from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from sentiment_engine.utils.io import write_json

COMPONENT_COLUMNS = [
    "headline_risk_score",
    "market_relevance_score",
    "text_contradiction_score",
    "post_burst_score",
    "direction_flip_score",
    "same_topic_score",
    "volatility_regime_score",
]

WEIGHT_NAMES = [
    "headline_risk",
    "market_relevance",
    "text_contradiction",
    "post_burst",
    "direction_flip",
    "same_topic",
    "volatility_regime",
]


@dataclass(frozen=True)
class WhipsawParams:
    weights: dict[str, float]
    soft_threshold: float
    hard_threshold: float


def tune_whipsaw_parameters(
    scored_events: pd.DataFrame,
    *,
    report_path: str | Path,
    n_trials: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    _validate_input(scored_events)
    ordered = scored_events.sort_values("received_at_utc").reset_index(drop=True).copy()
    split_index = max(1, int(len(ordered) * 0.65))
    train = ordered.iloc[:split_index].copy()
    holdout = ordered.iloc[split_index:].copy()
    if holdout.empty:
        raise ValueError("Whipsaw tuning requires at least one chronological holdout row")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: _objective(trial, train), n_trials=n_trials)

    best = _params_from_trial(study.best_trial)
    default = WhipsawParams(
        weights={
            "headline_risk": 0.35,
            "market_relevance": 0.30,
            "text_contradiction": 0.15,
            "post_burst": 0.10,
            "direction_flip": 0.07,
            "same_topic": 0.03,
            "volatility_regime": 0.00,
        },
        soft_threshold=0.55,
        hard_threshold=0.75,
    )
    report = {
        "status": "completed",
        "optimizer": "optuna.tpe",
        "n_trials": n_trials,
        "seed": seed,
        "row_count": int(len(ordered)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split": {
            "train_end_utc": str(train["received_at_utc"].max()),
            "holdout_start_utc": str(holdout["received_at_utc"].min()),
            "method": "time_ordered_holdout",
        },
        "best_value_train": round(float(study.best_value), 6),
        "best_params": {
            "weights": best.weights,
            "soft_whipsaw_threshold": best.soft_threshold,
            "hard_whipsaw_threshold": best.hard_threshold,
        },
        "default_params": {
            "weights": default.weights,
            "soft_whipsaw_threshold": default.soft_threshold,
            "hard_whipsaw_threshold": default.hard_threshold,
        },
        "train_metrics_best": _evaluate(train, best),
        "holdout_metrics_best": _evaluate(holdout, best),
        "holdout_metrics_default": _evaluate(holdout, default),
        "methodology_notes": [
            "Optuna tuning uses only existing live-safe whipsaw component scores.",
            "The chronological holdout is tiny because the repo uses fixtures; do not promote these params.",
            "Tuned params are reported only and are not written back into live configuration.",
        ],
    }
    write_json(report_path, report)
    return report


def _objective(trial: optuna.Trial, train: pd.DataFrame) -> float:
    params = _params_from_trial(trial)
    metrics = _evaluate(train, params)
    soft = metrics["soft_risk"]
    hard = metrics["hard_kill"]
    prediction_rate = metrics["risk_prediction_rate"]
    return (
        0.40 * soft["recall"]
        + 0.25 * soft["precision"]
        + 0.20 * hard["precision"]
        + 0.10 * hard["recall"]
        - 0.05 * max(0.0, prediction_rate - 0.60)
    )


def _params_from_trial(trial: optuna.Trial) -> WhipsawParams:
    raw_weights = {
        name: trial.suggest_float(f"weight_{name}", 0.0, 1.0)
        for name in WEIGHT_NAMES
    }
    total = sum(raw_weights.values())
    if total <= 0:
        weights = {name: 1.0 / len(raw_weights) for name in raw_weights}
    else:
        weights = {name: round(value / total, 6) for name, value in raw_weights.items()}
    soft_threshold = trial.suggest_float("soft_whipsaw_threshold", 0.20, 0.80)
    hard_threshold = trial.suggest_float(
        "hard_whipsaw_threshold",
        min(0.95, soft_threshold + 0.05),
        0.95,
    )
    return WhipsawParams(
        weights=weights,
        soft_threshold=round(float(soft_threshold), 6),
        hard_threshold=round(float(hard_threshold), 6),
    )


def _evaluate(frame: pd.DataFrame, params: WhipsawParams) -> dict[str, Any]:
    scores = _scores(frame, params)
    actual = frame["market_whipsaw_flag"].astype(bool)
    soft_pred = scores >= params.soft_threshold
    hard_pred = scores >= params.hard_threshold
    return {
        "soft_risk": _binary_metrics(actual, soft_pred),
        "hard_kill": _binary_metrics(actual, hard_pred),
        "risk_prediction_rate": round(float(soft_pred.mean()), 6),
        "mean_score_actual_true": _safe_mean(scores[actual]),
        "mean_score_actual_false": _safe_mean(scores[~actual]),
    }


def _scores(frame: pd.DataFrame, params: WhipsawParams) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for weight_name, component_column in zip(WEIGHT_NAMES, COMPONENT_COLUMNS, strict=True):
        score = score + params.weights[weight_name] * frame[component_column].astype(float)
    return score.clip(lower=0.0, upper=1.0)


def _binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    true_positive = int((actual & predicted).sum())
    false_positive = int((~actual & predicted).sum())
    true_negative = int((~actual & ~predicted).sum())
    false_negative = int((actual & ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    }


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.mean()), 6)


def _validate_input(scored_events: pd.DataFrame) -> None:
    missing = [column for column in [*COMPONENT_COLUMNS, "market_whipsaw_flag"] if column not in scored_events]
    if missing:
        raise ValueError(f"Cannot tune whipsaw params; missing columns: {missing}")
