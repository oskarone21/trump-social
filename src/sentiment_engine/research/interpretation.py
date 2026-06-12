from __future__ import annotations

from pathlib import Path
from typing import Any

from sentiment_engine.utils.io import write_json

MIN_HUMAN_LABELS_FOR_DL = 500
FIXTURE_SOURCE_PREFIX = "fixture"
INTERPRETATION_JSON = "research_interpretation.json"
INTERPRETATION_MARKDOWN = "research_interpretation.md"


def build_interpretation_report(reports: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    model_rows = _model_rows(reports.get("classifier"))
    gates = _readiness_gates(reports)
    return {
        "status": "real_research_ready" if all(gates.values()) else "not_research_ready",
        "model_comparison": model_rows,
        "best_fixture_model": _best_model(model_rows),
        "data_quality": _data_quality(reports),
        "whipsaw_summary": _whipsaw_summary(reports.get("whipsaw")),
        "backtest_summary": _backtest_summary(reports.get("backtest")),
        "readiness_gates": gates,
        "interpretation": _interpretation(model_rows, reports, gates),
        "methodology_notes": [
            "Current metrics are fixture smoke checks unless readiness gates are true.",
            "Model comparison uses the temporal holdout reported by classifier training.",
            "No live-trading readiness is implied without licensed data and shadow validation.",
        ],
    }


def write_interpretation_report(
    *, reports: dict[str, dict[str, Any] | None], report_dir: Path
) -> dict[str, Any]:
    payload = build_interpretation_report(reports)
    write_json(report_dir / INTERPRETATION_JSON, payload)
    (report_dir / INTERPRETATION_MARKDOWN).write_text(
        render_interpretation_markdown(payload),
        encoding="utf-8",
    )
    return payload


def render_interpretation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Truth Social NQ Research Interpretation",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Model Comparison",
        "",
        "| Model | Type | Status | Accuracy | Macro F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["model_comparison"]:
        lines.append(
            f"| {row['model']} | {row['model_type']} | {row['status']} | "
            f"{_fmt(row.get('accuracy'))} | {_fmt(row.get('macro_f1'))} |"
        )
    lines.extend(
        [
            "",
            "## Readiness Gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    for gate, passed in payload["readiness_gates"].items():
        lines.append(f"| {gate} | {passed} |")
    lines.extend(["", "## Interpretation", ""])
    for item in payload["interpretation"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Backtest Summary", ""])
    for key, value in payload["backtest_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Methodology Notes", ""])
    for note in payload["methodology_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _model_rows(classifier_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not classifier_report:
        return []
    specs = [
        ("naive", "Naive majority", "baseline"),
        ("rules", "Rules", "rules"),
        ("tfidf_logreg", "TF-IDF logistic regression", "ml"),
        ("tfidf_linear_svm", "TF-IDF linear SVM", "ml"),
        ("lightgbm", "LightGBM text/context", "ml"),
        ("neural_tfidf_mlp", "PyTorch TF-IDF MLP", "dl_smoke"),
    ]
    rows = []
    for key, name, model_type in specs:
        raw = classifier_report.get(key)
        if not raw:
            continue
        metrics = raw.get("metrics", raw)
        status = raw.get("status", "trained")
        classification = metrics.get("classification_report", {})
        rows.append(
            {
                "key": key,
                "model": name,
                "model_type": model_type,
                "status": status,
                "accuracy": _round_or_none(classification.get("accuracy")),
                "macro_f1": _round_or_none(metrics.get("macro_f1")),
            }
        )
    return rows


def _best_model(model_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    trained = [
        row for row in model_rows if row["status"] == "trained" and row["macro_f1"] is not None
    ]
    if not trained:
        return None
    return max(trained, key=lambda row: row["macro_f1"])


def _data_quality(reports: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    post = reports.get("post") or {}
    archive = reports.get("archive") or {}
    market = reports.get("market") or {}
    labels = reports.get("human_labels") or {}
    return {
        "fixture_post_rows": post.get("row_count"),
        "archive_post_rows": archive.get("row_count"),
        "archive_empty_text_rows": archive.get("empty_text_rows"),
        "market_rows": market.get("row_count"),
        "market_valid_rows": market.get("valid_rows"),
        "market_source_names": market.get("source_names", []),
        "human_label_rows": labels.get("row_count", 0),
        "human_label_unique_events": labels.get("unique_event_count", 0),
    }


def _whipsaw_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "row_count": report.get("row_count"),
        "soft_precision": (report.get("soft_risk") or {}).get("precision"),
        "soft_recall": (report.get("soft_risk") or {}).get("recall"),
        "hard_precision": (report.get("hard_kill") or {}).get("precision"),
        "hard_recall": (report.get("hard_kill") or {}).get("recall"),
    }


def _backtest_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "trade_count": report.get("trade_count"),
        "net_pnl_before_usd": report.get("net_pnl_before_usd"),
        "net_pnl_after_usd": report.get("net_pnl_after_usd"),
        "kill_switch_value_usd": report.get("kill_switch_value_usd"),
        "avoided_losing_trade_usd": report.get("avoided_losing_trade_usd"),
        "missed_winning_trade_usd": report.get("missed_winning_trade_usd"),
    }


def _readiness_gates(reports: dict[str, dict[str, Any] | None]) -> dict[str, bool]:
    archive = reports.get("archive") or {}
    market = reports.get("market") or {}
    labels = reports.get("human_labels") or {}
    source_names = [str(item) for item in market.get("source_names", [])]
    licensed_market_loaded = bool(source_names) and not any(
        item.startswith(FIXTURE_SOURCE_PREFIX) for item in source_names
    )
    human_label_minimum_met = int(labels.get("row_count", 0) or 0) >= MIN_HUMAN_LABELS_FOR_DL
    return {
        "real_post_archive_verified": int(archive.get("row_count", 0) or 0) >= 1000,
        "licensed_market_data_loaded": licensed_market_loaded,
        "human_label_minimum_met": human_label_minimum_met,
        "transformer_training_ready": licensed_market_loaded and human_label_minimum_met,
    }


def _interpretation(
    model_rows: list[dict[str, Any]],
    reports: dict[str, dict[str, Any] | None],
    gates: dict[str, bool],
) -> list[str]:
    notes = []
    best = _best_model(model_rows)
    classifier = reports.get("classifier") or {}
    test_rows = classifier.get("test_rows")
    if best:
        notes.append(
            f"Best fixture holdout macro F1 is {best['macro_f1']} from {best['model']} "
            f"on {test_rows} test rows."
        )
    if not gates["licensed_market_data_loaded"]:
        notes.append("Licensed, archive-covering NQ/MNQ market data is not loaded.")
    if not gates["human_label_minimum_met"]:
        notes.append(
            f"Human labels are below the {MIN_HUMAN_LABELS_FOR_DL} row transformer gate."
        )
    if not gates["transformer_training_ready"]:
        notes.append("FinBERT/DeBERTa fine-tuning is not methodologically ready.")
    notes.append(
        "Fixture backtest PnL changes are accounting checks only and must not be treated as edge."
    )
    return notes


def _round_or_none(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
