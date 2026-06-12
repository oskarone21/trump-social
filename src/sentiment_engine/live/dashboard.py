from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_engine.schemas import SignalRecord
from sentiment_engine.utils.io import write_json


def build_dashboard(
    *,
    signal: SignalRecord,
    whipsaw_report: dict[str, Any],
    backtest_report: dict[str, Any],
    scored_events: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "latest_signal.json", signal.model_dump(mode="json"))
    rows = "\n".join(_event_row(row) for row in scored_events.tail(10).to_dict("records"))
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Truth Social NQ Risk Dashboard</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
    body {{ margin: 0; background: #f5f7fa; color: #1b1f24; }}
    header {{ background: #101820; color: #fff; padding: 24px 32px; }}
    main {{ padding: 24px 32px; max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .label {{ color: #57606a; font-size: 13px; }}
    .risk-HARD_KILL {{ color: #b42318; }}
    .risk-SOFT_RISK {{ color: #b54708; }}
    .risk-WATCH {{ color: #175cd3; }}
    .risk-NONE {{ color: #067647; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8dee4; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #d8dee4; text-align: left; font-size: 13px; }}
    th {{ background: #eef2f6; }}
    code {{ background: #eef2f6; padding: 2px 4px; border-radius: 4px; }}
    .post {{ max-width: 520px; }}
  </style>
</head>
<body>
  <header>
    <h1>Truth Social NQ Risk Dashboard</h1>
    <div>Fixture-backed research monitor. Shadow/advisory mode only.</div>
  </header>
  <main>
    <section class="grid">
      {_card("Latest Action", signal.kill_switch["action"])}
      {_card("Direction", signal.direction_signal)}
      {_card("Risk Level", signal.risk["whipsaw_risk_level"], f"risk-{signal.risk['whipsaw_risk_level']}")}
      {_card("Whipsaw Score", f"{signal.risk['whipsaw_score']:.3f}")}
      {_card("Backtest Delta", f"${backtest_report['kill_switch_value_usd']:.2f}")}
      {_card("Filtered / Reduced", f"{backtest_report['filtered_trades']} / {backtest_report['reduced_trades']}")}
    </section>
    <h2>Latest Signal</h2>
    <div class="card">
      <div class="label">Post</div>
      <p>{html.escape(signal.text_clean)}</p>
      <div><code>{html.escape(signal.explanation["human_readable_reason"])}</code></div>
    </div>
    <h2>Whipsaw Evaluation</h2>
    <section class="grid">
      {_card("Soft Precision", f"{whipsaw_report['soft_risk']['precision']:.3f}")}
      {_card("Soft Recall", f"{whipsaw_report['soft_risk']['recall']:.3f}")}
      {_card("Hard Precision", f"{whipsaw_report['hard_kill']['precision']:.3f}")}
      {_card("Hard Recall", f"{whipsaw_report['hard_kill']['recall']:.3f}")}
    </section>
    <h2>Recent Events</h2>
    <table>
      <thead>
        <tr><th>Post ID</th><th>Risk</th><th>Score</th><th>Topics</th><th class="post">Text</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output


def _card(label: str, value: str, class_name: str = "") -> str:
    safe_label = html.escape(label)
    safe_value = html.escape(str(value))
    return f'<div class="card"><div class="label">{safe_label}</div><div class="metric {class_name}">{safe_value}</div></div>'


def _event_row(row: dict[str, Any]) -> str:
    topics = row.get("rule_topic_labels", [])
    if hasattr(topics, "tolist"):
        topics = topics.tolist()
    return (
        "<tr>"
        f"<td>{html.escape(str(row['post_id']))}</td>"
        f"<td class=\"risk-{html.escape(str(row['whipsaw_risk_level']))}\">{html.escape(str(row['whipsaw_risk_level']))}</td>"
        f"<td>{float(row['whipsaw_score']):.3f}</td>"
        f"<td>{html.escape(', '.join(str(item) for item in topics))}</td>"
        f"<td class=\"post\">{html.escape(str(row['text_clean']))}</td>"
        "</tr>"
    )
