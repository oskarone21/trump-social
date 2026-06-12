from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_engine.config import ensure_output_dirs, load_config
from sentiment_engine.research.shadow import write_shadow_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a shadow/advisory report from generated signal and backtest artefacts."
    )
    parser.add_argument("config", nargs="?", default="configs/research.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    report = write_shadow_report(
        report_dir=config.paths.report_dir,
        latest_signal=_read_json(config.paths.report_dir / "latest_signal.json"),
        backtest_report=_read_json(config.paths.report_dir / "backtest_report.json"),
        backtest_trades=pd.read_csv(config.paths.report_dir / "backtest_event_audit.csv"),
        interpretation_report=_read_optional_json(
            config.paths.report_dir / "research_interpretation.json"
        ),
    )
    print(
        "shadow report complete: "
        f"status={report['status']}, trades={report['trade_summary']['trade_count']}"
    )
    print(f"shadow report: {config.paths.report_dir / 'shadow_report.json'}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _read_json(path)


if __name__ == "__main__":
    main()
