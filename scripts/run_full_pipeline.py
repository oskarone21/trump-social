from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_engine.cli import main


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/research.yaml"
    main(["--config", config_path, "run-full"])
