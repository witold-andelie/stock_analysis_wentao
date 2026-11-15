"""The Streamlit dashboard renders every tab without exceptions (fixture artifacts and empty store)."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_renders_with_artifacts(art_dir):
    (art_dir / "metrics.json").write_text(json.dumps({
        "data_through": "2026-11-13", "universe_size": 3, "regime": {"state": "normal"},
        "strategies": [{"name": "model_top5", "cagr": 0.1, "sharpe": 1.0, "max_drawdown": -0.1,
                        "report_card": {"profit_factor": 1.5}}],
        "benchmarks": [{"name": "SPY_buy_hold", "cagr": 0.12, "sharpe": 1.1, "max_drawdown": -0.2}],
        "guide": {"verdict": "release does NOT add value"}, "walkforward": {"cagr": 0.08}}))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 8


def test_dashboard_renders_empty(empty_art_dir):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
