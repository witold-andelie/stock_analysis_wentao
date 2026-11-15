"""Fixtures: isolated artifacts dir with small fake artifacts matching the contract."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _write(art: Path) -> None:
    today = pd.Timestamp.today().normalize()
    sig = pd.DataFrame({
        "date": [today] * 3, "ticker": ["AAA", "BBB", "CCC"], "company": ["A", "B", "C"],
        "sector": ["Tech"] * 3, "cap_bucket": ["large"] * 3, "event_family": ["earnings", "shock", "earnings"],
        "ret_1d": [0.05, -0.12, 0.01], "prob": [0.7, 0.6, 0.3], "tier": ["STRONG_BUY", "BUY", "SKIP"],
        "suggested_weight": [0.2, None, 0.0], "entry_rule": ["open"] * 3,
        "entry_ref_price": [100.0, 50.0, 10.0], "stop_loss_price": [85.0, 42.5, 8.5],
        "take_profit_price": [125.0, 62.5, 12.5], "exit_by_date": [today + pd.Timedelta(days=14)] * 3,
        "edgar_items": ["2.02"] * 3, "edgar_category": ["x"] * 3, "sue": [1.0, 0.0, 0.0],
        "regime_state": ["NORMAL"] * 3, "model_name": ["m"] * 3,
    })
    sig.to_parquet(art / "signals_latest.parquet", index=False)
    sig.assign(date=today - pd.Timedelta(days=5)).to_parquet(art / "events_recent.parquet", index=False)
    pd.DataFrame({"ticker": ["AAA"] * 70, "date": pd.bdate_range(end=today, periods=70),
                  "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}).to_parquet(
        art / "prices_recent.parquet", index=False)
    pd.DataFrame({"ticker": ["AAA", "BBB"], "company": ["A", "B"], "sector": ["Tech"] * 2,
                  "sub_industry": ["s"] * 2, "cap_bucket": ["large"] * 2, "cik": [1, 2],
                  "wikipedia_title": ["a", "b"]}).to_parquet(art / "universe.parquet", index=False)
    pd.DataFrame({"strategy": ["prob"] * 2, "ticker": ["AAA", "BBB"], "event_family": ["earnings"] * 2,
                  "prob": [0.7, 0.6], "entry_date": [today - pd.Timedelta(days=30)] * 2, "entry_price": [1.0, 1.0],
                  "exit_date": [today - pd.Timedelta(days=20)] * 2, "exit_price": [1.1, 0.9],
                  "ret_net": [0.1, -0.1], "pnl": [10.0, -10.0], "exit_reason": ["time"] * 2}
                 ).to_parquet(art / "trades_backtest.parquet", index=False)
    pd.DataFrame({"date": pd.bdate_range(end=today, periods=30).tolist() * 2,
                  "strategy": ["prob"] * 30 + ["SPY"] * 30, "equity": [10000.0 + i for i in range(60)]}
                 ).to_parquet(art / "equity_curves.parquet", index=False)
    pd.DataFrame({"model": ["m"] * 3, "feature": ["f1", "f2", "f3"], "importance": [0.5, 0.3, 0.2]}
                 ).to_parquet(art / "feature_importance.parquet", index=False)
    pd.DataFrame({"dimension": ["event_family", "tier"], "value": ["earnings", "BUY"], "n": [10, 5],
                  "precision": [0.6, 0.5], "avg_growth": [0.02, 0.01]}).to_parquet(
        art / "precision_breakdown.parquet", index=False)
    (art / "metrics.json").write_text(json.dumps({
        "generated_at": "2025-01-01", "data_through": "2025-01-01", "universe_size": 2, "n_events": 3,
        "model": {"name": "m", "auc": 0.6}, "models": [],
        "strategies": [{"name": "prob", "cagr": 0.2, "sharpe": 1.0, "max_drawdown": -0.1}],
        "benchmarks": [{"name": "SPY", "cagr": 0.1, "sharpe": 0.8, "max_drawdown": -0.2}],
        "regime": {"state": "NORMAL", "points": 0}}))
    (art / "run_log.jsonl").write_text(json.dumps({"started": "a", "finished": "b", "status": "ok", "steps": []}) + "\n")


@pytest.fixture
def art_dir(tmp_path, monkeypatch):
    from driftrev import config

    art = tmp_path / "artifacts"
    art.mkdir()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", art)
    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "DRIFTREV_ALLOW_ORDERS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DRIFTREV_API_TOKEN", "secret")
    _write(art)
    return art


@pytest.fixture
def empty_art_dir(tmp_path, monkeypatch):
    from driftrev import config

    art = tmp_path / "empty"
    art.mkdir()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", art)
    monkeypatch.setenv("DRIFTREV_API_TOKEN", "secret")
    return art


@pytest.fixture
def client(art_dir):
    from fastapi.testclient import TestClient

    from driftrev.api import app

    c = TestClient(app)
    c.headers.update({"Authorization": "Bearer secret"})
    return c
