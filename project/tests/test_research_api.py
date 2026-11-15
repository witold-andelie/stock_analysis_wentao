"""GET /v1/research + MCP/tool contract for the research tables."""

import json

import pandas as pd
from fastapi.testclient import TestClient

from driftrev import agent_tools, api

H = {"Authorization": "Bearer secret"}


def test_research_lists_and_returns_tables(art_dir):
    (art_dir / "metrics.json").write_text(json.dumps({"walkforward": {"cagr": 0.12}, "guide": {"verdict": "x"}}))
    pd.DataFrame({"variant": ["pooled", "cross_sectional"], "val_auc": [0.54, 0.56],
                  "bucket": pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])}).to_parquet(art_dir / "model_comparison.parquet")
    c = TestClient(api.app)
    listing = c.get("/v1/research", headers=H).json()
    assert "model_comparison" in listing["available_tables"] and listing["walkforward"]["cagr"] == 0.12
    rows = c.get("/v1/research", params={"table": "model_comparison"}, headers=H).json()
    assert rows["count"] == 2 and rows["rows"][1]["variant"] == "cross_sectional"
    assert c.get("/v1/research", params={"table": "nope"}, headers=H).status_code == 400
    missing = c.get("/v1/research", params={"table": "walkforward"}, headers=H).json()
    assert missing["count"] == 0 and "not produced" in missing["message"]
    assert c.get("/v1/research").status_code in (401, 403)


def test_family_prefix_filter(art_dir):
    c = TestClient(api.app)
    out = c.get("/v1/events/recent", params={"event_family": "earn", "days": 365}, headers=H).json()
    assert out["count"] >= 1


def test_tool_enum_matches_real_families():
    spec = {t["function"]["name"]: t["function"] for t in agent_tools.TOOLS}
    fams = spec["get_latest_signals"]["parameters"]["properties"]["event_family"]["enum"]
    assert set(fams) == {"earnings_up", "earnings_down", "earnings_crash", "shock_crash", "momentum_breakout"}
    assert agent_tools.ROUTES["get_research"] == {"method": "GET", "path": "/v1/research"}


def test_backtest_summary_with_nan_metrics_is_valid_json(art_dir):
    """Regression (audit F4): NaN in metrics.json made /v1/backtest/summary return HTTP 500."""
    (art_dir / "metrics.json").write_text(
        '{"strategies": [{"name": "s", "cagr": 0.1, "sharpe": NaN, "report_card": {"avg_r_multiple": NaN}}],'
        ' "benchmarks": [{"name": "SPY_buy_hold", "sharpe": Infinity}]}')
    r = TestClient(api.app).get("/v1/backtest/summary", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["strategies"][0]["sharpe"] is None and body["benchmarks"][0]["sharpe"] is None


def test_save_metrics_writes_strict_json(art_dir):
    import math

    from driftrev import artifacts
    artifacts.save_metrics({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": math.nan}})
    text = (art_dir / "metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert artifacts.load_metrics() == {"a": None, "b": [1.0, None], "c": {"d": None}}
