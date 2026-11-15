"""Function-calling tool definitions (OpenAI format, accepted by xAI Grok) for the REST API.

Single source of truth: `_SPECS`. `TOOLS` is the OpenAI list; `ROUTES` maps each tool name
to (HTTP method, path template) so a dispatcher can call the API. `agent/tools.json` (+ routes.json) is
generated with `python -m driftrev.agent_tools`. Order submission is intentionally NOT
exposed as a tool: an LLM should not be able to place trades by default.
"""

from __future__ import annotations

import json
from pathlib import Path

_S, _I, _N, _B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}

# name, method, path, description, properties, required
_SPECS = [
    ("get_latest_signals", "GET", "/v1/signals/latest",
     "Get the newest actionable trade signals (ranked by probability) with market regime and a plain-English summary.",
     {"min_prob": {**_N, "description": "Minimum probability 0-1"},
      "tier": {**_S, "enum": ["STRONG_BUY", "BUY", "WATCH", "SKIP", "AVOID", "NO_CALL"]},
      "event_family": {**_S, "enum": ["earnings_up", "earnings_down", "earnings_crash", "shock_crash", "momentum_breakout"]},
      "limit": {**_I, "description": "Max rows (default 20)"}}, []),
    ("get_recent_events", "GET", "/v1/events/recent",
     "List recent events (earnings, crashes, momentum breakouts) with model probabilities and tiers.",
     {"days": {**_I, "description": "Lookback days (default 30)"}, "ticker": _S,
      "event_family": {**_S, "enum": ["earnings_up", "earnings_down", "earnings_crash", "shock_crash", "momentum_breakout"]}, "limit": _I}, []),
    ("get_ticker", "GET", "/v1/tickers/{ticker}",
     "Get details for one ticker: universe info, recent prices, events, signals and any open signal.",
     {"ticker": {**_S, "description": "Ticker symbol, e.g. AAPL"}}, ["ticker"]),
    ("get_backtest_summary", "GET", "/v1/backtest/summary",
     "Get out-of-sample backtest metrics for the strategies and benchmarks.", {}, []),
    ("get_backtest_trades", "GET", "/v1/backtest/trades",
     "List simulated backtest trades, newest first.", {"strategy": _S, "limit": _I}, []),
    ("get_backtest_equity", "GET", "/v1/backtest/equity",
     "Get equity curves of the strategies (weekly by default).", {"strategy": _S, "weekly": _B}, []),
    ("get_model_info", "GET", "/v1/model",
     "Get model metrics, top feature importances and precision breakdown.",
     {"top_n": _I, "dimension": {**_S, "description": "Filter breakdown by dimension"}}, []),
    ("get_research", "GET", "/v1/research",
     "Get research evidence: list tables and headline verdicts, or one table (model_comparison, walkforward, "
     "holding_period, uniquely_correct, guide_arms, survivorship, cost_stress, ...). Use it to justify or question a signal.",
     {"table": {**_S, "enum": ["model_comparison", "composition", "walkforward", "holding_period", "uniquely_correct",
                                "guide_arms", "guide_sensitivity", "cost_stress", "survivorship", "pred_eval",
                                "vector_sims", "calibration", "capacity_study", "cost_models", "filing_text_study"]},
      "limit": _I}, []),
    ("get_regime", "GET", "/v1/regime", "Get the market fragility regime state and triggers.", {}, []),
    ("run_pipeline", "POST", "/v1/pipeline/run",
     "Start the data update + scoring pipeline in the background; returns a job_id. Takes minutes.",
     {"mode": {**_S, "enum": ["update", "file"]}, "retrain": _B}, []),
    ("get_job_status", "GET", "/v1/jobs/{job_id}", "Check the status of a pipeline job.",
     {"job_id": _S}, ["job_id"]),
    ("get_runs", "GET", "/v1/runs", "Get the recent pipeline run log.", {"limit": _I}, []),
    ("preview_orders", "POST", "/v1/orders/preview",
     "Dry-run: show the bracket orders that would be placed for current BUY signals. Sends nothing.",
     {"max_positions": _I, "capital": _N}, []),
    ("get_portfolio", "GET", "/v1/portfolio", "Get the broker account equity and positions (paper by default).", {}, []),
]

TOOLS = [
    {"type": "function", "function": {
        "name": n, "description": d,
        "parameters": {"type": "object", "properties": props, "required": req}}}
    for n, _m, _p, d, props, req in _SPECS
]
ROUTES = {n: {"method": m, "path": p} for n, m, p, *_ in _SPECS}


def write_tools_json(directory: Path | None = None) -> Path:
    """Write tools.json (pure OpenAI list) and routes.json (name -> method/path)."""
    directory = directory or Path(__file__).resolve().parents[2] / "agent"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "routes.json").write_text(json.dumps(ROUTES, indent=2) + "\n")
    out = directory / "tools.json"
    out.write_text(json.dumps(TOOLS, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(write_tools_json())
