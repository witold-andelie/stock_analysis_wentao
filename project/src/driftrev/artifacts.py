"""Artifact contract: files the pipeline writes into ARTIFACTS_DIR and that the API, MCP
server, dashboard, broker and Telegram-style consumers read. Keep this module the single
source of truth for names and columns.

signals_latest.parquet   newest actionable events (one row per ticker/event)
    date, ticker, company, sector, cap_bucket, event_family, ret_1d, prob, tier,
    suggested_weight, entry_rule, entry_ref_price, stop_loss_price, take_profit_price,
    exit_by_date, edgar_items, edgar_category, sue, regime_state, model_name
events_recent.parquet    every event of the last ~90 days with its probability
    (same columns as signals_latest plus target columns when already known)
predictions_test.parquet out-of-sample predictions used for the backtest
    date, ticker, event_family, prob, y_true, growth_future
trades_backtest.parquet  strategy, ticker, event_family, prob, entry_date, entry_price,
    exit_date, exit_price, ret_net, pnl, exit_reason
equity_curves.parquet    date, strategy, equity
feature_importance.parquet  model, feature, importance
precision_breakdown.parquet dimension, value, n, precision, avg_growth
metrics.json             {generated_at, data_through, universe_size, n_events,
                          model{...}, models[...], strategies[...], benchmarks[...], regime{...}}
run_log.jsonl            one JSON object per pipeline run (started, finished, status, steps)
prices_recent.parquet    ticker, date, open, high, low, close, volume (last ~260 sessions)
universe.parquet         ticker, company, sector, sub_industry, cap_bucket, cik, wikipedia_title
model.joblib             fitted model bundle {model, features, threshold, name}

Consumers (API / MCP / dashboard) must read ONLY these files and never open the DuckDB
warehouse: DuckDB allows one writer process, and the scheduler may be writing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import config

FILES = {
    "signals": "signals_latest.parquet",
    "events": "events_recent.parquet",
    "predictions": "predictions_test.parquet",
    "trades": "trades_backtest.parquet",
    "equity": "equity_curves.parquet",
    "importance": "feature_importance.parquet",
    "breakdown": "precision_breakdown.parquet",
    "prices_recent": "prices_recent.parquet",
    "universe": "universe.parquet",
}
METRICS = "metrics.json"
RUN_LOG = "run_log.jsonl"
MODEL = "model.joblib"

TIERS = ("STRONG_BUY", "BUY", "WATCH", "SKIP", "AVOID", "NO_CALL")


def path(name: str) -> Path:
    return config.ARTIFACTS_DIR / FILES.get(name, name)


def save_frame(name: str, df: pd.DataFrame) -> Path:
    config.ensure_dirs()
    p = path(name)
    df.to_parquet(p, index=False)
    return p


def load_frame(name: str) -> pd.DataFrame:
    p = path(name)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _nan_to_none(obj):
    """Strict JSON has no NaN/Infinity: metrics with undefined values (e.g. Sharpe of an empty
    strategy) must become null, or API/MCP responses fail with HTTP 500."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj


def save_metrics(metrics: dict) -> Path:
    config.ensure_dirs()
    p = config.ARTIFACTS_DIR / METRICS
    clean = _nan_to_none(json.loads(json.dumps(metrics, default=str)))
    p.write_text(json.dumps(clean, indent=2, allow_nan=False))
    return p


def load_metrics() -> dict:
    p = config.ARTIFACTS_DIR / METRICS
    return _nan_to_none(json.loads(p.read_text())) if p.exists() else {}  # tolerate files written before the fix


def append_run_log(entry: dict) -> None:
    config.ensure_dirs()
    with open(config.ARTIFACTS_DIR / RUN_LOG, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def load_run_log(limit: int = 20) -> list[dict]:
    p = config.ARTIFACTS_DIR / RUN_LOG
    if not p.exists():
        return []
    lines = p.read_text().strip().splitlines()[-limit:]
    return [json.loads(x) for x in lines if x.strip()]


def to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (timestamps -> ISO dates, NaN -> None)."""
    if df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))
