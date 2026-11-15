"""Business logic shared by the REST API and the MCP server.

Pure functions returning JSON-safe dicts. Everything reads ONLY pipeline artifacts via
`driftrev.artifacts` (never the DuckDB warehouse: the scheduler may be writing to it).
Missing artifacts never raise a stack trace; they produce an explanatory message.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from . import artifacts

log = logging.getLogger("driftrev.service")


class ServiceError(Exception):
    """A client-visible error; the API maps `status` to the HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


NO_DATA_MSG = (
    "No pipeline artifacts found yet. Run the pipeline first "
    "(POST /v1/pipeline/run or `python -m driftrev.pipeline --mode update`)."
)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise date columns so filters and JSON export behave."""
    out = df.copy()
    for c in ("date", "exit_by_date", "entry_date", "exit_date"):
        if c in out.columns and not pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def _filter_common(df: pd.DataFrame, ticker: str | None = None, event_family: str | None = None) -> pd.DataFrame:
    if ticker and "ticker" in df:
        df = df[df["ticker"].str.upper() == ticker.upper()]
    if event_family and "event_family" in df:
        # prefix match: "earnings" selects earnings_up / earnings_down / earnings_crash
        df = df[df["event_family"].str.lower().str.startswith(event_family.lower())]
    return df


def _check_limit(limit: int, hi: int = 1000) -> int:
    return max(1, min(int(limit), hi))


# ------------------------------------------------------------------ health
def health() -> dict[str, Any]:
    metrics = artifacts.load_metrics()
    present = {name: artifacts.path(name).exists() for name in artifacts.FILES}
    present["metrics"] = (artifacts.config.ARTIFACTS_DIR / artifacts.METRICS).exists()
    runs = artifacts.load_run_log(1)
    last = runs[-1] if runs else None
    return {
        "status": "ok" if present["metrics"] else "no_data",
        "data_through": metrics.get("data_through"),
        "generated_at": metrics.get("generated_at"),
        "artifacts_present": present,
        # /health is public: expose only the outcome and timing, not error text or step details
        # (the full run log stays behind the bearer token at /v1/runs)
        "last_run": {k: (last or {}).get(k) for k in ("started", "finished", "mode", "status", "seconds")} if last else None,
        "last_run_status": (last or {}).get("status"),
        "message": None if present["metrics"] else NO_DATA_MSG,
    }


# ------------------------------------------------------------------ signals
def _regime_state(metrics: dict, sig: pd.DataFrame) -> str | None:
    reg = metrics.get("regime") or {}
    state = reg.get("state") or reg.get("regime_state")
    if not state and not sig.empty and "regime_state" in sig:
        state = sig["regime_state"].dropna().iloc[0] if sig["regime_state"].notna().any() else None
    return state


def _signal_summary(rows: pd.DataFrame, regime: str | None, data_through: str | None) -> str:
    """Plain-English text an LLM agent can relay verbatim."""
    head = f"Data through {data_through or 'unknown'}; market regime: {regime or 'unknown'}."
    if rows.empty:
        return head + " No actionable signals match the filters."
    parts = []
    for r in rows.head(5).itertuples():
        w = getattr(r, "suggested_weight", None)
        bits = [f"{r.ticker} ({getattr(r, 'event_family', '?')}) tier {getattr(r, 'tier', '?')}, "
                f"P(+2% in 10d)={float(r.prob):.0%}"]
        if w is not None and pd.notna(w):
            bits.append(f"weight {float(w):.0%}")
        if pd.notna(getattr(r, "stop_loss_price", None)):
            bits.append(f"stop {float(r.stop_loss_price):.2f}")
        parts.append(", ".join(bits))
    txt = f"{head} {len(rows)} signal(s). Top: " + "; ".join(parts) + "."
    if regime and str(regime).upper() == "STRESSED":
        txt += " Market is STRESSED: the strategy pauses new entries."
    return txt + " Research output, not investment advice."


def latest_signals(min_prob: float | None = None, tier: str | None = None,
                   event_family: str | None = None, limit: int = 20) -> dict[str, Any]:
    metrics = artifacts.load_metrics()
    sig = artifacts.load_frame("signals")
    regime = _regime_state(metrics, sig)
    if sig.empty:
        return {"data_through": metrics.get("data_through"), "regime": regime, "count": 0,
                "signals": [], "summary": "No signals available. " + NO_DATA_MSG}
    sig = _clean(sig)
    if "date" in sig:  # newest events only
        sig = sig[sig["date"] == sig["date"].max()] if sig["date"].notna().any() else sig
    sig = _filter_common(sig, None, event_family)
    if min_prob is not None:
        sig = sig[sig["prob"] >= min_prob]
    if tier:
        sig = sig[sig["tier"].str.upper() == tier.upper()]
    sig = sig.sort_values("prob", ascending=False).head(_check_limit(limit))
    return {
        "data_through": metrics.get("data_through"),
        "regime": regime,
        "regime_detail": metrics.get("regime"),
        "count": len(sig),
        "signals": artifacts.to_records(sig),
        "summary": _signal_summary(sig, regime, metrics.get("data_through")),
    }


def recent_events(days: int = 30, ticker: str | None = None, event_family: str | None = None,
                  limit: int = 100) -> dict[str, Any]:
    ev = artifacts.load_frame("events")
    if ev.empty:
        return {"count": 0, "events": [], "message": NO_DATA_MSG}
    ev = _clean(ev)
    cutoff = ev["date"].max() - timedelta(days=int(days))
    ev = _filter_common(ev[ev["date"] >= cutoff], ticker, event_family)
    ev = ev.sort_values(["date", "prob"], ascending=[False, False]).head(_check_limit(limit))
    return {"count": len(ev), "events": artifacts.to_records(ev)}


def ticker_info(ticker: str) -> dict[str, Any]:
    t = ticker.upper().strip()
    uni = artifacts.load_frame("universe")
    prices = artifacts.load_frame("prices_recent")
    if uni.empty and prices.empty:
        raise ServiceError(404, NO_DATA_MSG)
    u = uni[uni["ticker"].str.upper() == t] if not uni.empty else uni
    p = prices[prices["ticker"].str.upper() == t] if not prices.empty else prices
    if u.empty and p.empty:
        raise ServiceError(404, f"Ticker {t} is not in the universe or recent prices.")
    p = _clean(p).sort_values("date").tail(60) if not p.empty else p
    ev = artifacts.load_frame("events")
    ev = _clean(_filter_common(ev, t)).sort_values("date", ascending=False).head(10) if not ev.empty else ev
    sig = artifacts.load_frame("signals")
    sig = _clean(_filter_common(sig, t)).sort_values("date", ascending=False) if not sig.empty else sig
    open_sig = None
    if not sig.empty and "exit_by_date" in sig:
        live = sig[(sig["exit_by_date"] >= pd.Timestamp.today().normalize()) &
                   (sig["tier"].isin(["STRONG_BUY", "BUY"]))]
        if not live.empty:
            open_sig = artifacts.to_records(live.head(1))[0]
    return {
        "ticker": t,
        "universe": artifacts.to_records(u.head(1))[0] if not u.empty else None,
        "prices": artifacts.to_records(p),
        "recent_events": artifacts.to_records(ev),
        "signals": artifacts.to_records(sig.head(10)),
        "open_signal": open_sig,
    }


# ------------------------------------------------------------------ backtest
def backtest_summary() -> dict[str, Any]:
    m = artifacts.load_metrics()
    if not m:
        return {"strategies": [], "benchmarks": [], "message": NO_DATA_MSG}
    return {"data_through": m.get("data_through"), "strategies": m.get("strategies", []),
            "benchmarks": m.get("benchmarks", [])}


def backtest_trades(strategy: str | None = None, limit: int = 100) -> dict[str, Any]:
    tr = artifacts.load_frame("trades")
    if tr.empty:
        return {"count": 0, "trades": [], "message": NO_DATA_MSG}
    tr = _clean(tr)
    if strategy and "strategy" in tr:
        tr = tr[tr["strategy"] == strategy]
    tr = tr.sort_values("entry_date", ascending=False).head(_check_limit(limit, 5000))
    return {"count": len(tr), "trades": artifacts.to_records(tr)}


def backtest_equity(strategy: str | None = None, weekly: bool = True) -> dict[str, Any]:
    eq = artifacts.load_frame("equity")
    if eq.empty:
        return {"count": 0, "equity": [], "message": NO_DATA_MSG}
    eq = _clean(eq)
    if strategy:
        eq = eq[eq["strategy"] == strategy]
    if weekly and not eq.empty:  # last observation of each week keeps payloads small
        eq = (eq.sort_values("date").set_index("date").groupby("strategy")["equity"]
              .resample("W").last().dropna().reset_index())
    return {"count": len(eq), "equity": artifacts.to_records(eq)}


# ------------------------------------------------------------------ model / regime
def model_info(top_n: int = 15, dimension: str | None = None) -> dict[str, Any]:
    m = artifacts.load_metrics()
    imp = artifacts.load_frame("importance")
    if not imp.empty:
        imp = imp.sort_values("importance", ascending=False).head(max(1, int(top_n)))
    br = artifacts.load_frame("breakdown")
    if not br.empty and dimension:
        br = br[br["dimension"] == dimension]
    return {
        "model": m.get("model", {}),
        "feature_importance": artifacts.to_records(imp),
        "precision_breakdown": artifacts.to_records(br),
        "dimensions": sorted(artifacts.load_frame("breakdown").get("dimension", pd.Series(dtype=str)).unique().tolist()),
        "message": None if m else NO_DATA_MSG,
    }


RESEARCH_TABLES = {  # name -> artifact file (written by pipeline.train_and_backtest / transform)
    "model_comparison": "model_comparison.parquet",
    "composition": "composition.parquet",
    "walkforward": "walkforward.parquet",
    "holding_period": "holding_period.parquet",
    "uniquely_correct": "uniquely_correct.parquet",
    "guide_arms": "guide_arms.parquet",
    "guide_sensitivity": "guide_sensitivity.parquet",
    "cost_stress": "cost_stress.parquet",
    "survivorship": "survivorship.parquet",
    "pred_eval": "pred_eval.parquet",
    "vector_sims": "vector_sims.parquet",
    "calibration": "calibration.parquet",
    "capacity_study": "capacity_study.parquet",
    "cost_models": "cost_models.parquet",
    "filing_text_study": "filing_text_study.parquet",
}


def research(table: str | None = None, limit: int = 200) -> dict[str, Any]:
    """Research tables for an agent: without `table`, list what exists plus the headline blocks
    (walk-forward, guide verdict); with `table`, return its rows."""
    m = artifacts.load_metrics()
    available = {k: artifacts.path(v).exists() for k, v in RESEARCH_TABLES.items()}
    if table is None:
        return {"available_tables": [k for k, ok in available.items() if ok],
                "walkforward": m.get("walkforward"), "guide": m.get("guide"),
                "versions": m.get("versions"), "message": None if m else NO_DATA_MSG}
    if table not in RESEARCH_TABLES:
        raise ServiceError(400, f"unknown table {table!r}; choose one of {sorted(RESEARCH_TABLES)}")
    df = artifacts.load_frame(RESEARCH_TABLES[table])
    if df.empty:
        return {"table": table, "count": 0, "rows": [], "message": f"{table} not produced yet; run the pipeline with --retrain"}
    for c in df.columns:  # interval / object columns -> str so JSON stays valid
        if df[c].dtype == "object" or str(df[c].dtype).startswith("interval"):
            df[c] = df[c].astype(str)
    return {"table": table, "count": len(df), "rows": artifacts.to_records(df.head(_check_limit(limit, 5000)))}


def regime() -> dict[str, Any]:
    m = artifacts.load_metrics()
    return {"data_through": m.get("data_through"), "regime": m.get("regime", {}),
            "message": None if m else NO_DATA_MSG}


# ------------------------------------------------------------------ pipeline jobs
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_job(job_id: str, mode: str, retrain: bool) -> None:
    try:
        from . import pipeline  # lazy: the core module is heavy and may not exist in tests

        result = pipeline.run_pipeline(mode=mode, retrain=retrain)
        upd = {"status": "succeeded", "result": result}
    except Exception as exc:  # job errors are reported, never raised into the server
        log.exception("pipeline job %s failed", job_id)
        upd = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    with _LOCK:
        _JOBS[job_id].update(upd, finished=_now())


def start_pipeline(mode: str = "update", retrain: bool = False) -> dict[str, Any]:
    if mode not in ("update", "file"):
        raise ServiceError(422, "mode must be 'update' or 'file'")
    with _LOCK:
        if any(j["status"] == "running" for j in _JOBS.values()):
            raise ServiceError(409, "A pipeline job is already running.")
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {"job_id": job_id, "status": "running", "mode": mode, "retrain": retrain,
                         "started": _now(), "finished": None, "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, mode, retrain), daemon=True).start()
    return {"job_id": job_id, "status": "running", "poll": f"/v1/jobs/{job_id}"}


def job_status(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise ServiceError(404, f"Unknown job {job_id}")
        return dict(job)


def runs(limit: int = 20) -> dict[str, Any]:
    return {"runs": artifacts.load_run_log(_check_limit(limit, 200))}


# ------------------------------------------------------------------ broker
def _orders_allowed() -> bool:
    return os.environ.get("DRIFTREV_ALLOW_ORDERS", "").lower() == "true"


def orders_preview(max_positions: int | None = None, capital: float | None = None) -> dict[str, Any]:
    from . import broker

    return broker.preview_orders(max_positions=max_positions or 5, capital=capital)


def orders_submit(confirm: bool) -> dict[str, Any]:
    from . import broker

    if confirm is not True:
        raise ServiceError(400, "Refused: set confirm=true to submit orders.")
    if not _orders_allowed():
        raise ServiceError(403, "Refused: order submission is disabled. Set DRIFTREV_ALLOW_ORDERS=true on the server.")
    return broker.submit_orders(confirm=True)


def portfolio() -> dict[str, Any]:
    from . import broker

    return broker.account_summary()
