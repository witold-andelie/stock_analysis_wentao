"""FastAPI app exposing the pipeline artifacts to agents and dashboards.

Run: uvicorn driftrev.api:app --port 8000
Every route has a stable operation_id so OpenAPI importers (agents) get clean tool names.
Logic lives in `driftrev.service`; this module only does HTTP concerns.
"""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from . import security, service
from .service import ServiceError

app = FastAPI(
    title="Drift-or-Reversal Signals API",
    version="1.0.0",
    description=(
        "Event-driven ML signals for US stocks (S&P 1500): post-earnings drift and crash-day "
        "reversal. Each signal is the model probability that the stock gains >=2% from next "
        "open to the close 10 sessions later. Research output, not investment advice. "
        "Auth: `Authorization: Bearer <DRIFTREV_API_TOKEN>`."
    ),
)


def require_auth(authorization: str | None = Header(default=None)) -> None:
    status, msg = security.check_bearer(authorization)
    if status != 200:
        raise HTTPException(status, msg, headers={"WWW-Authenticate": "Bearer"} if status == 401 else None)


def _run(fn, *args, **kwargs):
    """Translate ServiceError into HTTP errors; anything else becomes a clean 500 message."""
    try:
        return fn(*args, **kwargs)
    except ServiceError as e:
        raise HTTPException(e.status, e.message)


AUTH = [Depends(require_auth)]


class PipelineRequest(BaseModel):
    mode: Literal["update", "file"] = Field("update", description="'update' downloads latest data; 'file' uses the local snapshot.")
    retrain: bool = Field(False, description="Refit the model before scoring.")


class PreviewRequest(BaseModel):
    max_positions: int | None = Field(None, ge=1, le=50, description="Max simultaneous positions (default 5).")
    capital: float | None = Field(None, gt=0, description="Capital for sizing; defaults to broker equity or 10000.")


class SubmitRequest(BaseModel):
    confirm: bool = Field(False, description="Must be true. Server also needs DRIFTREV_ALLOW_ORDERS=true.")


@app.get("/health", operation_id="health", summary="Service health (public)",
         description="Public liveness check: data_through, which artifacts exist, last pipeline run status.")
def health():
    return service.health()


@app.get("/v1/signals/latest", dependencies=AUTH, operation_id="get_latest_signals",
         summary="Latest actionable signals",
         description="Newest signals ranked by probability, with market regime and a plain-English summary to relay.")
def latest_signals(min_prob: float | None = Query(None, ge=0, le=1, description="Minimum probability."),
                   tier: str | None = Query(None, description="STRONG_BUY, BUY, WATCH, SKIP, AVOID or NO_CALL."),
                   event_family: str | None = Query(None, description="'earnings' or 'shock'."),
                   limit: int = Query(20, ge=1, le=200)):
    return _run(service.latest_signals, min_prob, tier, event_family, limit)


@app.get("/v1/events/recent", dependencies=AUTH, operation_id="get_recent_events",
         summary="Recent events with probabilities",
         description="Earnings and crash-day events of the last N days with model probabilities.")
def recent_events(days: int = Query(30, ge=1, le=365), ticker: str | None = None,
                  event_family: str | None = Query(None, description="'earnings' or 'shock'."),
                  limit: int = Query(100, ge=1, le=1000)):
    return _run(service.recent_events, days, ticker, event_family, limit)


@app.get("/v1/tickers/{ticker}", dependencies=AUTH, operation_id="get_ticker",
         summary="Ticker detail",
         description="Universe info, last ~60 daily bars, recent events/signals and any open signal for a ticker.")
def ticker(ticker: str):
    return _run(service.ticker_info, ticker)


@app.get("/v1/backtest/summary", dependencies=AUTH, operation_id="get_backtest_summary",
         summary="Backtest metrics", description="Out-of-sample strategy metrics (CAGR, Sharpe, max drawdown) vs benchmarks.")
def backtest_summary():
    return _run(service.backtest_summary)


@app.get("/v1/backtest/trades", dependencies=AUTH, operation_id="get_backtest_trades",
         summary="Backtest trades", description="Simulated trades, newest first.")
def backtest_trades(strategy: str | None = None, limit: int = Query(100, ge=1, le=5000)):
    return _run(service.backtest_trades, strategy, limit)


@app.get("/v1/backtest/equity", dependencies=AUTH, operation_id="get_backtest_equity",
         summary="Backtest equity curves", description="Equity per strategy; weekly downsampled by default.")
def backtest_equity(strategy: str | None = None, weekly: bool = True):
    return _run(service.backtest_equity, strategy, weekly)


@app.get("/v1/model", dependencies=AUTH, operation_id="get_model_info",
         summary="Model info", description="Model metrics, top feature importances and precision breakdown by dimension.")
def model(top_n: int = Query(15, ge=1, le=100), dimension: str | None = Query(None, description="Filter breakdown, e.g. 'event_family'.")):
    return _run(service.model_info, top_n, dimension)


@app.get("/v1/regime", dependencies=AUTH, operation_id="get_regime",
         summary="Market regime", description="Fragility monitor state (NORMAL/ELEVATED/STRESSED) and its triggers.")
def regime():
    return _run(service.regime)


@app.post("/v1/pipeline/run", dependencies=AUTH, operation_id="run_pipeline", status_code=202,
          summary="Start pipeline run",
          description="Starts the data+model pipeline in a background thread (minutes). Returns job_id; 409 if one is running.")
def pipeline_run(req: PipelineRequest):
    return _run(service.start_pipeline, req.mode, req.retrain)


@app.get("/v1/jobs/{job_id}", dependencies=AUTH, operation_id="get_job_status",
         summary="Pipeline job status", description="Status (running/succeeded/failed), result or error of a job.")
def job(job_id: str):
    return _run(service.job_status, job_id)


@app.get("/v1/runs", dependencies=AUTH, operation_id="get_runs",
         summary="Pipeline run log", description="Recent pipeline runs from the run log.")
def runs(limit: int = Query(20, ge=1, le=200)):
    return _run(service.runs, limit)


@app.post("/v1/orders/preview", dependencies=AUTH, operation_id="preview_orders",
          summary="Preview orders (dry run)",
          description="Sizes bracket orders for current BUY/STRONG_BUY signals. Never sends anything.")
def orders_preview(req: PreviewRequest | None = None):
    req = req or PreviewRequest()
    return _run(service.orders_preview, req.max_positions, req.capital)


@app.post("/v1/orders/submit", dependencies=AUTH, operation_id="submit_orders",
          summary="Submit orders (guarded)",
          description="Places the previewed DAY bracket orders at Alpaca. Requires confirm=true AND server env DRIFTREV_ALLOW_ORDERS=true.")
def orders_submit(req: SubmitRequest):
    return _run(service.orders_submit, req.confirm)


@app.get("/v1/portfolio", dependencies=AUTH, operation_id="get_portfolio",
         summary="Broker account and positions", description="Alpaca account summary, or a 'broker not configured' message.")
def portfolio():
    return _run(service.portfolio)


@app.get("/v1/agent/tools", dependencies=AUTH, operation_id="get_agent_tools",
         summary="Function-calling tool list", description="OpenAI/xAI function-calling tool definitions for this API.")
def agent_tools():
    from . import agent_tools as at

    return at.TOOLS
