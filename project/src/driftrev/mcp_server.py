"""MCP server (FastMCP) exposing the same read tools as the REST API.

  python -m driftrev.mcp_server --transport stdio
  python -m driftrev.mcp_server --transport streamable-http --host 0.0.0.0 --port 8001

streamable-http requires the same bearer token as the API (DRIFTREV_API_TOKEN).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from . import security, service
from .service import ServiceError

log = logging.getLogger("driftrev.mcp")

mcp = FastMCP("drift-or-reversal")


def _safe(fn, *a, **kw) -> dict[str, Any]:
    """Tools return {'error': ...} instead of raising, so the LLM can read the problem."""
    try:
        return fn(*a, **kw)
    except ServiceError as e:
        return {"error": e.message, "status": e.status}
    except Exception as e:  # noqa: BLE001
        log.exception("tool failed")
        return {"error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def get_latest_signals(min_prob: float | None = None, tier: str | None = None,
                       event_family: str | None = None, limit: int = 20) -> dict:
    """Newest actionable signals ranked by probability, with regime and a plain-English summary."""
    return _safe(service.latest_signals, min_prob, tier, event_family, limit)


@mcp.tool()
def get_recent_events(days: int = 30, ticker: str | None = None,
                      event_family: str | None = None, limit: int = 100) -> dict:
    """Recent earnings/crash events with model probabilities."""
    return _safe(service.recent_events, days, ticker, event_family, limit)


@mcp.tool()
def get_ticker(ticker: str) -> dict:
    """Universe info, recent prices, events, signals and any open signal for one ticker."""
    return _safe(service.ticker_info, ticker)


@mcp.tool()
def get_backtest_summary() -> dict:
    """Out-of-sample backtest metrics for strategies and benchmarks."""
    return _safe(service.backtest_summary)


@mcp.tool()
def get_model_info(top_n: int = 15, dimension: str | None = None) -> dict:
    """Model metrics, top feature importances and precision breakdown."""
    return _safe(service.model_info, top_n, dimension)


@mcp.tool()
def get_regime() -> dict:
    """Market fragility regime state and triggers."""
    return _safe(service.regime)


@mcp.tool()
def run_pipeline(mode: str = "update", retrain: bool = False) -> dict:
    """Start the pipeline in the background (minutes); returns job_id. Poll get_job_status."""
    return _safe(service.start_pipeline, mode, retrain)


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Status/result/error of a pipeline job started with run_pipeline."""
    return _safe(service.job_status, job_id)


@mcp.tool()
def preview_orders(max_positions: int = 5, capital: float | None = None) -> dict:
    """Dry-run bracket orders for current BUY signals. Never submits anything."""
    return _safe(service.orders_preview, max_positions, capital)


class BearerAuthMiddleware:
    """Minimal ASGI middleware guarding the HTTP transport with the shared token."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            status, msg = security.check_bearer(headers.get("authorization"))
            if status != 200:
                await JSONResponse({"error": msg}, status_code=status)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_http_app():
    """Starlette app for streamable-http, wrapped with auth (also used by tests)."""
    return BearerAuthMiddleware(mcp.streamable_http_app())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.transport == "stdio":
        mcp.run("stdio")
        return
    import uvicorn

    uvicorn.run(build_http_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
