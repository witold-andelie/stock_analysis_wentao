"""Alpaca broker bridge (PAPER trading by default).

Safety model:
  * preview_orders() never touches the network for orders; with no keys it is a pure dry run.
  * submit_orders() needs confirm=True AND env DRIFTREV_ALLOW_ORDERS=true AND API keys.
  * ALPACA_PAPER defaults to true; live trading needs an explicit ALPACA_PAPER=false.
Orders are DAY market BRACKET orders (stop-loss + take-profit legs) so risk is capped
at the broker even if our process dies.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date
from typing import Any

import pandas as pd

from . import artifacts

log = logging.getLogger("driftrev.broker")

BUY_TIERS = ("STRONG_BUY", "BUY")
DEFAULT_CAPITAL = 10_000.0


def _keys() -> tuple[str, str] | None:
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    return (k, s) if k and s else None


def _paper() -> bool:
    return os.environ.get("ALPACA_PAPER", "true").lower() != "false"


def _allowed() -> bool:
    return os.environ.get("DRIFTREV_ALLOW_ORDERS", "").lower() == "true"


def configured() -> bool:
    return _keys() is not None


def _trading_client():
    from alpaca.trading.client import TradingClient  # lazy: keeps offline paths import-light

    k, s = _keys()  # type: ignore[misc]
    return TradingClient(k, s, paper=_paper())


def size_orders(signals: pd.DataFrame, equity: float, max_positions: int = 5) -> list[dict[str, Any]]:
    """Pure sizing math (unit-tested). notional = equity * suggested_weight, fallback equal
    weight 1/max_positions; whole shares from entry_ref_price; positions with qty 0 skipped."""
    if signals.empty:
        return []
    df = signals[signals["tier"].isin(BUY_TIERS)].sort_values("prob", ascending=False)
    df = df.drop_duplicates("ticker").head(max_positions)
    out = []
    for r in df.to_dict("records"):
        price = r.get("entry_ref_price")
        if price is None or pd.isna(price) or price <= 0:
            continue
        w = r.get("suggested_weight")
        if w is None or pd.isna(w) or w <= 0:
            w = 1.0 / max_positions
        notional = equity * float(w)
        qty = int(math.floor(notional / float(price)))
        if qty < 1:
            continue
        out.append({
            "ticker": r["ticker"], "side": "buy", "qty": qty, "tier": r["tier"],
            "prob": float(r["prob"]), "weight": float(w), "entry_ref_price": float(price),
            "notional": round(qty * float(price), 2),
            "stop_loss_price": _num(r.get("stop_loss_price")),
            "take_profit_price": _num(r.get("take_profit_price")),
            "exit_by_date": str(pd.Timestamp(r["exit_by_date"]).date()) if pd.notna(r.get("exit_by_date")) else None,
            "order_type": "market", "time_in_force": "day", "order_class": "bracket",
        })
    return out


def _num(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def account_summary() -> dict[str, Any]:
    if not configured():
        return {"configured": False,
                "message": "Broker not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper keys recommended)."}
    try:
        client = _trading_client()
        acct = client.get_account()
        pos = [{"ticker": p.symbol, "qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value), "unrealized_pl": float(p.unrealized_pl)}
               for p in client.get_all_positions()]
        return {"configured": True, "paper": _paper(), "equity": float(acct.equity),
                "cash": float(acct.cash), "buying_power": float(acct.buying_power), "positions": pos}
    except Exception as exc:
        log.warning("alpaca account error: %s", exc)
        return {"configured": True, "paper": _paper(), "error": f"{type(exc).__name__}: {exc}"}


def _held_tickers() -> set[str]:
    return {p["ticker"] for p in account_summary().get("positions", [])}


def preview_orders(max_positions: int = 5, capital: float | None = None) -> dict[str, Any]:
    """Dry-run order plan. Uses broker equity when keys exist, else `capital` or 10 000."""
    equity, source = capital or DEFAULT_CAPITAL, "capital" if capital else "default"
    if configured() and capital is None:
        acct = account_summary()
        if "equity" in acct:
            equity, source = acct["equity"], "broker"
    sig = artifacts.load_frame("signals")
    orders = size_orders(sig, equity, max_positions)
    return {
        "dry_run": True, "paper": _paper(), "equity_used": equity, "equity_source": source,
        "broker_configured": configured(), "orders_enabled": _allowed(),
        "count": len(orders), "orders": orders,
        "message": "Preview only; nothing was sent." if orders else "No BUY/STRONG_BUY signals to trade.",
    }


def submit_orders(confirm: bool = False, max_positions: int = 5) -> dict[str, Any]:
    if not confirm:
        return {"submitted": False, "message": "confirm=True required."}
    if not _allowed():
        return {"submitted": False, "message": "DRIFTREV_ALLOW_ORDERS is not 'true'."}
    if not configured():
        return {"submitted": False, "message": "Broker keys missing."}
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

    plan = preview_orders(max_positions)
    held = _held_tickers()
    client = _trading_client()
    pending = _open_order_tickers(client)   # audit: a second submit the same day must not duplicate
    results = []
    for o in plan["orders"]:
        if o["ticker"] in held or o["ticker"] in pending:
            results.append({"ticker": o["ticker"], "status": "skipped",
                            "reason": "already held" if o["ticker"] in held else "open order exists"})
            continue
        try:
            # GTC: with DAY, Alpaca can expire the bracket's stop/target legs at the close and leave
            # the position unprotected. client_order_id makes a same-day resubmit idempotent.
            req = MarketOrderRequest(
                symbol=o["ticker"], qty=o["qty"], side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                client_order_id=f"driftrev-{o['ticker']}-{date.today():%Y%m%d}",
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(o["stop_loss_price"], 2)),
                take_profit=TakeProfitRequest(limit_price=round(o["take_profit_price"], 2)),
            )
            res = client.submit_order(req)
            results.append({"ticker": o["ticker"], "status": "submitted", "order_id": str(res.id)})
        except Exception as exc:
            log.warning("order %s failed: %s", o["ticker"], exc)
            results.append({"ticker": o["ticker"], "status": "error", "reason": str(exc)})
    return {"submitted": True, "paper": _paper(), "results": results}


def _open_order_tickers(client) -> set[str]:
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        return {o.symbol for o in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))}
    except Exception as exc:  # unknown -> be conservative and log; held-position guard still applies
        log.warning("could not list open orders: %s", exc)
        return set()


def expired_tickers(sig: pd.DataFrame, today: date) -> list[str]:
    """Tickers whose MOST RECENT signal has passed its exit_by_date. Using the latest signal per
    ticker prevents an old signal's expiry from closing a position opened on a newer signal."""
    if sig.empty or "exit_by_date" not in sig:
        return []
    s = sig.assign(exit_by_date=pd.to_datetime(sig["exit_by_date"], errors="coerce")).dropna(subset=["exit_by_date"])
    latest = s.groupby("ticker")["exit_by_date"].max()
    return sorted(latest[latest.dt.date < today].index)


def close_expired_positions(today: date | None = None) -> dict[str, Any]:
    """Close positions whose signal's exit_by_date has passed (10-session hold). Dry run
    unless orders are allowed and keys are configured."""
    today = today or date.today()
    # events_recent also carries exit_by_date, so positions from older signals are covered
    sig = pd.concat([artifacts.load_frame("signals"), artifacts.load_frame("events")], ignore_index=True)
    if sig.empty or "exit_by_date" not in sig or not configured():
        return {"dry_run": True, "expired": [], "message": "No signals or broker not configured."}
    expired = expired_tickers(sig, today)
    held = [t for t in _held_tickers() if t in expired]
    dry = not _allowed()
    results = []
    if not dry:
        client = _trading_client()
        for t in held:
            try:
                client.close_position(t)
                results.append({"ticker": t, "status": "closed"})
            except Exception as exc:
                results.append({"ticker": t, "status": "error", "reason": str(exc)})
    return {"dry_run": dry, "expired": held, "results": results}
