"""Interactive Brokers bridge for Xetra (PAPER trading by default).

IBKR_ENABLED must be true before connecting. Submission also requires confirm=True
and DRIFTREV_ALLOW_ORDERS=true; paper mode requires a DU account. ib_async is lazy
so offline previews work without TWS / IB Gateway or the optional library installed.
Expiry closing cancels bracket exits before selling; a cancellation error blocks
the sell.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any

import pandas as pd

from . import artifacts
from .broker import BUY_TIERS as BUY_TIERS
from .broker import DEFAULT_CAPITAL, expired_tickers, size_orders

log = logging.getLogger("driftrev.broker_ibkr")


def _paper() -> bool:
    return os.environ.get("IBKR_PAPER", "true").lower() != "false"


def _allowed() -> bool:
    return os.environ.get("DRIFTREV_ALLOW_ORDERS", "").lower() == "true"


def configured() -> bool:
    try:
        int(os.environ.get("IBKR_PORT", "4002"))
    except ValueError:
        return False
    return os.environ.get("IBKR_ENABLED", "false").lower() == "true"


def _ib_api():
    from ib_async import IB, LimitOrder, MarketOrder, Stock, StopOrder

    return SimpleNamespace(IB=IB, Stock=Stock, MarketOrder=MarketOrder, LimitOrder=LimitOrder, StopOrder=StopOrder)


@contextmanager
def _connect():
    ib = _ib_api().IB()
    try:
        account = os.environ.get("IBKR_ACCOUNT", "")
        ib.connect(os.environ.get("IBKR_HOST", "127.0.0.1"), int(os.environ.get("IBKR_PORT", "4002")),
                   clientId=int(os.environ.get("IBKR_CLIENT_ID", "17")), timeout=10, readonly=False, account=account)
        if not account:
            accounts = ib.managedAccounts()
            if not accounts:
                raise RuntimeError("No IBKR account available.")
            account = accounts[0]
        if _paper() and not account.startswith("DU"):
            raise RuntimeError(f"paper mode but account {account} is not a paper (DU...) account")
        yield ib, account
    finally:
        ib.disconnect()


def _eur_balances(ib, account: str) -> dict[str, float | None]:
    tags = {v.tag: float(v.value) for v in ib.accountSummary(account)
            if v.account == account and v.currency == "EUR" and v.tag in ("NetLiquidation", "TotalCashValue")}
    return {"equity": tags.get("NetLiquidation"), "cash": tags.get("TotalCashValue")}


def _eur_positions(ib, account: str) -> list:
    return [p for p in ib.positions(account) if p.account == account
            and p.contract.secType == "STK" and p.contract.currency == "EUR"]


def account_summary() -> dict[str, Any]:
    if not configured():
        return {"configured": False,
                "message": "IBKR not configured. Set IBKR_ENABLED=true and run TWS or IB Gateway (paper port 4002/7497)."}
    try:
        with _connect() as (ib, account):
            positions = [{"ticker": f"{p.contract.symbol}.DE", "qty": float(p.position), "avg_cost": float(p.avgCost)}
                         for p in _eur_positions(ib, account)]
            return {"configured": True, "paper": _paper(), "account": account, "currency": "EUR",
                    **_eur_balances(ib, account), "positions": positions}
    except Exception as exc:
        log.warning("ibkr account error: %s", exc)
        return {"configured": True, "paper": _paper(), "error": f"{type(exc).__name__}: {exc}"}


def preview_orders(max_positions: int = 5, capital: float | None = None) -> dict[str, Any]:
    """Dry-run order plan, using EUR broker equity when available; never places orders."""
    equity, source = capital or DEFAULT_CAPITAL, "capital" if capital else "default"
    if configured() and capital is None:
        acct = account_summary()
        if acct.get("equity") is not None:
            equity, source = acct["equity"], "broker"
    orders = size_orders(artifacts.load_frame("signals"), equity, max_positions)
    return {
        "dry_run": True, "broker": "ibkr", "exchange": "IBIS", "paper": _paper(),
        "equity_used": equity, "equity_source": source,
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
        return {"submitted": False, "message": "IBKR not configured."}
    try:
        with _connect() as (ib, account):
            api = _ib_api()
            equity = _eur_balances(ib, account)["equity"]
            orders = size_orders(artifacts.load_frame("signals"),
                                 DEFAULT_CAPITAL if equity is None else equity, max_positions)
            held = {p.contract.symbol for p in _eur_positions(ib, account) if p.position != 0}
            pending = {t.order.orderRef for t in ib.openTrades()}
            results = []
            for o in orders:
                ticker = o["ticker"]
                symbol = ticker.removesuffix(".DE")
                reason = None
                if not ticker.endswith(".DE"):
                    reason = "not a Xetra symbol"
                elif symbol in held:
                    reason = "already held"
                elif any(ref.startswith(f"driftrev-{symbol}-") for ref in pending):
                    reason = "open order exists"
                elif o["stop_loss_price"] is None or o["take_profit_price"] is None:
                    reason = "missing stop/target"
                if reason:
                    results.append({"ticker": ticker, "status": "skipped", "reason": reason})
                    continue
                try:
                    contracts = ib.qualifyContracts(api.Stock(symbol, "IBIS", "EUR"))
                except Exception as exc:
                    log.warning("ibkr contract %s failed: %s", ticker, exc)
                    contracts = []
                if not contracts:
                    results.append({"ticker": ticker, "status": "skipped", "reason": "contract not found"})
                    continue
                try:
                    ref = f"driftrev-{symbol}-{date.today():%Y%m%d}"
                    parent = api.MarketOrder("BUY", o["qty"], orderId=ib.client.getReqId(), transmit=False,
                                             tif="DAY", orderRef=ref, account=account)
                    target = api.LimitOrder("SELL", o["qty"], round(o["take_profit_price"], 2),
                                            orderId=ib.client.getReqId(), parentId=parent.orderId, transmit=False,
                                            tif="GTC", orderRef=ref, account=account)
                    stop = api.StopOrder("SELL", o["qty"], round(o["stop_loss_price"], 2),
                                         orderId=ib.client.getReqId(), parentId=parent.orderId, transmit=True,
                                         tif="GTC", orderRef=ref, account=account)
                    # A DAY market parent submitted after the Xetra close executes at the next session's
                    # open; GTC children keep the position protected.
                    for order in (parent, target, stop):
                        ib.placeOrder(contracts[0], order)
                    pending.add(ref)
                    results.append({"ticker": ticker, "status": "submitted", "order_id": str(parent.orderId)})
                except Exception as exc:
                    log.warning("ibkr order %s failed: %s", ticker, exc)
                    results.append({"ticker": ticker, "status": "error", "reason": str(exc)})
            return {"submitted": True, "broker": "ibkr", "paper": _paper(), "account": account, "results": results}
    except Exception as exc:
        log.warning("ibkr submission error: %s", exc)
        return {"submitted": False, "broker": "ibkr", "paper": _paper(), "error": f"{type(exc).__name__}: {exc}"}


def close_expired_positions(today: date | None = None) -> dict[str, Any]:
    """Close held, expired signals; dry run unless orders are allowed and IBKR is configured."""
    today = today or date.today()
    sig = pd.concat([artifacts.load_frame("signals"), artifacts.load_frame("events")], ignore_index=True)
    if sig.empty or "exit_by_date" not in sig or not configured():
        return {"dry_run": True, "expired": [], "message": "No signals or broker not configured."}
    expired = expired_tickers(sig, today)
    try:
        with _connect() as (ib, account):
            held = [p for p in _eur_positions(ib, account) if f"{p.contract.symbol}.DE" in expired]
            dry = not _allowed()
            results = []
            if not dry:
                api = _ib_api()
                for p in held:
                    symbol = p.contract.symbol
                    if p.position <= 0:
                        results.append({"ticker": f"{symbol}.DE", "status": "skipped", "reason": "not a long position"})
                        continue
                    try:
                        ref = f"driftrev-close-{symbol}-{today:%Y%m%d}"
                        trades = ib.openTrades()
                        if any(t.order.orderRef == ref for t in trades):
                            results.append({"ticker": f"{symbol}.DE", "status": "skipped",
                                            "reason": "close order already open"})
                            continue
                        cancelled_exits = 0
                        # Cancel exits before selling so leftover GTC children cannot sell
                        # shares no longer held and open an unwanted short position.
                        for trade in trades:
                            if trade.contract.symbol == symbol and trade.order.orderRef.startswith(f"driftrev-{symbol}-"):
                                ib.cancelOrder(trade.order)
                                cancelled_exits += 1
                        ib.placeOrder(p.contract, api.MarketOrder("SELL", float(p.position), tif="DAY",
                                      orderRef=ref, account=account))
                        results.append({"ticker": f"{symbol}.DE", "status": "closed", "cancelled_exits": cancelled_exits})
                    except Exception as exc:
                        results.append({"ticker": f"{symbol}.DE", "status": "error", "reason": str(exc)})
            return {"dry_run": dry, "expired": [f"{p.contract.symbol}.DE" for p in held], "results": results}
    except Exception as exc:
        log.warning("ibkr closing error: %s", exc)
        return {"dry_run": True, "expired": [], "error": f"{type(exc).__name__}: {exc}"}
