"""R7.6 IBKR bridge: all broker interactions use an offline fake."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from driftrev import broker, broker_ibkr, config, service


@dataclass
class Stock:
    symbol: str
    exchange: str
    currency: str
    secType: str = "STK"


@dataclass(kw_only=True)
class Order:
    orderId: int = 0
    parentId: int = 0
    transmit: bool = True
    tif: str = ""
    orderRef: str = ""
    account: str = ""


@dataclass
class MarketOrder(Order):
    action: str
    totalQuantity: float
    orderType: str = field(default="MKT", init=False)


@dataclass
class LimitOrder(MarketOrder):
    lmtPrice: float
    orderType: str = field(default="LMT", init=False)


@dataclass
class StopOrder(MarketOrder):
    auxPrice: float
    orderType: str = field(default="STP", init=False)


def _balance(tag, value, currency="EUR", account="DU1234567"):
    return SimpleNamespace(account=account, tag=tag, value=str(value), currency=currency, modelCode="")


def _position(symbol="SAP", qty=20, currency="EUR", account="DU1234567", sec_type="STK"):
    return SimpleNamespace(account=account, contract=Stock(symbol, "IBIS", currency, sec_type),
                           position=qty, avgCost=95.5)


@pytest.fixture(autouse=True)
def fake_ib(monkeypatch):
    for name in ("IBKR_ENABLED", "IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID", "IBKR_ACCOUNT", "IBKR_PAPER",
                 "DRIFTREV_ALLOW_ORDERS"):
        monkeypatch.delenv(name, raising=False)
    state = SimpleNamespace(instances=[], calls=[], placed=[], cancelled=[], qualified=[], accounts=["DU1234567"],
                            balances=[_balance("NetLiquidation", 10_000), _balance("TotalCashValue", 4000)],
                            positions=[], trades=[], not_found=set(), qualification_errors=set(),
                            connect_error=None, place_error=None, cancel_error=None, next_id=100)

    class IB:
        def __init__(self):
            state.instances.append(self)
            self.client = self
            self.connected = False
            self.disconnected = False

        def connect(self, host, port, clientId, timeout, readonly, account):
            state.calls.append(("connect", host, port, clientId, timeout, readonly, account))
            if state.connect_error:
                raise state.connect_error
            self.connected = True
            return self

        def disconnect(self):
            state.calls.append(("disconnect",))
            self.connected = False
            self.disconnected = True

        def isConnected(self):
            return self.connected

        def managedAccounts(self):
            state.calls.append(("managedAccounts",))
            return state.accounts

        def accountSummary(self, account):
            state.calls.append(("accountSummary", account))
            return state.balances

        def positions(self, account):
            state.calls.append(("positions", account))
            return state.positions

        def openTrades(self):
            state.calls.append(("openTrades",))
            return list(state.trades)

        def qualifyContracts(self, *contracts):
            state.qualified.extend(contracts)
            if contracts[0].symbol in state.qualification_errors:
                raise ValueError("unknown contract")
            return [c for c in contracts if c.symbol not in state.not_found]

        def getReqId(self):
            state.next_id += 1
            return state.next_id

        def placeOrder(self, contract, order):
            state.calls.append(("placeOrder", contract, order))
            state.placed.append((contract, order))
            if state.place_error:
                raise state.place_error
            if not order.orderId:
                order.orderId = self.getReqId()
            trade = SimpleNamespace(contract=contract, order=order, orderStatus=SimpleNamespace(status="Submitted"))
            state.trades.append(trade)
            return trade

        def cancelOrder(self, order):
            state.calls.append(("cancelOrder", order))
            if state.cancel_error:
                raise state.cancel_error
            state.cancelled.append(order)

    api = SimpleNamespace(IB=IB, Stock=Stock, MarketOrder=MarketOrder, LimitOrder=LimitOrder, StopOrder=StopOrder)
    monkeypatch.setattr(broker_ibkr, "_ib_api", lambda: api)
    return state


def _enable(monkeypatch):
    monkeypatch.setenv("IBKR_ENABLED", "true")
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "true")


def _signals(art_dir, tickers=("SAP.DE",), **changes):
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append({"ticker": ticker, "tier": "STRONG_BUY", "prob": 0.9 - i * 0.01,
                     "suggested_weight": 0.2, "entry_ref_price": 100.0,
                     "stop_loss_price": 85.126, "take_profit_price": 125.129,
                     "exit_by_date": date.today() + timedelta(days=14), **changes})
    sig = pd.DataFrame(rows)
    sig.to_parquet(art_dir / "signals_latest.parquet", index=False)
    return sig


def test_not_configured_by_default(art_dir, fake_ib, monkeypatch):
    assert not broker_ibkr.configured()
    assert broker_ibkr.account_summary() == {
        "configured": False,
        "message": "IBKR not configured. Set IBKR_ENABLED=true and run TWS or IB Gateway (paper port 4002/7497).",
    }
    plan = broker_ibkr.preview_orders()
    assert plan["broker"] == "ibkr" and plan["exchange"] == "IBIS" and plan["dry_run"]
    assert plan["paper"] and plan["equity_used"] == broker.DEFAULT_CAPITAL
    assert plan["orders"] == broker.size_orders(pd.read_parquet(art_dir / "signals_latest.parquet"), 10_000)
    assert broker_ibkr.preview_orders(max_positions=1, capital=20_000)["count"] == 1
    assert not fake_ib.instances and not fake_ib.placed
    monkeypatch.setenv("IBKR_ENABLED", "true")
    assert broker_ibkr.configured()
    monkeypatch.setenv("IBKR_PORT", "invalid")
    assert not broker_ibkr.configured()
    assert not broker_ibkr.account_summary()["configured"]
    assert broker_ibkr.preview_orders()["equity_source"] == "default"
    assert not fake_ib.instances


def test_submit_refusals(art_dir, fake_ib, monkeypatch):
    assert broker_ibkr.submit_orders() == {"submitted": False, "message": "confirm=True required."}
    assert broker_ibkr.submit_orders(confirm=True) == {
        "submitted": False, "message": "DRIFTREV_ALLOW_ORDERS is not 'true'.",
    }
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "true")
    assert broker_ibkr.submit_orders(confirm=True) == {"submitted": False, "message": "IBKR not configured."}
    monkeypatch.setenv("IBKR_ENABLED", "true")
    monkeypatch.setenv("IBKR_PORT", "invalid")
    assert not broker_ibkr.submit_orders(confirm=True)["submitted"]
    assert not fake_ib.instances and not fake_ib.calls


def test_paper_guard(art_dir, fake_ib, monkeypatch):
    _signals(art_dir)
    _enable(monkeypatch)
    fake_ib.accounts = ["U1234567"]
    error = "RuntimeError: paper mode but account U1234567 is not a paper (DU...) account"
    refused = broker_ibkr.submit_orders(confirm=True)
    assert not refused["submitted"] and refused["error"] == error
    assert not fake_ib.placed and fake_ib.instances[-1].disconnected
    assert broker_ibkr.account_summary() == {"configured": True, "paper": True, "error": error}
    assert fake_ib.instances[-1].disconnected
    fake_ib.accounts = ["DU1234567"]
    assert broker_ibkr.submit_orders(confirm=True)["results"][0]["status"] == "submitted"
    assert len(fake_ib.placed) == 3 and fake_ib.instances[-1].disconnected
    fake_ib.trades.clear()
    monkeypatch.setenv("IBKR_ACCOUNT", "U7654321")
    assert not broker_ibkr.submit_orders(confirm=True)["submitted"]
    monkeypatch.setenv("IBKR_PAPER", "false")
    submitted = broker_ibkr.submit_orders(confirm=True)
    assert submitted["submitted"] and not submitted["paper"] and submitted["account"] == "U7654321"
    assert all(o.account == "U7654321" for _, o in fake_ib.placed[-3:])
    fake_ib.connect_error = TimeoutError("gateway unavailable")
    assert broker_ibkr.submit_orders(confirm=True)["error"] == "TimeoutError: gateway unavailable"
    assert fake_ib.instances[-1].disconnected


def test_bracket_structure(art_dir, fake_ib, monkeypatch):
    _signals(art_dir)
    _enable(monkeypatch)
    result = broker_ibkr.submit_orders(confirm=True)
    assert result == {"submitted": True, "broker": "ibkr", "paper": True, "account": "DU1234567",
                      "results": [{"ticker": "SAP.DE", "status": "submitted", "order_id": "101"}]}
    assert len(fake_ib.instances) == 1
    assert fake_ib.calls[0] == ("connect", "127.0.0.1", 4002, 17, 10, False, "")
    assert fake_ib.calls[-1] == ("disconnect",)
    assert fake_ib.qualified == [Stock("SAP", "IBIS", "EUR")]
    assert len(fake_ib.placed) == 3
    assert all(c == Stock("SAP", "IBIS", "EUR") for c, _ in fake_ib.placed)
    parent, target, stop = [o for _, o in fake_ib.placed]
    assert (parent.orderType, parent.action, parent.tif, parent.transmit, parent.parentId) == ("MKT", "BUY", "DAY", False, 0)
    assert (target.orderType, target.action, target.tif, target.transmit) == ("LMT", "SELL", "GTC", False)
    assert (stop.orderType, stop.action, stop.tif, stop.transmit) == ("STP", "SELL", "GTC", True)
    assert target.parentId == stop.parentId == parent.orderId
    assert target.lmtPrice == 125.13 and stop.auxPrice == 85.13
    assert [o.orderId for _, o in fake_ib.placed] == [101, 102, 103]
    assert all(o.totalQuantity == 20 and o.account == "DU1234567"
               and o.orderRef == f"driftrev-SAP-{date.today():%Y%m%d}" for _, o in fake_ib.placed)
    # Real submission errors remain per-ticker errors and still disconnect.
    fake_ib.trades.clear()
    fake_ib.place_error = RuntimeError("order rejected")
    assert broker_ibkr.submit_orders(confirm=True)["results"] == [
        {"ticker": "SAP.DE", "status": "error", "reason": "order rejected"},
    ]
    assert fake_ib.instances[-1].disconnected


def test_idempotency_and_held(art_dir, fake_ib, monkeypatch):
    _signals(art_dir)
    _enable(monkeypatch)
    for positions, refs, reason in [
        ([], ["driftrev-SAP-19990101"], "open order exists"),
        ([_position()], [], "already held"),
    ]:
        fake_ib.positions = positions
        fake_ib.trades = [SimpleNamespace(contract=Stock("SAP", "IBIS", "EUR"),
                                         order=MarketOrder("BUY", 20, orderRef=ref)) for ref in refs]
        assert broker_ibkr.submit_orders(confirm=True)["results"] == [
            {"ticker": "SAP.DE", "status": "skipped", "reason": reason},
        ]
    assert not fake_ib.placed and not fake_ib.qualified
    fake_ib.positions, fake_ib.trades = [], []
    _signals(art_dir, ("AAA", "UNKNOWN.DE", "ERROR.DE", "NOSTOP.DE", "NOTARGET.DE"))
    fake_ib.not_found.add("UNKNOWN")
    fake_ib.qualification_errors.add("ERROR")
    sig = pd.read_parquet(art_dir / "signals_latest.parquet")
    sig.loc[sig.ticker == "NOSTOP.DE", "stop_loss_price"] = None
    sig.loc[sig.ticker == "NOTARGET.DE", "take_profit_price"] = None
    sig.to_parquet(art_dir / "signals_latest.parquet", index=False)
    results = broker_ibkr.submit_orders(confirm=True)["results"]
    assert [r["reason"] for r in results] == ["not a Xetra symbol", "contract not found", "contract not found",
                                             "missing stop/target", "missing stop/target"]
    assert all(r["status"] == "skipped" for r in results) and not fake_ib.placed
    _signals(art_dir)
    fake_ib.trades = [SimpleNamespace(order=MarketOrder("BUY", 1, orderRef="driftrev-SAPX-20260101"))]
    assert broker_ibkr.submit_orders(confirm=True)["results"][0]["status"] == "submitted"
    assert broker_ibkr.submit_orders(confirm=True)["results"][0]["reason"] == "open order exists"
    assert len(fake_ib.placed) == 3


def test_account_summary_eur(art_dir, fake_ib, monkeypatch):
    _enable(monkeypatch)
    fake_ib.balances = [_balance("NetLiquidation", 22_000), _balance("NetLiquidation", 99_000, "USD"),
                        _balance("TotalCashValue", 6000), _balance("TotalCashValue", 88_000, "USD")]
    fake_ib.positions = [_position(), _position("AAPL", currency="USD"), _position("FUT", sec_type="FUT")]
    assert broker_ibkr.account_summary() == {
        "configured": True, "paper": True, "account": "DU1234567", "currency": "EUR", "equity": 22_000.0,
        "cash": 6000.0, "positions": [{"ticker": "SAP.DE", "qty": 20.0, "avg_cost": 95.5}],
    }
    assert fake_ib.instances[-1].disconnected
    plan = broker_ibkr.preview_orders()
    assert (plan["equity_source"], plan["equity_used"]) == ("broker", 22_000)
    before = len(fake_ib.instances)
    assert broker_ibkr.preview_orders(capital=1234)["equity_used"] == 1234
    assert len(fake_ib.instances) == before
    fake_ib.balances = [_balance("NetLiquidation", 99_000, "USD")]
    assert broker_ibkr.account_summary()["equity"] is None
    assert broker_ibkr.preview_orders()["equity_used"] == broker.DEFAULT_CAPITAL
    assert broker_ibkr.preview_orders()["equity_source"] == "default"
    # All connection settings and the chosen account are read again at call time.
    monkeypatch.setenv("IBKR_HOST", "fake-gateway")
    monkeypatch.setenv("IBKR_PORT", "7497")
    monkeypatch.setenv("IBKR_CLIENT_ID", "42")
    monkeypatch.setenv("IBKR_ACCOUNT", "DU7654321")
    assert broker_ibkr.account_summary()["account"] == "DU7654321"
    assert ("connect", "fake-gateway", 7497, 42, 10, False, "DU7654321") in fake_ib.calls
    monkeypatch.delenv("IBKR_ACCOUNT")
    fake_ib.accounts = []
    assert broker_ibkr.account_summary()["error"] == "RuntimeError: No IBKR account available."
    assert fake_ib.instances[-1].disconnected and not fake_ib.placed


def test_close_expired_positions(art_dir, fake_ib, monkeypatch):
    today = date(2026, 11, 22)
    sig = _signals(art_dir, ("SAP.DE", "FRESH.DE", "UNHELD.DE"), exit_by_date=today - timedelta(days=1))
    # A newer event protects FRESH; an older event covers a position missing from signals.
    events = sig.iloc[[1]].assign(exit_by_date=today + timedelta(days=1))
    events = pd.concat([events, sig.iloc[[0]].assign(ticker="OLD.DE")], ignore_index=True)
    events.to_parquet(art_dir / "events_recent.parquet", index=False)
    fake_ib.positions = [_position(), _position("FRESH"), _position("OLD", qty=7), _position("AAPL", currency="USD")]
    assert broker_ibkr.close_expired_positions(today)["dry_run"]
    assert not fake_ib.instances
    _enable(monkeypatch)
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "false")
    dry = broker_ibkr.close_expired_positions(today)
    assert dry["dry_run"] and set(dry["expired"]) == {"SAP.DE", "OLD.DE"} and dry["results"] == []
    assert not fake_ib.placed and fake_ib.instances[-1].disconnected
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "true")
    closed = broker_ibkr.close_expired_positions(today)
    assert not closed["dry_run"] and set(closed["expired"]) == {"SAP.DE", "OLD.DE"}
    assert closed["results"] == [{"ticker": "SAP.DE", "status": "closed", "cancelled_exits": 0},
                                 {"ticker": "OLD.DE", "status": "closed", "cancelled_exits": 0}]
    assert len(fake_ib.placed) == 2
    for (contract, order), qty in zip(fake_ib.placed, (20, 7)):
        assert (order.orderType, order.action, order.totalQuantity, order.tif) == ("MKT", "SELL", qty, "DAY")
        assert order.orderRef == f"driftrev-close-{contract.symbol}-{today:%Y%m%d}"
        assert order.account == "DU1234567" and order.transmit and not order.parentId
    assert fake_ib.instances[-1].disconnected
    fake_ib.accounts = ["U1234567"]
    assert "paper mode" in broker_ibkr.close_expired_positions(today)["error"]
    assert len(fake_ib.placed) == 2 and fake_ib.instances[-1].disconnected


def test_close_cancels_bracket_exits_first(art_dir, fake_ib, monkeypatch):
    today = date(2026, 11, 22)
    _signals(art_dir, exit_by_date=today - timedelta(days=1))
    _enable(monkeypatch)
    position = _position(qty=10)
    fake_ib.positions = [position]
    stop = StopOrder("SELL", 10, 85, tif="GTC", orderRef="driftrev-SAP-20261101")
    target = LimitOrder("SELL", 10, 125, tif="GTC", orderRef="driftrev-SAP-20261101")
    bmw = MarketOrder("BUY", 5, orderRef="driftrev-BMW-20261101")
    fake_ib.trades = [SimpleNamespace(contract=position.contract, order=stop),
                      SimpleNamespace(contract=position.contract, order=target),
                      SimpleNamespace(contract=Stock("BMW", "IBIS", "EUR"), order=bmw)]
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "false")
    dry = broker_ibkr.close_expired_positions(today)
    assert dry == {"dry_run": True, "expired": ["SAP.DE"], "results": []}
    assert not fake_ib.cancelled and not fake_ib.placed
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "true")
    closed = broker_ibkr.close_expired_positions(today)
    assert closed["results"] == [{"ticker": "SAP.DE", "status": "closed", "cancelled_exits": 2}]
    assert fake_ib.cancelled == [stop, target]
    assert len(fake_ib.placed) == 1
    contract, order = fake_ib.placed[0]
    assert contract == position.contract
    assert (order.orderType, order.action, order.totalQuantity) == ("MKT", "SELL", 10)
    assert order.orderRef == f"driftrev-close-SAP-{today:%Y%m%d}"
    assert [call for call in fake_ib.calls if call[0] in ("cancelOrder", "placeOrder")] == [
        ("cancelOrder", stop), ("cancelOrder", target), ("placeOrder", contract, order),
    ]
    assert fake_ib.instances[-1].disconnected


def test_close_skips_short_and_duplicates(art_dir, fake_ib, monkeypatch):
    today = date(2026, 11, 22)
    _signals(art_dir, exit_by_date=today - timedelta(days=1))
    _enable(monkeypatch)
    stop = SimpleNamespace(contract=Stock("SAP", "IBIS", "EUR"),
                           order=StopOrder("SELL", 10, 85, orderRef="driftrev-SAP-20261101"))
    close = SimpleNamespace(contract=stop.contract,
                            order=MarketOrder("SELL", 10, orderRef=f"driftrev-close-SAP-{today:%Y%m%d}"))
    fake_ib.trades = [stop, close]
    for qty in (-5, 0):
        fake_ib.positions = [_position(qty=qty)]
        short = broker_ibkr.close_expired_positions(today)
        assert short["results"] == [{"ticker": "SAP.DE", "status": "skipped", "reason": "not a long position"}]
    fake_ib.positions = [_position(qty=10)]
    duplicate = broker_ibkr.close_expired_positions(today)
    assert duplicate["results"] == [{"ticker": "SAP.DE", "status": "skipped", "reason": "close order already open"}]
    assert not fake_ib.placed and not fake_ib.cancelled
    assert all(instance.disconnected for instance in fake_ib.instances)


def test_close_cancel_failure_blocks_sell(art_dir, fake_ib, monkeypatch):
    today = date(2026, 11, 22)
    _signals(art_dir, exit_by_date=today - timedelta(days=1))
    _enable(monkeypatch)
    position = _position(qty=10)
    fake_ib.positions = [position]
    stop = StopOrder("SELL", 10, 85, orderRef="driftrev-SAP-20261101")
    fake_ib.trades = [SimpleNamespace(contract=position.contract, order=stop)]
    fake_ib.cancel_error = RuntimeError("cancellation failed")
    failed = broker_ibkr.close_expired_positions(today)
    assert not failed["dry_run"]
    assert failed["results"] == [{"ticker": "SAP.DE", "status": "error", "reason": "cancellation failed"}]
    assert ("cancelOrder", stop) in fake_ib.calls
    assert not fake_ib.placed
    assert fake_ib.instances[-1].disconnected


def test_routing_by_market(art_dir, fake_ib, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])
    assert service.broker_for_market() is broker
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    assert service.broker_for_market() is broker_ibkr
    monkeypatch.setattr(broker_ibkr, "account_summary", lambda: {"ibkr": "portfolio"})
    assert service.portfolio() == {"ibkr": "portfolio"}
    monkeypatch.setattr(broker_ibkr, "preview_orders", lambda **kw: kw)
    assert service.orders_preview(max_positions=2, capital=500) == {"max_positions": 2, "capital": 500}
    monkeypatch.setattr(broker_ibkr, "submit_orders", lambda **kw: kw)
    with pytest.raises(service.ServiceError) as exc:
        service.orders_submit(confirm=False)
    assert exc.value.status == 400
    with pytest.raises(service.ServiceError) as exc:
        service.orders_submit(confirm=True)
    assert exc.value.status == 403
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "true")
    assert service.orders_submit(confirm=True) == {"confirm": True}
    assert not fake_ib.instances


def test_requirements_pin():
    root = Path(__file__).resolve().parents[1]
    for name in ("requirements.txt", "requirements-runtime.txt"):
        assert "ib_async==2.1.0" in (root / name).read_text().splitlines()
