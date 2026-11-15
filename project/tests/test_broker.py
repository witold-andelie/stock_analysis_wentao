import pandas as pd

from driftrev import broker


def _sig():
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"], "tier": ["STRONG_BUY", "BUY", "BUY", "SKIP"],
        "prob": [0.8, 0.7, 0.6, 0.9], "suggested_weight": [0.25, None, 0.5, 0.1],
        "entry_ref_price": [100.0, 30.0, 1000.0, 5.0],
        "stop_loss_price": [85.0, 25.5, 850.0, 4.0], "take_profit_price": [125.0, 37.5, 1250.0, 6.0],
        "exit_by_date": pd.Timestamp("2025-02-01"),
    })


def test_sizing():
    o = {x["ticker"]: x for x in broker.size_orders(_sig(), 10_000, max_positions=5)}
    assert o["A"]["qty"] == 25                      # 2500 / 100
    assert o["B"]["qty"] == 66                      # fallback 1/5 -> 2000 / 30 = 66
    assert "C" in o and o["C"]["qty"] == 5          # 5000 / 1000
    assert "D" not in o                             # SKIP tier ignored
    assert o["A"]["stop_loss_price"] == 85.0


def test_max_positions_and_tiny_equity():
    assert len(broker.size_orders(_sig(), 10_000, max_positions=2)) == 2
    assert broker.size_orders(_sig(), 50, max_positions=5) == []   # cannot afford a share


def test_preview_offline(art_dir):
    p = broker.preview_orders(capital=20_000)
    assert p["dry_run"] and p["equity_source"] == "capital" and not p["broker_configured"]
    assert broker.preview_orders()["equity_used"] == 10_000
    assert broker.submit_orders(confirm=True)["submitted"] is False


def test_expiry_uses_latest_signal_per_ticker():
    from datetime import date

    import pandas as pd

    from driftrev.broker import expired_tickers
    sig = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                        "exit_by_date": ["2026-11-01", "2026-11-30", "2026-11-05"]})
    # AAA has a newer signal (exit 11-30): its old 11-01 expiry must not close the fresh position
    assert expired_tickers(sig, date(2026, 11, 10)) == ["BBB"]
    assert expired_tickers(pd.DataFrame(), date(2026, 11, 10)) == []
