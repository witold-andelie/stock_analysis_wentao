"""Audit F: backtest accounting (benchmarks, metrics, simulator invariants)."""

import numpy as np
import pandas as pd
import pytest

from driftrev import simulate


def _panel(n_names=60, n_days=80, star_in_index=False, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for i in range(n_names):
        r = rng.normal(0, 0.01, n_days)
        if i == 0:
            r += 0.02                              # 'Z': a future index addition that rallies hard
        score = np.full(n_days, 0.01 * (i + 1) / n_names)
        if i == 0:
            score[:] = 1.0                         # top momentum score on every date
        rows.append(pd.DataFrame({"ticker": f"T{i:02d}", "date": dates, "ret_1d": r, "is_liquid": True,
                                  "close_sma20": 1 + score, "close_sma50": 1 + score, "close_sma200": 1 + score,
                                  "mom_12_1": score, "in_index": (i != 0) or star_in_index}))
    return pd.concat(rows, ignore_index=True), dates


def test_momentum_baseline_excludes_names_not_yet_in_index():
    panel, dates = _panel(star_in_index=False)
    with_star_blocked = simulate.momentum_baseline(panel, dates[5], dates[-1], 10_000)
    panel_member, _ = _panel(star_in_index=True)
    with_star_member = simulate.momentum_baseline(panel_member, dates[5], dates[-1], 10_000)
    assert with_star_member.iloc[-1] > with_star_blocked.iloc[-1] * 1.05      # the rally name matters ...
    no_star = simulate.momentum_baseline(panel[panel["ticker"] != "T00"], dates[5], dates[-1], 10_000)
    pd.testing.assert_series_equal(with_star_blocked, no_star)                 # ... and is fully excluded when not a member


def test_sortino_uses_downside_deviation_not_std_of_losses():
    idx = pd.bdate_range("2024-01-01", periods=6)
    eq = pd.Series(100 * np.cumprod([1, 1.02, 0.99, 1.03, 0.98, 1.02]), index=idx)
    r = eq.pct_change().dropna()
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    assert simulate.perf_metrics(eq, "x")["sortino"] == pytest.approx(r.mean() * 252 / dd)


def test_long_cash_never_negative_and_costs_both_sides():
    dates = pd.bdate_range("2024-01-01", periods=14)
    px = np.full(len(dates), 10.0)
    panel = pd.DataFrame({"ticker": "X", "date": dates, "open": px, "high": px, "low": px, "close": px})
    ev = pd.DataFrame({"ticker": ["X"], "date": [dates[0]], "next_date": [dates[1]], "close": [10.0],
                       "event_family": ["shock_crash"], "regime_state": ["normal"], "cap_bucket": ["small"], "p": [0.9]})
    cfg = simulate.SimConfig(name="t", prob_col="p", min_prob=0.5, max_positions=1, hold_days=10, initial_capital=1000,
                             max_notional_pct=5.0)                     # asks for 5x equity: must be capped by cash
    eq, tr = simulate.exact_sim(ev, panel, cfg)
    t = tr.iloc[0]
    assert t["notional"] <= 1000 / (1 + 0.0025) + 1e-6                 # small cap one-way 25 bps
    assert t["pnl"] == pytest.approx(-2 * 0.0025 * t["notional"])      # flat price: only the two cost legs
    assert eq.iloc[-1] == pytest.approx(1000 - 2 * 0.0025 * t["notional"])
    assert t["entry_date"] == dates[1] and t["days_held"] == 10        # next open, exits at close of session 10


def test_report_charts_declared_headline_not_test_best(art_dir, monkeypatch, tmp_path):
    """The charted strategy is the pre-declared one, not the model_* strategy with the best TEST Sharpe."""
    import json

    from driftrev import config, report
    monkeypatch.setattr(config, "SITE_DIR", tmp_path / "site")
    (art_dir / "metrics.json").write_text(json.dumps({
        "data_through": "2026-11-13", "regime": {"state": "normal", "points": 0}, "best_strategy": "model_top5",
        "model": {"name": "rf", "threshold": 0.55, "test_auc": 0.56},
        "strategies": [{"name": "model_top5", "cagr": 0.05, "sharpe": 0.4, "max_drawdown": -0.1, "n_trades": 50},
                       {"name": "model_lucky", "cagr": 0.9, "sharpe": 3.0, "max_drawdown": -0.1, "n_trades": 5}],
        "benchmarks": []}))
    d = pd.bdate_range("2024-01-01", periods=40)
    pd.DataFrame({"date": list(d) * 2, "strategy": ["model_top5"] * 40 + ["model_lucky"] * 40,
                  "equity": list(np.linspace(100, 105, 40)) + list(np.linspace(100, 190, 40))}
                 ).to_parquet(art_dir / "equity_curves.parquet", index=False)
    report.build()
    html = (tmp_path / "site" / "index.html").read_text()
    assert "model_top5 (" in html and "model_lucky (" not in html


def _one_trade(opens, highs, lows, closes, **kw):
    dates = pd.bdate_range("2024-01-01", periods=len(opens))
    panel = pd.DataFrame({"ticker": "X", "date": dates, "open": opens, "high": highs, "low": lows, "close": closes})
    ev = pd.DataFrame({"ticker": ["X"], "date": [dates[0]], "next_date": [dates[1]], "close": [10.0],
                       "event_family": ["shock_crash"], "regime_state": ["normal"], "cap_bucket": ["mid"], "p": [0.9]})
    cfg = simulate.SimConfig(name="t", prob_col="p", min_prob=0.5, max_positions=1, initial_capital=1000, **kw)
    return simulate.exact_sim(ev, panel, cfg)


def test_gap_down_through_stop_fills_at_open_not_stop():
    _, tr = _one_trade([10, 10, 8.0, 8, 8], [10, 10.1, 8.1, 8, 8], [10, 9.9, 7.9, 8, 8], [10, 10, 8, 8, 8],
                       stop_loss=0.15, hold_days=10)
    assert tr.iloc[0]["exit_reason"] == "stop_loss" and tr.iloc[0]["exit_price"] == pytest.approx(8.0)


def test_stop_beats_take_profit_in_the_same_bar():
    _, tr = _one_trade([10, 10, 10, 10], [10, 10.1, 13.0, 10], [10, 9.9, 8.0, 10], [10, 10, 10, 10],
                       stop_loss=0.15, take_profit=0.25, hold_days=10)
    assert tr.iloc[0]["exit_reason"] == "stop_loss"


def test_entry_is_next_open_never_signal_close_and_gap_tolerance():
    # signal close 10.0, next open 11.0: filled at 11.0 (not 10); skipped if the gap exceeds the tolerance
    kw = dict(hold_days=2)
    _, tr = _one_trade([10, 11.0, 11, 11], [10, 11, 11, 11], [10, 11, 11, 11], [10, 11, 11, 11], **kw)
    assert tr.iloc[0]["entry_price"] == pytest.approx(11.0)
    _, tr2 = _one_trade([10, 11.0, 11, 11], [10, 11, 11, 11], [10, 11, 11, 11], [10, 11, 11, 11],
                        hold_days=2, gap_tolerance=0.03, atr_stop_mult=2.0, risk_per_trade=0.005)
    assert tr2.empty


def test_overlapping_signals_same_ticker_are_not_double_bought():
    dates = pd.bdate_range("2024-01-01", periods=15)
    px = np.full(15, 10.0)
    panel = pd.DataFrame({"ticker": "X", "date": dates, "open": px, "high": px, "low": px, "close": px})
    ev = pd.DataFrame({"ticker": ["X", "X"], "date": [dates[0], dates[2]], "next_date": [dates[1], dates[3]],
                       "close": [10.0, 10.0], "event_family": ["shock_crash"] * 2, "regime_state": ["normal"] * 2,
                       "cap_bucket": ["mid"] * 2, "p": [0.9, 0.9]})
    cfg = simulate.SimConfig(name="t", prob_col="p", min_prob=0.5, max_positions=5, hold_days=10, initial_capital=1000)
    _, tr = simulate.exact_sim(ev, panel, cfg)
    assert len(tr) == 1
