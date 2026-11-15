"""Core research invariants: event timing, no look-ahead, taxonomy, simulation math."""

import numpy as np
import pandas as pd
import pytest

from driftrev import features, indicators, simulate
from driftrev.signals import assign_tier


def _px(n=300, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"ticker": ticker, "date": dates, "open": close * 0.995, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
                         "dividends": 0.0, "splits": 0.0})


def _ctx(px):
    d = px["date"]
    spy = pd.Series(np.linspace(300, 330, len(d)), index=d)
    ctx = pd.DataFrame({"spy_close": spy, "spy_ret_1d": spy.pct_change()}, index=d)
    for h in features.HORIZONS:
        ctx[f"spy_future_{h}d"] = spy.shift(-h) / spy.shift(-1)
    return ctx


def test_earnings_after_close_maps_to_next_session():
    dates = pd.bdate_range("2024-01-01", periods=10).values
    filings = pd.DataFrame({"cik": 1, "form": "8-K", "items": ["2.02,9.01", "2.02", "2.02"],
                            "accepted_et": pd.to_datetime(["2024-01-02 16:05", "2024-01-04 07:00", "2024-01-06 10:00"])})
    out = features.earnings_days(filings, 1, dates)
    assert list(pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04", "2024-01-08"]


def test_features_have_no_look_ahead():
    """Changing the last 15 rows must not change any feature before them (targets excluded)."""
    px = _px()
    a = features.ticker_features(px, _ctx(px), None)
    px2 = px.copy()
    px2.loc[px2.index[-15:], ["open", "high", "low", "close"]] *= 1.5
    b = features.ticker_features(px2, _ctx(px2), None)
    cutoff = len(px) - 15
    target_like = [c for c in a.columns if "future" in c or c.startswith(("mae_", "mfe_")) or c in ("next_date",)]
    cols = [c for c in a.columns if c not in target_like + ["ticker", "date", "prev_date"]]
    pd.testing.assert_frame_equal(a.loc[:cutoff - 1, cols], b.loc[:cutoff - 1, cols])


def test_rsi_bounds():
    s = pd.Series(np.random.default_rng(1).normal(0, 1, 200)).cumsum() + 100
    r = indicators.rsi(s).dropna()
    assert r.between(0, 100).all()


def test_edgar_taxonomy_priority():
    ev = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2024-03-05")], "cik": [7],
                       "prev_date": [pd.Timestamp("2024-03-04")], "next_date": [pd.Timestamp("2024-03-06")]})
    fl = pd.DataFrame({"cik": [7, 7], "form": ["8-K", "424B5"], "items": ["2.02,9.01", ""],
                       "accepted_et": pd.to_datetime(["2024-03-04 16:30", "2024-03-05 08:00"])})
    out = features.edgar_features(ev, fl)
    # an offering outranks the earnings explanation (dilution is the more damaging cause)
    assert out.loc[0, "edgar_category"] == "offering"
    assert out.loc[0, "item_2_02_results"] == 1


def test_exact_sim_stop_loss_and_costs():
    dates = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.DataFrame({"ticker": "X", "date": dates, "open": [10, 10, 9.5, 8, 8, 8.0],
                          "high": [10, 10.2, 9.6, 8.2, 8, 8.0], "low": [10, 9.8, 8.4, 7.9, 8, 8.0],
                          "close": [10, 10, 8.5, 8.0, 8, 8.0]})
    ev = pd.DataFrame({"ticker": ["X"], "date": [dates[0]], "next_date": [dates[1]], "event_family": ["shock_crash"],
                       "regime_state": ["normal"], "p": [0.9]})
    cfg = simulate.SimConfig(name="t", prob_col="p", min_prob=0.5, max_positions=1, hold_days=10, stop_loss=0.15,
                             initial_capital=1000)
    eq, tr = simulate.exact_sim(ev, panel, cfg)
    assert len(tr) == 1 and tr.iloc[0]["exit_reason"] == "stop_loss"
    assert tr.iloc[0]["exit_price"] == pytest.approx(8.5)      # stop at 10 * 0.85, not the lower low
    assert eq.iloc[-1] < 1000 * 0.85                           # loss plus round-trip costs


def test_tier_downgrade_in_stressed_regime():
    assert assign_tier(0.80, 0.55, "normal", 0.05) == "STRONG_BUY"
    assert assign_tier(0.80, 0.55, "stressed", 0.05) == "BUY"
    assert assign_tier(0.60, 0.55, "stressed", 0.01) == "WATCH"
    assert assign_tier(float("nan"), 0.55, "normal", None) == "NO_CALL"
