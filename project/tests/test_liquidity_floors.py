"""Offline market-specific event and rolling-window liquidity floors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_de_pipeline import _build, _ohlcv, _world
from test_de_pipeline import de_env as de_env

from driftrev import config, de_membership, de_pipeline, de_universe, features, prices, rolling_window, store


def test_us_values_match_existing_constants():
    assert config.MARKETS["us"].min_dollar_volume == config.EventRule().min_dollar_volume
    assert config.MARKETS["us"].breakout_min_median_turnover == rolling_window.MIN_MEDIAN_DOLLAR_VOL
    assert config.MARKETS["de"].min_dollar_volume == 1e6
    assert config.MARKETS["de"].breakout_min_median_turnover == 5e6


def test_ticker_features_floor():
    dates = pd.bdate_range("2020-01-01", periods=300, name="date")
    stock = _ohlcv("LOW.DE", dates, 1, base=50.0, volume=30_000).assign(
        open=50.0, high=50.5, low=49.5, close=50.0)
    ctx = pd.DataFrame({"spy_close": 100.0, "spy_ret_1d": 0.0}, index=dates)
    for horizon in features.HORIZONS:
        ctx[f"spy_future_{horizon}d"] = 1.0
    default = features.ticker_features(stock, ctx, None)
    lower = features.ticker_features(stock, ctx, None, min_dollar_volume=1e6)
    defined = default["dollar_vol_20"].notna()
    assert defined.any()
    assert default.loc[defined, "dollar_vol_20"].eq(1.5e6).all()
    assert not default.loc[defined, "is_liquid"].any()
    assert lower.loc[defined, "is_liquid"].all()


def test_arm_flags_floor():
    frame = pd.DataFrame({
        "rw_median_dollar_vol": [7e6], "rw_prior_strength": [1.0], "growth_20d": [1.1],
        "rw_breakout_5d": [1.0], "rw_days_since_high": [4.0], "rw_pullback_atr": [1.0],
        "close_loc": [0.9], "rw_volume_mult_med": [2.0], "rw_release_20_3": [1.0],
    })
    default = rolling_window.arm_flags(frame)
    lower = rolling_window.arm_flags(frame, min_median_turnover=5e6)
    assert default.loc[0, "arm_a"] == 0
    assert lower.loc[0, "arm_a"] == 1
    assert default.eq(0).all().all()
    assert lower.eq(1).all().all()


def test_prior_strength_floor():
    dates = pd.bdate_range("2020-01-01", periods=60, name="date")
    names = ("LOW", "MID", "HIGH")
    close = pd.DataFrame({name: 50 * (1 + rate) ** np.arange(len(dates))
                          for name, rate in zip(names, (0.001, 0.002, 0.003))}, index=dates)
    turnover = pd.DataFrame(dict(zip(names, (3e6, 7e6, 20e6))), index=dates)
    benchmark = pd.Series(100.0, index=dates)
    default = rolling_window.prior_strength(close, turnover, benchmark)
    lower = rolling_window.prior_strength(close, turnover, benchmark, min_median_turnover=5e6)
    assert not default.iloc[:21].notna().any().any()
    assert not lower.iloc[:21].notna().any().any()
    assert set(default.columns[default.notna().any()]) == {"HIGH"}
    assert set(lower.columns[lower.notna().any()]) == {"MID", "HIGH"}
    assert default.loc[dates[21]:, "HIGH"].eq(1.0).all()
    assert lower.loc[dates[21]:, "MID"].eq(0.5).all()
    assert lower.loc[dates[21]:, "HIGH"].eq(1.0).all()


def test_build_all_defaults_unchanged():
    dates = pd.bdate_range("2020-01-01", periods=300, name="date")
    market = pd.concat([_ohlcv(ticker, dates, seed) for seed, ticker in
                        enumerate(("SPY", "IWM", "MDY", "^VIX", "^VIX3M"), 1)], ignore_index=True)
    stocks = pd.concat([_ohlcv(ticker, dates, seed, volume=1e6, crash=True)
                        for seed, ticker in enumerate(("AAA", "BBB"), 10)], ignore_index=True)
    uni = pd.DataFrame({"ticker": ["AAA", "BBB"], "company": ["Alpha", "Beta"], "sector": "unknown",
                        "sub_industry": None, "cap_bucket": "large", "cik": [1, 2],
                        "wikipedia_title": None, "sector_etf": None, "is_current_member": True})
    membership = pd.DataFrame({"ticker": uni["ticker"], "index_name": "sp500",
                               "start_date": dates[0], "end_date": pd.NaT})
    filings = pd.DataFrame({"cik": [1, 2], "form": "8-K", "items": "2.02",
                            "accepted_et": dates[200] + pd.Timedelta(hours=8)})
    for fl in (filings, pd.DataFrame()):
        args = (uni, stocks, market, pd.DataFrame(), fl, pd.DataFrame(), pd.DataFrame())
        default = features.build_all(*args, membership=membership)
        explicit_none = features.build_all(*args, membership=membership, min_dollar_volume=None,
                                           breakout_min_median_turnover=None)
        explicit_us = features.build_all(*args, membership=membership, min_dollar_volume=2e6,
                                         breakout_min_median_turnover=10e6)
        assert not default[0].empty
        for original, none_frame, us_frame in zip(default, explicit_none, explicit_us):
            pd.testing.assert_frame_equal(original, none_frame)
            pd.testing.assert_frame_equal(original, us_frame)


@pytest.mark.usefixtures("de_env")
def test_transform_de_uses_de_floors():
    universe, membership, stocks, market, macro = _world()
    dates = macro.index
    low = _ohlcv("LOW.DE", dates, 20, base=50.0, volume=30_000, crash=True)
    low["close"] = np.where(np.arange(len(dates)) < 250, 50.0, 42.5)
    low["open"] = low["close"].shift(1).fillna(50.0)
    low["high"] = low[["open", "close"]].max(axis=1) * 1.01
    low["low"] = low[["open", "close"]].min(axis=1) * 0.99
    stocks = pd.concat([stocks, low], ignore_index=True)
    universe = pd.concat([universe, universe.iloc[[0]].assign(
        company_key="Low", name="Low AG", symbol="LOW.DE", price_source="xetra",
        yahoo_name="Low AG", matched_query="Low")], ignore_index=True)
    membership = pd.concat([membership, membership.iloc[[0]].assign(
        company_key="Low", name="Low AG")], ignore_index=True)
    for table, frame in ((de_universe.TABLE, universe), (de_membership.TABLE, membership),
                         (prices.PRICE_TABLE, stocks), (prices.MARKET_TABLE, market),
                         ("macro", macro.rename_axis("date").reset_index().rename(columns={"bund10y": "value"})
                          .assign(series="bund10y"))):
        store.replace_table(table, frame)
    result = de_pipeline.transform_de()
    events = result["events"].loc[result["events"]["ticker"].eq("LOW.DE")]
    assert len(events) == 1
    event = events.iloc[0]
    assert event["date"] == dates[250]
    assert event["event_family"] == "shock_crash"
    assert event["cap_bucket"] == "large"
    assert event["dollar_vol_20"] == 1.5e6
    assert np.isclose(event["ret_1d"], -0.15)
    assert event["volume_mult"] == 5
    assert event["is_liquid"]
    assert config.WAREHOUSE.exists()
    us_events, us_panel, _ = _build((universe, membership, stocks, market, macro))
    assert "LOW.DE" not in set(us_events["ticker"])
    assert not us_panel.loc[us_panel["ticker"].eq("LOW.DE"), "is_liquid"].any()
