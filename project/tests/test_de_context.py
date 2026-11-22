"""Offline contract and point-in-time checks for German market context."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from driftrev import config, features, indicators
from driftrev.de_context import US_MACRO_COLS, context_columns_us, market_context_de


def _synthetic_market(tickers: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=400, name="date")
    frames = []
    for seed, ticker in enumerate(tickers, 1):
        rng = np.random.default_rng(seed)
        close = pd.Series(100 * np.exp(rng.normal(0, 0.01, len(dates)).cumsum()), index=dates)
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
            "high": close.to_numpy(), "low": close.to_numpy(), "close": close.to_numpy(),
            "volume": 0, "dividends": 0.0, "splits": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)


def _benchmark_close(market: pd.DataFrame, ticker: str) -> pd.Series:
    return market.loc[market["ticker"].eq(ticker)].set_index("date")["close"].sort_index()


@pytest.fixture
def de_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    market = _synthetic_market(["^GDAXI", "^MDAXI", "^SDAXI"])
    dates = pd.date_range(market["date"].min(), market["date"].max(), name="date")
    rng = np.random.default_rng(4)
    macro = pd.DataFrame({"bund10y": -0.4 + rng.normal(0, 0.01, len(dates)).cumsum()}, index=dates)
    return market, macro


def test_contract_superset_of_us_columns(de_inputs):
    market, macro = de_inputs
    ctx = market_context_de(market, macro)
    tickers = ["SPY", "IWM", "MDY", "^VIX", "^VIX3M", *sorted(set(config.SECTOR_ETFS.values()))]
    market_us = _synthetic_market(tickers)
    rng = np.random.default_rng(4)
    macro_us = pd.DataFrame({col: rng.normal(size=len(macro)) for col in config.FRED_SERIES.values()},
                            index=macro.index)
    us = context_columns_us(market_us, macro_us)
    assert set(ctx.columns) >= set(us) - {col for col in us if col.startswith("etf_ret_")}
    assert not any(col.startswith("etf_ret_") for col in ctx)
    assert set(US_MACRO_COLS) == set(config.FRED_SERIES.values())


def test_benchmark_mapping(de_inputs):
    market, macro = de_inputs
    ctx = market_context_de(market, macro)
    dax = _benchmark_close(market, "^GDAXI")
    sdax = _benchmark_close(market, "^SDAXI")
    mdax = _benchmark_close(market, "^MDAXI")
    pd.testing.assert_series_equal(ctx["spy_close"], dax, check_names=False)
    for n in (1, 5, 20):
        pd.testing.assert_series_equal(ctx[f"spy_ret_{n}d"], dax.pct_change(n), check_names=False)
    for n in (1, 20):
        pd.testing.assert_series_equal(ctx[f"iwm_ret_{n}d"], sdax.pct_change(n), check_names=False)
    pd.testing.assert_series_equal(ctx["small_vs_large_20d"], sdax.pct_change(20) - dax.pct_change(20),
                                   check_names=False)
    pd.testing.assert_series_equal(ctx["mdax_ret_20d"], mdax.pct_change(20), check_names=False)
    pd.testing.assert_series_equal(ctx["mid_vs_large_20d"], mdax.pct_change(20) - dax.pct_change(20),
                                   check_names=False)
    pd.testing.assert_series_equal(ctx["spy_above_200"], (dax > indicators.sma(dax, 200)).astype(float),
                                   check_names=False)
    pd.testing.assert_series_equal(ctx["spy_dd_252"], dax / dax.rolling(252, min_periods=60).max() - 1,
                                   check_names=False)
    opn = market.loc[market["ticker"].eq("^GDAXI")].set_index("date")["open"]
    for h in features.HORIZONS:
        pd.testing.assert_series_equal(ctx[f"spy_future_{h}d"], dax.shift(-h) / opn.shift(-1),
                                       check_names=False)
    # Only dates with a primary-benchmark close define the context, regardless of input order.
    sparse = market.copy()
    sparse.loc[sparse["ticker"].eq("^GDAXI") & sparse["date"].eq(dax.index[50]), "close"] = np.nan
    actual = market_context_de(sparse.iloc[::-1], macro)
    pd.testing.assert_index_equal(actual.index, dax.drop(dax.index[50]).index)


def test_volatility_gauge(de_inputs, monkeypatch):
    market, macro = de_inputs
    ctx = market_context_de(market, macro)
    dax = _benchmark_close(market, "^GDAXI")
    vol = indicators.realised_vol(dax, 20)
    term = indicators.realised_vol_term(dax, 20, 60)
    for col in ("vix", "rv_20"):
        pd.testing.assert_series_equal(ctx[col], vol, check_names=False)
    for col in ("vix_term", "rv_term"):
        pd.testing.assert_series_equal(ctx[col], term, check_names=False)
    pd.testing.assert_series_equal(ctx["vix_chg_5d"], vol - vol.shift(5), check_names=False)
    assert ctx["vix"].iloc[:20].isna().all()
    assert ctx["vix"].iloc[20:].notna().all()
    rv = config.RealisedVol(short_window=10, long_window=40, periods_per_year=365)
    monkeypatch.setattr(config, "SETTINGS", replace(config.SETTINGS, realised_vol=rv))
    custom = market_context_de(market, macro)
    vol = indicators.realised_vol(dax, 10, 365)
    term = indicators.realised_vol_term(dax, 10, 40, 365)
    for col in ("vix", "rv_20"):
        pd.testing.assert_series_equal(custom[col], vol, check_names=False)
    for col in ("vix_term", "rv_term"):
        pd.testing.assert_series_equal(custom[col], term, check_names=False)


def test_correction_matches_features(de_inputs):
    market, macro = de_inputs
    ctx = market_context_de(market, macro)
    expected = features.correction_state(_benchmark_close(market, "^GDAXI"))
    pd.testing.assert_frame_equal(ctx[expected.columns], expected)


def test_bund_lagged_one_session(de_inputs):
    market, macro = de_inputs
    ctx = market_context_de(market, macro)
    assert np.isnan(ctx["bund10y"].iloc[0])
    for previous, date in zip(ctx.index[:-1], ctx.index[1:]):
        known = macro.loc[macro.index <= previous, "bund10y"].dropna()
        assert ctx.loc[date, "bund10y"] == known.iloc[-1]
    pd.testing.assert_series_equal(ctx["bund10y_chg_20d"], ctx["bund10y"] - ctx["bund10y"].shift(20),
                                   check_names=False)
    # A weekend quote must carry into Monday, then become available on Tuesday.
    weekends = macro.loc[macro.index.dayofweek >= 5]
    weekend_ctx = market_context_de(market, weekends)
    tuesday = ctx.index[ctx.index.dayofweek == 1][0]
    sunday = tuesday - pd.Timedelta(days=2)
    assert weekend_ctx.loc[tuesday, "bund10y"] == weekends.loc[sunday, "bund10y"]
    assert weekend_ctx["bund10y"].iloc[:4].isna().all()
    for absent in (pd.DataFrame(), pd.DataFrame(index=macro.index), macro.rename(columns={"bund10y": "other"})):
        missing = market_context_de(market, absent)
        assert missing[["bund10y", "bund10y_chg_20d"]].isna().all().all()


def test_us_only_columns_are_nan(de_inputs):
    ctx = market_context_de(*de_inputs)
    columns = ["sector_corr_20d", "sector_corr_pctile", *US_MACRO_COLS, "credit_chg_20d", "curve_10y2y"]
    assert ctx[columns].isna().all().all()


def test_point_in_time(de_inputs):
    market, macro = de_inputs
    full = market_context_de(market, macro)
    cutoff = full.index[299]
    truncated = market_context_de(market.loc[market["date"] <= cutoff], macro.loc[macro.index <= cutoff])
    columns = [col for col in full if not col.startswith("spy_future_")]
    assert len(truncated) == 300
    pd.testing.assert_frame_equal(truncated[columns], full[columns].iloc[:300])


def test_missing_benchmarks(de_inputs):
    market, macro = de_inputs
    dax_only = market.loc[market["ticker"].eq("^GDAXI")]
    ctx = market_context_de(dax_only, macro)
    columns = ["iwm_ret_1d", "iwm_ret_20d", "small_vs_large_20d", "mdax_ret_20d", "mid_vs_large_20d"]
    assert ctx[columns].isna().all().all()
    pd.testing.assert_index_equal(ctx.index, _benchmark_close(market, "^GDAXI").index)
    for missing in (market.loc[market["ticker"].ne("^GDAXI")], market.iloc[:0]):
        with pytest.raises(ValueError, match=r"market_context_de needs \^GDAXI"):
            market_context_de(missing, macro)


def test_regime_table_accepts_de_context(de_inputs):
    ctx = market_context_de(*de_inputs)
    panel = pd.concat([
        pd.DataFrame({"ticker": ticker, "date": ctx.index,
                      "close_sma200": np.where((np.arange(len(ctx)) + i) % 4 == 0, 0.9, 1.1),
                      "is_crash": (np.arange(len(ctx)) + i) % 50 == 0})
        for i, ticker in enumerate(("FAKE1", "FAKE2", "FAKE3"))
    ], ignore_index=True)
    reg = features.regime_table(ctx, panel)
    assert set(reg["regime_state"]) <= {"normal", "elevated", "stressed"}
    expected = (ctx["vix_term"] > config.SETTINGS.regime.vix_term_ratio_trigger).astype(int)
    pd.testing.assert_series_equal(reg["flag_vix_term"], expected, check_names=False)
