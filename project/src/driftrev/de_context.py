"""German market context using the market-neutral column contract from features.

The historical US names describe roles in every market: ``spy_*`` is the primary
large-cap benchmark, ``iwm_*`` the small-cap benchmark, and ``vix*`` the volatility
gauge and its term ratio. For Germany these are DAX, SDAX and DAX realised
volatility (no VDAX is available). The US implementation remains unchanged.

Column mapping (P = DAX close, O = DAX open, S = SDAX close, M = MDAX close,
all aligned to dates with a DAX close):

Column                         Definition
-----------------------------  ------------------------------------------------
spy_close                      P
spy_ret_1d/5d/20d               P.pct_change(1/5/20)
spy_above_200                  (P > ind.sma(P, 200)).astype(float)
spy_dd_252                     P / P.rolling(252, min_periods=60).max() - 1
in_correction                  features.correction_state(P): correction flag
correction_depth               features.correction_state(P): drawdown from ATH
days_in_correction             features.correction_state(P): consecutive sessions
iwm_ret_1d/20d                 S.pct_change(1/20)
small_vs_large_20d             iwm_ret_20d - spy_ret_20d
vix                           ind.realised_vol(P, short_window, periods_per_year)
vix_chg_5d                     vix - vix.shift(5)
vix_term                      ind.realised_vol_term(P, short_window, long_window,
                                                    periods_per_year)
spy_future_{h}d                P.shift(-h) / O.shift(-1), h in features.HORIZONS
sector_corr_20d/pctile         NaN (German sector series are a later task)
US_MACRO_COLS                  NaN (dgs10, dgs5, dgs2, fedfunds, credit_spread, t10y2y)
credit_chg_20d, curve_10y2y     NaN (US-only concepts)
bund10y                       Forward-fill on the union of macro and DAX dates,
                              reindex to DAX dates, then shift(1); NaN if absent
bund10y_chg_20d                bund10y - bund10y.shift(20)
mdax_ret_20d                   M.pct_change(20)
mid_vs_large_20d               mdax_ret_20d - spy_ret_20d
rv_20, rv_term                 vix, vix_term (explicit German names)

RealisedVol windows and annualisation come from config.SETTINGS at call time.
Missing SDAX or MDAX produce NaN columns; missing DAX raises ValueError. There
are no etf_ret_* columns. No look-ahead: every non-target column uses only data
available by that row's close, with Bund yields lagged one trading session as
they are published after the close. Only spy_future_* targets use future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features
from . import indicators as ind

US_MACRO_COLS = ("dgs10", "dgs5", "dgs2", "fedfunds", "credit_spread", "t10y2y")


def market_context_de(market: pd.DataFrame, macro_wide: pd.DataFrame) -> pd.DataFrame:
    """Transform German benchmark prices and Bund yields into daily market context."""
    close = market.pivot_table(index="date", columns="ticker", values="close").sort_index()
    if "^GDAXI" not in close:
        raise ValueError("market_context_de needs ^GDAXI")
    dax = close["^GDAXI"].dropna()
    ctx = pd.DataFrame(index=dax.index)
    ctx["spy_close"] = dax
    ctx["spy_ret_1d"] = dax.pct_change()
    ctx["spy_ret_5d"] = dax.pct_change(5)
    ctx["spy_ret_20d"] = dax.pct_change(20)
    ctx["spy_above_200"] = (dax > ind.sma(dax, 200)).astype(float)
    ctx["spy_dd_252"] = dax / dax.rolling(252, min_periods=60).max() - 1
    ctx = ctx.join(features.correction_state(dax))

    ctx["iwm_ret_1d"] = np.nan
    ctx["iwm_ret_20d"] = np.nan
    if "^SDAXI" in close:
        sdax = close["^SDAXI"].reindex(ctx.index).ffill()  # explicit pad: same values as pct_change's old default
        ctx["iwm_ret_1d"] = sdax.pct_change()
        ctx["iwm_ret_20d"] = sdax.pct_change(20)
    ctx["small_vs_large_20d"] = ctx["iwm_ret_20d"] - ctx["spy_ret_20d"]
    rv = config.SETTINGS.realised_vol
    ctx["vix"] = ind.realised_vol(dax, rv.short_window, rv.periods_per_year)
    ctx["vix_chg_5d"] = ctx["vix"] - ctx["vix"].shift(5)
    ctx["vix_term"] = ind.realised_vol_term(dax, rv.short_window, rv.long_window, rv.periods_per_year)

    opn = market.pivot_table(index="date", columns="ticker", values="open").sort_index()
    dax_open = opn.reindex(index=ctx.index, columns=["^GDAXI"])["^GDAXI"]
    for h in features.HORIZONS:
        ctx[f"spy_future_{h}d"] = dax.shift(-h) / dax_open.shift(-1)

    for col in ("sector_corr_20d", "sector_corr_pctile", *US_MACRO_COLS, "credit_chg_20d", "curve_10y2y"):
        ctx[col] = np.nan
    ctx["bund10y"] = np.nan
    if "bund10y" in macro_wide and not macro_wide.empty:
        bund = macro_wide["bund10y"]
        ctx["bund10y"] = (bund.reindex(ctx.index.union(bund.index)).sort_index().ffill()
                         .reindex(ctx.index).shift(1))
    ctx["bund10y_chg_20d"] = ctx["bund10y"] - ctx["bund10y"].shift(20)

    ctx["mdax_ret_20d"] = np.nan
    if "^MDAXI" in close:
        mdax = close["^MDAXI"].reindex(ctx.index).ffill()
        ctx["mdax_ret_20d"] = mdax.pct_change(20)
    ctx["mid_vs_large_20d"] = ctx["mdax_ret_20d"] - ctx["spy_ret_20d"]
    ctx["rv_20"] = ctx["vix"]
    ctx["rv_term"] = ctx["vix_term"]
    return ctx


def context_columns_us(market_us: pd.DataFrame, macro_wide_us: pd.DataFrame) -> list[str]:
    """Columns of features.market_context for the given US inputs (helper used by the contract test)."""
    return list(features.market_context(market_us, macro_wide_us).columns)
