"""Transformations: raw warehouse tables -> (1) a daily panel and (2) the event dataset.

Pipeline (each step is a function, called by `build_all`):

1. `market_context`      SPY/IWM/VIX/sector-ETF/FRED features by date (FRED lagged 1 row).
2. `ticker_features`     ~45 per-stock technical / momentum / volume / liquidity features
                         plus forward targets, for one ticker at a time (memory friendly).
3. `earnings_days`       8-K Item 2.02 acceptance time -> first session that can react.
4. event detection       earnings reaction days + non-earnings crash days, liquidity-filtered.
5. `edgar_features`      filing taxonomy around each event + prior-filing activity counts.
6. `attention_features`  Wikipedia pageview spikes (lagged one day).
7. `sue_features`        latest point-in-time SUE known at the event date.
8. `regime_table`        TradePulse-style fragility monitor (needs the whole daily panel).

No look-ahead: features at row t use data available by the close of t (filings up to the
next morning's 09:00 ET, because trades are entered at the next open). Targets use t+1..t+H.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, indicators as ind, rolling_window as rw

log = logging.getLogger(__name__)
S = config.SETTINGS
HORIZONS = (5, 10, 20)


# --------------------------------------------------------------------------------------
# 1. Market context
# --------------------------------------------------------------------------------------
def market_context(market: pd.DataFrame, macro_wide: pd.DataFrame) -> pd.DataFrame:
    close = market.pivot_table(index="date", columns="ticker", values="close").sort_index()
    opn = market.pivot_table(index="date", columns="ticker", values="open").sort_index()
    spy = close["SPY"].dropna()
    ctx = pd.DataFrame(index=spy.index)
    ctx["spy_close"] = spy
    ctx["spy_ret_1d"] = spy.pct_change()
    ctx["spy_ret_5d"] = spy.pct_change(5)
    ctx["spy_ret_20d"] = spy.pct_change(20)
    ctx["spy_above_200"] = (spy > ind.sma(spy, 200)).astype(float)
    ctx["spy_dd_252"] = spy / spy.rolling(252, min_periods=60).max() - 1
    iwm = close["IWM"].reindex(ctx.index)
    ctx["iwm_ret_1d"] = iwm.pct_change()
    ctx["iwm_ret_20d"] = iwm.pct_change(20)
    ctx["small_vs_large_20d"] = ctx["iwm_ret_20d"] - ctx["spy_ret_20d"]
    vix = close["^VIX"].reindex(ctx.index).ffill()
    vix3m = close["^VIX3M"].reindex(ctx.index).ffill()
    ctx["vix"] = vix
    ctx["vix_chg_5d"] = vix - vix.shift(5)
    ctx["vix_term"] = vix / vix3m
    # forward benchmark returns over the same entry/exit convention as the stock targets
    spy_open = opn["SPY"].reindex(ctx.index)
    for h in HORIZONS:
        ctx[f"spy_future_{h}d"] = spy.shift(-h) / spy_open.shift(-1)
    # sector ETF daily returns for sector-relative features
    for etf in sorted(set(config.SECTOR_ETFS.values())):
        if etf in close:
            ctx[f"etf_ret_{etf}"] = close[etf].reindex(ctx.index).pct_change()
    # sector co-movement: mean pairwise 20d correlation across the sector ETFs
    etf_rets = ctx[[c for c in ctx if c.startswith("etf_ret_")]]
    ctx["sector_corr_20d"] = _avg_pairwise_corr(etf_rets, S.regime.corr_window_days)
    ctx["sector_corr_pctile"] = ctx["sector_corr_20d"].rolling(252, min_periods=120).rank(pct=True)
    # FRED: forward-fill to trading days, then lag one row (published after the close)
    if not macro_wide.empty:
        m = macro_wide.reindex(ctx.index.union(macro_wide.index)).sort_index().ffill().reindex(ctx.index).shift(1)
        for col in m.columns:
            ctx[col] = m[col]
        if "credit_spread" in m:
            ctx["credit_chg_20d"] = m["credit_spread"] - m["credit_spread"].shift(20)
        if {"dgs10", "dgs2"} <= set(m.columns):
            ctx["curve_10y2y"] = m["dgs10"] - m["dgs2"]
    return ctx


def _avg_pairwise_corr(rets: pd.DataFrame, window: int) -> pd.Series:
    arr = rets.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for end in range(window, len(arr) + 1):
        w = arr[end - window:end]
        w = w[:, np.isfinite(w).all(axis=0)]
        if w.shape[1] < 5:
            continue
        c = np.corrcoef(w, rowvar=False)
        k = c.shape[0]
        out[end - 1] = (c.sum() - k) / (k * (k - 1))
    return pd.Series(out, index=rets.index)


# --------------------------------------------------------------------------------------
# 2. Per-ticker features
# --------------------------------------------------------------------------------------
def ticker_features(g: pd.DataFrame, ctx: pd.DataFrame, sector_etf: str | None,
                    strength: pd.Series | None = None) -> pd.DataFrame:
    """g: one ticker's OHLCV rows sorted by date. Returns g with feature + target columns."""
    g = g.sort_values("date").reset_index(drop=True)
    o, h, l, c, v = (g[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    div = g["dividends"].fillna(0).astype(float)
    f = pd.DataFrame({"ticker": g["ticker"], "date": g["date"], "open": o, "high": h, "low": l, "close": c, "volume": v})

    ret = (c + div) / c.shift(1) - 1
    tri = (1 + ret.fillna(0)).cumprod()          # total-return index
    f["ret_1d"] = ret
    for k in (3, 5, 10, 20, 60, 120, 252):
        f[f"growth_{k}d"] = tri / tri.shift(k)
    f["mom_12_1"] = tri.shift(21) / tri.shift(252) - 1
    for n in (20, 50, 200):
        f[f"close_sma{n}"] = c / ind.sma(c, n)
    f["sma50_sma200"] = ind.sma(c, 50) / ind.sma(c, 200)
    f["dist_52w_high"] = c / c.rolling(252, min_periods=120).max() - 1
    f["dist_52w_low"] = c / c.rolling(252, min_periods=120).min() - 1
    f["vol_20d"] = ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    f["vol_60d"] = ret.rolling(60, min_periods=60).std() * np.sqrt(252)
    f["vol_ratio_20_60"] = f["vol_20d"] / f["vol_60d"]
    f["atr_pct"] = ind.atr(h, l, c) / c
    f["rsi_14"] = ind.rsi(c)
    f["macd_hist_pct"] = ind.macd_hist(c) / c
    f["bb_pctb"] = ind.bollinger_pctb(c)
    f["stoch_k"] = ind.stoch_k(h, l, c)
    f["cci_20"] = ind.cci(h, l, c)
    f["mfi_14"] = ind.mfi(h, l, c, v)
    f["adx_14"] = ind.adx(h, l, c)
    f["obv_slope_20"] = ind.obv_slope(c, v)
    avg_v20 = ind.sma(v, 20).shift(1)          # average of the 20 sessions BEFORE t
    f["volume_mult"] = v / avg_v20
    lv = np.log1p(v)
    f["volume_z"] = (lv - lv.shift(1).rolling(60, min_periods=60).mean()) / lv.shift(1).rolling(60, min_periods=60).std()
    f["dollar_vol_20"] = ind.sma(c * v, 20).shift(1)
    f["log_dollar_vol"] = np.log10(f["dollar_vol_20"].clip(lower=1))
    pc = c.shift(1)
    f["gap_open"] = o / pc - 1
    f["range_pct"] = (h - l) / pc
    f["close_loc"] = (c - l) / (h - l).replace(0, np.nan)
    f["ret_open_close"] = c / o - 1
    f["down_streak_prior"] = ind.streak(ret, -1).shift(1)
    f["up_streak_prior"] = ind.streak(ret, 1).shift(1)

    m = ctx.reindex(g["date"]).reset_index(drop=True)
    beta = ind.rolling_beta(ret, m["spy_ret_1d"])
    f["beta_252"] = beta
    f["abn_ret_1d"] = ret - beta.fillna(1.0) * m["spy_ret_1d"]
    f["idio_vol_60"] = (ret - beta.fillna(1.0) * m["spy_ret_1d"]).rolling(60, min_periods=60).std() * np.sqrt(252)
    col = f"etf_ret_{sector_etf}" if sector_etf else None
    f["ret_vs_sector"] = ret - m[col] if col and col in m else np.nan
    f["rs_vs_spy_60d"] = f["growth_60d"] / (m["spy_close"] / m["spy_close"].shift(60))

    # ---- rolling-window pressure & release (strategy guide, Variant B) ----
    bench = m["spy_close"].astype(float)
    f = pd.concat([f, rw.window_features(c, bench), rw.setup_features(o, h, l, c, v)], axis=1)
    f["rw_prior_strength"] = strength.reindex(g["date"]).to_numpy() if strength is not None else np.nan
    f = pd.concat([f, rw.arm_flags(f)], axis=1)

    ev = S.event
    liquid = (c >= ev.min_price) & (f["dollar_vol_20"] >= ev.min_dollar_volume)
    f["is_liquid"] = liquid
    f["is_crash"] = (ret <= ev.max_ret_1d) & (f["volume_mult"] >= ev.min_volume_mult) & liquid
    f["n_crash_prior_252"] = f["is_crash"].astype(float).rolling(252, min_periods=1).sum().shift(1)
    f["prev_date"] = g["date"].shift(1)
    f["next_date"] = g["date"].shift(-1)

    # ---- targets: enter at next open, exit at close of t+H ----
    entry = o.shift(-1)
    for hz in HORIZONS:
        f[f"growth_future_{hz}d"] = c.shift(-hz) / entry
        f[f"growth_future_{hz}d_vs_spy"] = f[f"growth_future_{hz}d"] / m[f"spy_future_{hz}d"]
    hz = S.target.horizon
    lows = pd.concat([l.shift(-k) for k in range(1, hz + 1)], axis=1).min(axis=1, skipna=False)
    highs = pd.concat([h.shift(-k) for k in range(1, hz + 1)], axis=1).max(axis=1, skipna=False)
    f[f"mae_{hz}d"] = lows / entry - 1      # worst intraperiod move (for stop-loss analysis)
    f[f"mfe_{hz}d"] = highs / entry - 1
    return f


# --------------------------------------------------------------------------------------
# 3. Earnings reaction days
# --------------------------------------------------------------------------------------
def earnings_days(filings: pd.DataFrame, cik: int, dates: np.ndarray) -> pd.DataFrame:
    """Map each 8-K Item 2.02 to the first trading session able to react to it.

    after 16:00 ET -> next session; before 09:30 or intraday -> same session (or next
    session for weekend / holiday releases)."""
    fl = filings[(filings["cik"] == cik) & (filings["form"] == "8-K") & filings["items"].str.contains("2.02", regex=False)]
    if fl.empty or len(dates) == 0:
        return pd.DataFrame(columns=["date", "earn_accepted_et"])
    ts = pd.to_datetime(fl["accepted_et"])
    day = ts.dt.normalize()
    after_close = (ts.dt.hour * 60 + ts.dt.minute) >= 16 * 60
    side_date = np.where(after_close, (day + pd.Timedelta(days=1)).values, day.values)
    idx = np.searchsorted(dates, side_date, side="left")
    ok = idx < len(dates)
    out = pd.DataFrame({"date": dates[idx[ok]], "earn_accepted_et": ts.values[ok]})
    return out.sort_values("earn_accepted_et").drop_duplicates("date", keep="first")


# --------------------------------------------------------------------------------------
# 5. EDGAR taxonomy around each event
# --------------------------------------------------------------------------------------
OFFERING_FORMS = ("424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR", "FWP")
ITEM_FLAGS = {
    "1.01": "item_1_01_agreement", "1.02": "item_1_02_termination", "1.03": "item_1_03_bankruptcy",
    "2.01": "item_2_01_acquisition", "2.02": "item_2_02_results", "2.05": "item_2_05_restructuring",
    "2.06": "item_2_06_impairment", "3.01": "item_3_01_delisting", "3.02": "item_3_02_unreg_equity",
    "4.01": "item_4_01_auditor", "4.02": "item_4_02_nonreliance", "5.02": "item_5_02_officer",
    "7.01": "item_7_01_regfd", "8.01": "item_8_01_other",
}
CATEGORY_ORDER = [  # first match wins: the most damaging explanation dominates
    ("bankruptcy_delisting", lambda w: w["item_1_03_bankruptcy"] or w["item_3_01_delisting"] or w["form_delist"]),
    ("restatement_auditor", lambda w: w["item_4_02_nonreliance"] or w["item_4_01_auditor"] or w["form_nt"]),
    ("offering", lambda w: w["form_offering"] or w["item_3_02_unreg_equity"]),
    ("earnings", lambda w: w["item_2_02_results"] or w["form_periodic"]),
    ("impairment_restructuring", lambda w: w["item_2_06_impairment"] or w["item_2_05_restructuring"]),
    ("mgmt_change", lambda w: w["item_5_02_officer"]),
    ("deal", lambda w: w["item_1_01_agreement"] or w["item_1_02_termination"] or w["item_2_01_acquisition"]),
    ("regfd_other_8k", lambda w: w["item_7_01_regfd"] or w["item_8_01_other"] or w["form_8k"]),
]


def edgar_features(events: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """events: ticker, date, cik, prev_date, next_date. Returns one row per event with
    window flags (prev close 16:00 ET .. next morning 09:00 ET), the taxonomy category and
    prior-activity counts."""
    cols = list(ITEM_FLAGS.values()) + ["form_offering", "form_nt", "form_delist", "form_periodic", "form_8k"]
    rows = []
    fl = filings.dropna(subset=["accepted_et"]).sort_values("accepted_et")
    by_cik = {k: g for k, g in fl.groupby("cik")}
    for cik, evs in events.groupby("cik"):
        g = by_cik.get(cik)
        start = (pd.to_datetime(evs["prev_date"]).fillna(evs["date"] - pd.Timedelta(days=1)) + pd.Timedelta(hours=16)).values
        end = (pd.to_datetime(evs["next_date"]).fillna(evs["date"] + pd.Timedelta(days=1)) + pd.Timedelta(hours=9)).values
        if g is None:
            for i in evs.index:
                rows.append({"_idx": i, **{c: 0 for c in cols}, "edgar_n_window": 0})
            continue
        t = g["accepted_et"].values
        forms, items = g["form"].values, g["items"].values
        lo, hi = np.searchsorted(t, start, "right"), np.searchsorted(t, end, "right")
        for i, a, b in zip(evs.index, lo, hi):
            fw, iw = forms[a:b], ",".join(items[a:b])
            rec = {"_idx": i, "edgar_n_window": int(b - a)}
            for code, name in ITEM_FLAGS.items():
                rec[name] = int(code in iw)
            rec["form_offering"] = int(any(f in OFFERING_FORMS for f in fw))
            rec["form_nt"] = int(any(f.startswith("NT 10") for f in fw))
            rec["form_delist"] = int(any(f in ("25-NSE", "15-12B") for f in fw))
            rec["form_periodic"] = int(any(f in ("10-Q", "10-K") for f in fw))
            rec["form_8k"] = int(any(f.startswith("8-K") for f in fw))
            rows.append(rec)
        # prior activity counts (strictly before the event day's open)
        day0 = (evs["date"] + pd.Timedelta(hours=9, minutes=30)).values
        for name, mask, days in (
            ("n_offering_365d", np.isin(forms, OFFERING_FORMS), 365),
            ("n_8k_90d", np.char.startswith(forms.astype(str), "8-K"), 90),
            ("n_form4_30d", forms == "4", 30),
            ("n_13d_180d", np.char.startswith(forms.astype(str), "SC 13D") | np.char.startswith(forms.astype(str), "SCHEDULE 13D"), 180),
            ("n_late_filing_365d", np.char.startswith(forms.astype(str), "NT 10"), 365),
        ):
            tt = t[mask]
            cnt = np.searchsorted(tt, day0, "left") - np.searchsorted(tt, day0 - np.timedelta64(days, "D"), "left")
            for i, n in zip(evs.index, cnt):
                rows.append({"_idx": i, name: int(n)})
    df = pd.DataFrame(rows).groupby("_idx").first()
    df = df.reindex(events.index)
    count_cols = ["n_offering_365d", "n_8k_90d", "n_form4_30d", "n_13d_180d", "n_late_filing_365d"]
    df[cols + ["edgar_n_window"] + count_cols] = df[cols + ["edgar_n_window"] + count_cols].fillna(0).astype(int)
    df["edgar_category"] = "none"
    for name, rule in reversed(CATEGORY_ORDER):  # reversed so the first rule overwrites last
        hit = df.apply(lambda r: bool(rule(r)), axis=1)
        df.loc[hit, "edgar_category"] = name
    df["edgar_items"] = df[list(ITEM_FLAGS.values())].apply(
        lambda r: ",".join(k for k, n in ITEM_FLAGS.items() if r[n]), axis=1)
    return df


# --------------------------------------------------------------------------------------
# 6. Wikipedia attention
# --------------------------------------------------------------------------------------
def attention_features(pageviews: pd.DataFrame) -> pd.DataFrame:
    """Per article and calendar day: attention known at the START of that day
    (i.e. from views up to the previous day)."""
    if pageviews.empty:
        return pd.DataFrame(columns=["wikipedia_title", "date", "att_ratio_7_90", "att_spike_1d", "att_level"])
    out = []
    for title, g in pageviews.groupby("wikipedia_title"):
        s = g.set_index("date")["views"].asfreq("D")
        s = s.shift(1)  # day t only knows views through t-1
        recent = s.rolling(7, min_periods=5).mean()
        base = s.shift(7).rolling(90, min_periods=60).mean()
        med = s.shift(1).rolling(60, min_periods=40).median()
        out.append(pd.DataFrame({
            "wikipedia_title": title, "date": s.index,
            "att_ratio_7_90": np.log1p(recent) - np.log1p(base),
            "att_spike_1d": np.log1p(s) - np.log1p(med),
            "att_level": np.log1p(base),
        }))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------
# 7. Point-in-time SUE
# --------------------------------------------------------------------------------------
def sue_features(events: pd.DataFrame, sue: pd.DataFrame) -> pd.DataFrame:
    """Latest SUE whose 10-Q/10-K was filed on or before the event date."""
    if sue.empty:
        return pd.DataFrame(index=events.index, columns=["sue", "eps_yoy_pos", "sue_age_days", "sue_is_current"])
    left = events[["cik", "date"]].reset_index().dropna(subset=["cik"]).sort_values("date")
    left["cik"] = left["cik"].astype("int64")
    right = sue.dropna(subset=["filed"]).sort_values("filed")[["cik", "filed", "period_end", "sue", "eps_yoy_pos"]]
    right["cik"] = right["cik"].astype("int64")
    m = pd.merge_asof(left, right, left_on="date", right_on="filed", by="cik", direction="backward")
    m["sue_age_days"] = (m["date"] - m["filed"]).dt.days
    m["sue_is_current"] = ((m["date"] - m["period_end"]).dt.days <= 100) & (m["sue_age_days"] <= 3)
    return m.set_index("index")[["sue", "eps_yoy_pos", "sue_age_days", "sue_is_current"]].reindex(events.index)


# --------------------------------------------------------------------------------------
# 8. Regime monitor (TradePulse fragility idea)
# --------------------------------------------------------------------------------------
def regime_table(ctx: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    r = S.regime
    breadth = panel.assign(a=panel["close_sma200"] > 1).dropna(subset=["close_sma200"]).groupby("date")["a"].mean()
    crash_breadth = panel.groupby("date")["is_crash"].mean()
    reg = pd.DataFrame(index=ctx.index)
    reg["breadth_above_200"] = breadth.reindex(reg.index)
    reg["crash_breadth"] = crash_breadth.reindex(reg.index)
    flags = pd.DataFrame(index=reg.index)
    flags["flag_trend"] = ctx["spy_above_200"].eq(0) & ctx["spy_close"].notna()
    flags["flag_vix_term"] = ctx["vix_term"] > r.vix_term_ratio_trigger
    flags["flag_corr"] = ctx["sector_corr_pctile"] > r.corr_pctile_trigger
    flags["flag_breadth"] = reg["breadth_above_200"] < r.breadth_trigger
    flags["flag_credit"] = ctx.get("credit_chg_20d", pd.Series(np.nan, index=reg.index)) > r.credit_change_trigger
    reg = reg.join(flags.astype(int))
    reg["regime_points"] = flags.sum(axis=1).astype(int)
    reg["regime_state"] = np.select(
        [reg["regime_points"] < r.elevated_min_points, reg["regime_points"] < r.stressed_min_points],
        ["normal", "elevated"], default="stressed")
    return reg


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
PANEL_COLS = ["ticker", "date", "open", "high", "low", "close", "volume", "ret_1d", "close_sma20", "close_sma50",
              "close_sma200", "mom_12_1", "growth_20d", "dollar_vol_20", "vol_20d", "is_liquid", "is_crash"]


def build_all(uni: pd.DataFrame, px: pd.DataFrame, market: pd.DataFrame, macro_wide: pd.DataFrame,
              filings: pd.DataFrame, sue: pd.DataFrame, pageviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (events, daily_panel, market_ctx_with_regime)."""
    ctx = market_context(market, macro_wide)
    meta = uni.set_index("ticker")
    # cross-sectional prior strength (guide: top decile of 20d benchmark-relative return)
    close_w = px.pivot_table(index="date", columns="ticker", values="close").sort_index()
    dv_w = (px.assign(dv=px["close"] * px["volume"]).pivot_table(index="date", columns="ticker", values="dv")
            .sort_index().shift(1).rolling(20, min_periods=20).median())
    strength_w = rw.prior_strength(close_w, dv_w, ctx["spy_close"].reindex(close_w.index))
    del close_w, dv_w
    panels, events = [], []
    tickers = sorted(set(px["ticker"]) & set(meta.index))
    for i, (tkr, g) in enumerate(px[px["ticker"].isin(tickers)].groupby("ticker", sort=True), 1):
        if len(g) < 260:
            continue
        row = meta.loc[tkr]
        f = ticker_features(g, ctx, row.get("sector_etf"), strength_w[tkr] if tkr in strength_w else None)
        cik = row.get("cik")
        earn = earnings_days(filings, int(cik), f["date"].values) if pd.notna(cik) and not filings.empty else pd.DataFrame(columns=["date", "earn_accepted_et"])
        f = f.merge(earn, on="date", how="left")
        f["is_earnings"] = f["earn_accepted_et"].notna()
        panels.append(f[PANEL_COLS].astype({c: "float32" for c in PANEL_COLS[2:15]}))
        f["is_breakout"] = f["arm_a"] == 1
        ev = f[(f["is_earnings"] | f["is_crash"] | f["is_breakout"]) & f["is_liquid"]].copy()
        if not ev.empty:
            ev["cik"] = cik
            events.append(ev)
        if i % 200 == 0:
            log.info("features: %d/%d tickers", i, len(tickers))
    panel = pd.concat(panels, ignore_index=True)
    ev = pd.concat(events, ignore_index=True)

    # event family: what kind of information shock is this?
    ev["event_family"] = np.select(
        [ev["is_earnings"] & ev["is_crash"], ev["is_earnings"] & (ev["abn_ret_1d"] >= 0), ev["is_earnings"], ev["is_crash"]],
        ["earnings_crash", "earnings_up", "earnings_down", "shock_crash"], default="momentum_breakout")

    # earnings momentum: how did the previous earnings event go? (known long before t)
    earn = ev[ev["is_earnings"]].sort_values(["ticker", "date"])
    grp = earn.groupby("ticker")
    earn_hist = pd.DataFrame(index=earn.index)
    earn_hist["prev_ear"] = grp["abn_ret_1d"].shift(1)
    prev_pead = grp["growth_future_20d"].shift(1)
    gap_days = (earn["date"] - grp["date"].shift(1)).dt.days
    earn_hist["prev_pead_20d"] = prev_pead.where(gap_days > 35)   # only once fully realised
    earn_hist["beats_last4"] = grp["abn_ret_1d"].transform(lambda s: (s > 0).astype(float).shift(1).rolling(4, min_periods=1).sum())
    ev = ev.join(earn_hist)
    ev = ev.sort_values(["ticker", "date"])
    ev["days_since_prev_event"] = ev.groupby("ticker")["date"].diff().dt.days

    # universe meta
    ev = ev.merge(meta[["company", "sector", "sub_industry", "cap_bucket", "wikipedia_title"]].reset_index(), on="ticker", how="left")
    # EDGAR, attention, SUE
    if not filings.empty:
        ev = ev.join(edgar_features(ev[["ticker", "date", "cik", "prev_date", "next_date"]].dropna(subset=["cik"]).astype({"cik": "int64"}), filings))
    att = attention_features(pageviews)
    if not att.empty:
        ev = ev.merge(att, on=["wikipedia_title", "date"], how="left")
    ev = ev.join(sue_features(ev, sue))
    # market context + regime
    reg = regime_table(ctx, panel)
    ctx = ctx.join(reg)
    ctx_cols = ["spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_above_200", "spy_dd_252", "iwm_ret_20d",
                "small_vs_large_20d", "vix", "vix_chg_5d", "vix_term", "sector_corr_20d", "dgs10", "dgs2",
                "fedfunds", "credit_spread", "credit_chg_20d", "curve_10y2y", "breadth_above_200", "crash_breadth",
                "regime_points", "regime_state", "flag_trend", "flag_vix_term", "flag_corr", "flag_breadth", "flag_credit"]
    ev = ev.merge(ctx[[c for c in ctx_cols if c in ctx]].reset_index().rename(columns={"index": "date"}), on="date", how="left")
    ev = ev.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info("features: %d events, %d panel rows", len(ev), len(panel))
    return ev, panel, ctx
