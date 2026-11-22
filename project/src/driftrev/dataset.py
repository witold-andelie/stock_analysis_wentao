"""From the event table to an ML-ready frame: feature sets, engineered bins, dummies, split.

Course convention (Module 3/4): variables are grouped into explicit sets so every later
step (EDA, modeling, simulation, API) refers to the same names:

    TO_PREDICT  targets (future, never used as inputs)
    NUMERIC     numeric inputs
    CATEGORICAL raw categoricals that become DUMMIES
    DUMMIES     one-hot columns generated from CATEGORICAL (+ engineered bins)
    TO_DROP     identifiers / helper columns kept for analysis only
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
S = config.SETTINGS
TARGET = "is_positive_growth_10d"          # growth_future_10d >= 1.02
TARGET_GROWTH = f"growth_future_{S.target.horizon}d"

TO_PREDICT = [
    "growth_future_5d", "growth_future_10d", "growth_future_20d",
    "growth_future_5d_vs_spy", "growth_future_10d_vs_spy", "growth_future_20d_vs_spy",
    "mae_10d", "mfe_10d", TARGET, "is_positive_growth_10d_vs_spy",
]

NUMERIC = [
    # event-day reaction
    "ret_1d", "abn_ret_1d", "ret_vs_sector", "gap_open", "range_pct", "close_loc", "ret_open_close",
    "volume_mult", "volume_z",
    # momentum (HW1 idea: short-horizon momentum + price vs 20/50/200-day MAs)
    "growth_3d", "growth_5d", "growth_10d", "growth_20d", "growth_60d", "growth_120d", "growth_252d",
    "mom_12_1", "close_sma20", "close_sma50", "close_sma200", "sma50_sma200", "rs_vs_spy_60d",
    "dist_52w_high", "dist_52w_low", "down_streak_prior", "up_streak_prior",
    # technicals (TA-Lib equivalents)
    "rsi_14", "macd_hist_pct", "bb_pctb", "stoch_k", "cci_20", "mfi_14", "adx_14", "obv_slope_20",
    # risk / liquidity
    "vol_20d", "vol_60d", "vol_ratio_20_60", "atr_pct", "beta_252", "idio_vol_60", "log_dollar_vol",
    # event history (earnings momentum / PEAD of previous quarter)
    "n_crash_prior_252", "days_since_prev_event", "prev_ear", "prev_pead_20d", "beats_last4",
    # SEC EDGAR (new source)
    "sue", "eps_yoy_pos", "sue_age_days", "edgar_n_window", "n_offering_365d", "n_8k_90d",
    "n_form4_30d", "n_13d_180d", "n_late_filing_365d",
    "item_1_01_agreement", "item_1_02_termination", "item_1_03_bankruptcy", "item_2_01_acquisition",
    "item_2_02_results", "item_2_05_restructuring", "item_2_06_impairment", "item_3_01_delisting",
    "item_3_02_unreg_equity", "item_4_01_auditor", "item_4_02_nonreliance", "item_5_02_officer",
    "item_7_01_regfd", "item_8_01_other", "form_offering", "form_nt", "form_periodic",
    # Wikipedia attention (new source)
    "att_ratio_7_90", "att_spike_1d", "att_level",
    # rolling-window pressure & release (strategy guide, Variant B)
    "rw_dev_10", "rw_dev_20", "rw_max_dev_10", "rw_max_dev_20", "rw_best_lag_10", "rw_best_lag_20",
    "rw_release_10_1", "rw_release_10_3", "rw_release_10_5", "rw_release_20_1", "rw_release_20_3", "rw_release_20_5",
    "rw_proximity_10", "rw_proximity_20", "rw_days_since_high", "rw_pullback_atr", "rw_range_compression",
    "rw_breakout_5d", "rw_volume_mult_med", "rw_stop_dist_atr", "rw_prior_strength", "arm_a", "arm_b", "arm_c",
    # market & macro context
    "spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_above_200", "spy_dd_252", "iwm_ret_20d",
    "small_vs_large_20d", "vix", "vix_chg_5d", "vix_term", "sector_corr_20d", "dgs10", "dgs2",
    "dgs5", "fedfunds", "credit_spread", "credit_chg_20d", "curve_10y2y", "breadth_above_200",
    "crash_breadth", "regime_points",
    # HW1 Q4: analyst-consensus EPS surprise (yfinance, matched point-in-time by consensus.py)
    "eps_surprise_pct", "eps_beat", "eps_est_available", "eps_release_age_sessions",
    # HW1 Q3: market correction state (SPY >= 5% below its running high)
    "in_correction", "correction_depth", "days_in_correction",
    # R4 filing text (SD2.1): flags + Loughran-McDonald tone of the event window's filings.
    # Coverage: crash families all years; earnings 2022+ only (ft_text_available tells the model)
    "ft_n_filings_window", "ft_n_text", "ft_text_available", "ft_guidance_cut", "ft_guidance_raise",
    "ft_guidance_reaffirm", "ft_going_concern", "ft_offering_priced", "ft_restatement", "ft_material_weakness",
    "ft_investigation", "ft_clinical_failure", "ft_exec_departure", "ft_covenant", "ft_impairment", "ft_layoffs",
    "ft_bankruptcy", "ft_beat", "ft_miss", "ft_lm_negative_pct", "ft_lm_positive_pct", "ft_lm_uncertainty_pct",
    "ft_lm_net_tone", "ft_offering_discount",
    # NOTE: in_index / is_current_member are bookkeeping for the point-in-time universe, never features
]

# --- research: "which, not when" ---------------------------------------------------------
# Market-level inputs are identical for every event on a date, so a pooled model can only use
# them to time the market. Dropping them (plus calendar dummies) leaves stock-level information.
MARKET_NUMERIC = [
    "spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_above_200", "spy_dd_252", "iwm_ret_20d",
    "small_vs_large_20d", "vix", "vix_chg_5d", "vix_term", "sector_corr_20d", "dgs10", "dgs2", "dgs5",
    "fedfunds", "credit_spread", "credit_chg_20d", "curve_10y2y", "breadth_above_200", "crash_breadth",
    "regime_points", "in_correction", "correction_depth", "days_in_correction",
]
MARKET_DUMMY_PREFIXES = ("month_", "weekday_", "regime_state_", "bin_vix_")   # month_wom_ starts with month_
# stock-level numerics that get a per-date percentile rank (cs_rank_*) and a date-demeaned copy (cs_dm_*)
CS_COLS = [
    "ret_1d", "abn_ret_1d", "ret_vs_sector", "gap_open", "range_pct", "close_loc", "volume_mult", "volume_z",
    "growth_5d", "growth_20d", "growth_60d", "mom_12_1", "rs_vs_spy_60d", "dist_52w_high", "dist_52w_low",
    "rsi_14", "bb_pctb", "vol_20d", "atr_pct", "beta_252", "idio_vol_60", "log_dollar_vol", "sue",
    "att_ratio_7_90", "rw_dev_20", "rw_prior_strength", "momentum_blend", "prev_ear", "prev_pead_20d",
    "close_sma50", "close_sma200", "eps_surprise_pct",
]
FEATURE_SETS = ("pooled", "pooled_wom", "cs_keep_market", "cs_drop_market")

CATEGORICAL = [
    "event_family", "edgar_category", "edgar_category_text", "sector", "cap_bucket", "month", "weekday", "regime_state",
    # engineered bins (manual transformations -> dummies)
    "bin_reaction", "bin_rsi", "bin_sue", "bin_52w", "bin_vix", "bin_attention", "bin_momentum",
    "month_wom",                       # HW3: month x week-of-month dummies (calendar, market-level)
]

TO_DROP = ["ticker", "date", "company", "sub_industry", "wikipedia_title", "cik", "prev_date", "next_date",
           "earn_accepted_et", "edgar_items", "open", "high", "low", "close", "volume", "dollar_vol_20",
           "is_liquid", "is_crash", "is_earnings", "is_breakout", "sue_is_current", "rw_median_dollar_vol",
           "in_index", "is_current_member"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Binned / manual features (each bin set counts as one engineered feature)."""
    d = df.copy()
    missing = [c for c in NUMERIC if c not in d]
    if missing:
        d[missing] = np.nan
        log.info("Added %d missing numeric columns: %s", len(missing), ", ".join(missing))
    if "edgar_category_text" not in d:  # snapshots built before R4 have no text layer
        d["edgar_category_text"] = d.get("edgar_category", "none")
    d["edgar_category_text"] = d["edgar_category_text"].fillna(d.get("edgar_category", "none"))
    d["month"] = d["date"].dt.month_name()
    d["weekday"] = d["date"].dt.day_name()
    d["month_wom"] = d["date"].dt.month_name().str[:3] + "_w" + ((d["date"].dt.day - 1) // 7 + 1).astype(str)
    d["bin_reaction"] = pd.cut(d["abn_ret_1d"], [-np.inf, -0.20, -0.10, -0.03, 0.03, 0.10, np.inf],
                               labels=["crash_lt_-20", "drop_-20_-10", "down_-10_-3", "flat", "up_3_10", "jump_gt_10"]).astype(str)
    d["bin_rsi"] = pd.cut(d["rsi_14"], [-np.inf, 20, 30, 50, 70, np.inf],
                          labels=["rsi_lt20", "rsi_20_30", "rsi_30_50", "rsi_50_70", "rsi_gt70"]).astype(str)
    d["bin_sue"] = np.where(d["sue"].isna() | ~d["sue_is_current"].fillna(False).astype(bool), "sue_unknown",
                            pd.cut(d["sue"], [-np.inf, -1, 0, 1, np.inf], labels=["miss_big", "miss", "beat", "beat_big"]).astype(str))
    d["bin_52w"] = pd.cut(d["dist_52w_high"], [-np.inf, -0.5, -0.3, -0.15, -0.05, np.inf],
                          labels=["dd_gt50", "dd_30_50", "dd_15_30", "dd_5_15", "near_high"]).astype(str)
    d["bin_vix"] = pd.cut(d["vix"], [-np.inf, 15, 20, 30, np.inf], labels=["vix_lt15", "vix_15_20", "vix_20_30", "vix_gt30"]).astype(str)
    d["bin_attention"] = pd.cut(d["att_spike_1d"], [-np.inf, 0, 0.5, 1.5, np.inf],
                                labels=["att_quiet", "att_normal", "att_high", "att_extreme"]).astype(str)
    # HW1 blended momentum: price vs 20/50/200-day MAs + 12-1 momentum, binned into terciles
    blend = (d[["close_sma20", "close_sma50", "close_sma200"]].sub(1).mean(axis=1) + d["mom_12_1"].fillna(0)) / 2
    d["momentum_blend"] = blend
    d["bin_momentum"] = pd.cut(blend, [-np.inf, -0.05, 0.05, np.inf], labels=["mom_down", "mom_flat", "mom_up"]).astype(str)
    hz, thr = S.target.horizon, S.target.threshold
    d[TARGET] = (d[f"growth_future_{hz}d"] >= thr).astype(float).where(d[f"growth_future_{hz}d"].notna())
    d["is_positive_growth_10d_vs_spy"] = (d["growth_future_10d_vs_spy"] > 1).astype(float).where(d["growth_future_10d_vs_spy"].notna())
    return d


def numeric_features() -> list[str]:
    return NUMERIC + ["momentum_blend"]


def make_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df[CATEGORICAL].astype(str), prefix=CATEGORICAL, dtype="int8")
    return dummies, list(dummies.columns)


def temporal_split(dates: pd.Series) -> pd.Series:
    """train <= train_end < validation <= valid_end < test. With `Split.purge` (default) the rows whose label
    window (`purge_days` calendar days ~ 10 sessions) reaches past the end of their block are labelled
    "purged": their outcome is realised inside the NEXT block, so fitting on them would leak it."""
    s = S.split
    sp = pd.Series(np.select([dates <= s.train_end, dates <= s.valid_end], ["train", "validation"], "test"),
                   index=dates.index, dtype=object)
    if s.purge:
        gap = pd.Timedelta(days=s.purge_days)
        sp[(sp == "train") & (dates > pd.Timestamp(s.train_end) - gap)] = "purged"
        sp[(sp == "validation") & (dates > pd.Timestamp(s.valid_end) - gap)] = "purged"
    return sp


def prepare(events: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns the ML frame (identifiers + targets + features + split) and the feature list.
    Rows without a known target are kept (they are the *live* events to predict)."""
    d = add_engineered(events)
    dummies, dummy_cols = make_dummies(d)
    num = [c for c in numeric_features() if c in d]
    X = d[num].astype("float64").replace([np.inf, -np.inf], np.nan)
    keep = ["ticker", "company", "date", "sector", "cap_bucket", "event_family", "edgar_category", "edgar_items",
            "regime_state", "open", "close", "next_date", "sue_is_current", "atr_pct"] + [c for c in TO_PREDICT if c in d]
    ml = pd.concat([d[keep].reset_index(drop=True), X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    ml = ml.loc[:, ~ml.columns.duplicated()]
    ml["split"] = temporal_split(ml["date"])
    ml = add_cross_sectional(ml)
    return ml, feature_sets(ml)[config.research_settings().feature_set]


def fill_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """Course convention: inf -> NaN -> 0 (trees are fine with the sentinel; the
    `*_known` information is carried by the dummies / counts)."""
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_cross_sectional(ml: pd.DataFrame, cols: list[str] | None = None, min_events: int = 3) -> pd.DataFrame:
    """Per-date cross-sectional copies of stock-level numerics (point-in-time: only events of the
    SAME date, i.e. information known at that close, are compared with each other).

    cs_rank_<c>  percentile rank in (0, 1] among the events of the date (0.5 when fewer than
                 `min_events` events share the date, where a rank is meaningless)
    cs_dm_<c>    value minus the date mean (0 when the value is missing)
    Each event only sees its own date, so shuffling other dates cannot change a row's values."""
    cols = [c for c in (cols or CS_COLS) if c in ml.columns]
    out = ml.drop(columns=[c for c in ml.columns if c.startswith(("cs_rank_", "cs_dm_"))])
    if not cols:
        return out
    g = out.groupby("date")[cols]
    rank = g.rank(pct=True)
    n_day = out.groupby("date")["date"].transform("size")
    rank = rank.where(n_day >= min_events, 0.5).fillna(0.5).add_prefix("cs_rank_").astype("float32")
    dm = (out[cols] - g.transform("mean")).replace([np.inf, -np.inf], np.nan).fillna(0.0).add_prefix("cs_dm_").astype("float32")
    return pd.concat([out, rank, dm], axis=1)


def feature_sets(ml: pd.DataFrame) -> dict[str, list[str]]:
    """Named feature lists over one ML frame (the same columns, different subsets).

    pooled          the original set: stock-level + market-level numerics + dummies (no month_wom)
    pooled_wom      pooled + HW3 month_wom dummies
    cs_keep_market  pooled + cross-sectional ranks/demeaned copies
    cs_drop_market  stock-level + cross-sectional only (no market-level numerics, no calendar /
                    regime / VIX-bin dummies): the model cannot time the market"""
    non_target = [c for c in ml.columns if c not in TO_PREDICT]
    num = [c for c in numeric_features() if c in ml.columns]
    dummies = [c for c in non_target if c not in num and any(c.startswith(f"{k}_") for k in CATEGORICAL)]
    wom = [c for c in dummies if c.startswith("month_wom_")]
    base_dummies = [c for c in dummies if c not in wom]
    cs = [c for c in ml.columns if c.startswith(("cs_rank_", "cs_dm_"))]
    pooled = num + base_dummies
    stock_num = [c for c in num if c not in MARKET_NUMERIC]
    stock_dummies = [c for c in base_dummies if not c.startswith(MARKET_DUMMY_PREFIXES)]
    sets = {"pooled": pooled, "pooled_wom": pooled + wom, "cs_keep_market": pooled + cs,
            "cs_drop_market": stock_num + stock_dummies + cs}
    all_nan = [c for c in num + dummies + cs if ml[c].notna().sum() == 0]
    if all_nan:
        log.info("Dropped %d all-NaN feature columns: %s", len(all_nan), ", ".join(all_nan))
    return {name: [c for c in cols if c not in all_nan] for name, cols in sets.items()}
