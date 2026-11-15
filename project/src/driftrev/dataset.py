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

import numpy as np
import pandas as pd

from . import config

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
    "fedfunds", "credit_spread", "credit_chg_20d", "curve_10y2y", "breadth_above_200",
    "crash_breadth", "regime_points",
]

CATEGORICAL = [
    "event_family", "edgar_category", "sector", "cap_bucket", "month", "weekday", "regime_state",
    # engineered bins (manual transformations -> dummies)
    "bin_reaction", "bin_rsi", "bin_sue", "bin_52w", "bin_vix", "bin_attention", "bin_momentum",
]

TO_DROP = ["ticker", "date", "company", "sub_industry", "wikipedia_title", "cik", "prev_date", "next_date",
           "earn_accepted_et", "edgar_items", "open", "high", "low", "close", "volume", "dollar_vol_20",
           "is_liquid", "is_crash", "is_earnings", "is_breakout", "sue_is_current", "rw_median_dollar_vol"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Binned / manual features (each bin set counts as one engineered feature)."""
    d = df.copy()
    d["month"] = d["date"].dt.month_name()
    d["weekday"] = d["date"].dt.day_name()
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
    s = S.split
    return pd.Series(np.select([dates <= s.train_end, dates <= s.valid_end], ["train", "validation"], "test"), index=dates.index)


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
    features = num + dummy_cols
    return ml, features


def fill_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """Course convention: inf -> NaN -> 0 (trees are fine with the sentinel; the
    `*_known` information is carried by the dummies / counts)."""
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
