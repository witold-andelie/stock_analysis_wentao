"""Rolling-Window Pressure & Release (from the project's strategy guide, Variant B).

Guide thesis: after an unusually strong benchmark-relative advance, a stock may pause; as the
early gains age out of a rolling measurement window, the measured extreme falls mechanically
even if the price does not. A renewed advance is tradable only if genuine demand confirms it.

Scope: US equities only. US exchanges have no cumulative-deviation threshold rule (NYSE/Nasdaq
LULD is a 5-minute band), so "release" is used ONLY as a statistical feature; the guide's
exchange-rule variant (Variant A / Arm D) is out of scope for this project.

Definitions (guide Section 3), P = stock close, B = benchmark close (SPY):
    D(t, L)      = 100 * (P_t / P_{t-L} - B_t / B_{t-L})              percentage points
    M(t, W)      = max_{L=1..W} D(t, L)
    M_flat(t+j)  = max over the window at t+j with P and B frozen at their t values.
                   Lags L <= j then reference the frozen price (D = 0); lags L > j map to
                   D(t, L - j), so  M_flat(t+j, W) = max(0, M(t, W - j)).
    Release(j)   = M(t, W) - M_flat(t+j, W)  >= 0   (uses only data known at t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

WINDOWS = (10, 20)
RELEASE_STEPS = (1, 3, 5)
# research defaults from guide Section 4 (hypothetical starting points, not optimised)
MIN_MEDIAN_DOLLAR_VOL = 10e6
TOP_DECILE = 0.90
PAUSE_MIN, PAUSE_MAX = 3, 8
MAX_PULLBACK_ATR = 2.0
MIN_CLOSE_LOC = 0.75
MIN_VOLUME_MULT = 1.5


def deviation_matrix(close: pd.Series, bench: pd.Series, max_lag: int) -> np.ndarray:
    """Column L-1 holds D(t, L) for L = 1..max_lag."""
    c, b = close.to_numpy(float), bench.to_numpy(float)
    out = np.full((len(c), max_lag), np.nan)
    for lag in range(1, max_lag + 1):
        out[lag:, lag - 1] = 100 * (c[lag:] / c[:-lag] - b[lag:] / b[:-lag])
    return out


def window_features(close: pd.Series, bench: pd.Series) -> pd.DataFrame:
    wmax = max(WINDOWS)
    dev = deviation_matrix(close, bench, wmax)
    # running max over lags: cm[:, k] = M(t, k+1)
    cm = np.fmax.accumulate(np.where(np.isnan(dev), -np.inf, dev), axis=1)
    cm[np.isinf(cm)] = np.nan
    f = pd.DataFrame(index=close.index)
    f["rw_dev_10"] = dev[:, 9]
    f["rw_dev_20"] = dev[:, 19]
    for w in WINDOWS:
        m = cm[:, w - 1]
        f[f"rw_max_dev_{w}"] = m
        f[f"rw_best_lag_{w}"] = np.nanargmax(np.where(np.isnan(dev[:, :w]), -np.inf, dev[:, :w]), axis=1) + 1.0
        for j in RELEASE_STEPS:
            flat = np.maximum(0.0, cm[:, w - j - 1])
            f[f"rw_release_{w}_{j}"] = m - flat
        # proximity to the stock's own statistical extreme (no regulatory threshold in the US)
        f[f"rw_proximity_{w}"] = pd.Series(m, index=close.index) / pd.Series(m, index=close.index).rolling(252, min_periods=120).max()
    return f


def setup_features(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.DataFrame:
    """Consolidation + confirmation state at each close (all inputs <= t)."""
    f = pd.DataFrame(index=c.index)
    atr14 = ind.atr(h, l, c, 14)
    # sessions since the highest close of the prior 20 sessions (measured at t-1)
    prev = c.shift(1).to_numpy(float)
    since = np.full(len(prev), np.nan)
    if len(prev) >= 21:  # vectorised argmax over sliding 20-session windows ending at t-1
        win = np.lib.stride_tricks.sliding_window_view(prev[1:], 20)
        ok = np.isfinite(win).all(axis=1)
        since[20:] = np.where(ok, 19 - np.argmax(np.where(np.isfinite(win), win, -np.inf), axis=1), np.nan)
    f["rw_days_since_high"] = since
    impulse_high = c.shift(1).rolling(20, min_periods=20).max()
    pause_low = l.shift(1).rolling(PAUSE_MAX, min_periods=PAUSE_MIN).min()
    f["rw_pullback_atr"] = (impulse_high - pause_low) / atr14.shift(1)
    f["rw_range_compression"] = (h - l).shift(1).rolling(5, min_periods=5).mean() / (h - l).shift(6).rolling(15, min_periods=15).mean()
    f["rw_breakout_5d"] = (c > h.shift(1).rolling(5, min_periods=5).max()).astype(float)
    f["rw_volume_mult_med"] = v / v.shift(1).rolling(20, min_periods=20).median()
    f["rw_median_dollar_vol"] = (c * v).shift(1).rolling(20, min_periods=20).median()
    f["rw_stop_dist_atr"] = 2 * atr14 / c          # guide: initial stop ~2 ATR below entry
    return f


def prior_strength(close_wide: pd.DataFrame, dollar_med_wide: pd.DataFrame, spy: pd.Series,
                   lookback: int = 20, memory: int = 10) -> pd.DataFrame:
    """Cross-sectional percentile of D(t, 20) among liquid names, then the max over the
    previous `memory` sessions (t-memory .. t-1): "was this stock top-decile recently?"."""
    rel = 100 * (close_wide / close_wide.shift(lookback)).sub(spy / spy.shift(lookback), axis=0)
    rel = rel.where(dollar_med_wide >= MIN_MEDIAN_DOLLAR_VOL)
    rank = rel.rank(axis=1, pct=True)
    return rank.shift(1).rolling(memory, min_periods=1).max()


def arm_flags(f: pd.DataFrame) -> pd.DataFrame:
    """Guide Section 9.1 arms. A: base momentum breakout. B: + consolidation & confirmation.
    C: + positive flat-path window release. (D: regulatory events - not applicable in the US.)"""
    liquid = f["rw_median_dollar_vol"] >= MIN_MEDIAN_DOLLAR_VOL
    strong = (f["rw_prior_strength"] >= TOP_DECILE) & (f["growth_20d"] > 1)
    arm_a = liquid & strong & (f["rw_breakout_5d"] == 1)
    consolidation = f["rw_days_since_high"].between(PAUSE_MIN, PAUSE_MAX) & (f["rw_pullback_atr"] < MAX_PULLBACK_ATR)
    confirm = (f["close_loc"] >= MIN_CLOSE_LOC) & (f["rw_volume_mult_med"] >= MIN_VOLUME_MULT)
    arm_b = arm_a & consolidation & confirm
    arm_c = arm_b & (f["rw_release_20_3"] > 0)
    return pd.DataFrame({"arm_a": arm_a.astype(int), "arm_b": arm_b.astype(int), "arm_c": arm_c.astype(int)}, index=f.index)
