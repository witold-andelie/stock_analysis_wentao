"""Technical indicators in plain pandas/numpy (no TA-Lib C dependency, so Docker stays slim).

Every function takes one ticker's frame sorted by date and returns Series aligned to it.
Row t only uses rows <= t (no look-ahead); rolling windows require full windows.
Formulas follow the TA-Lib definitions used in the course's Module 2 notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing = EMA with alpha 1/n (used by RSI, ATR, ADX)."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, down = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n: int = 14) -> pd.Series:
    return wilder(true_range(h, l, c), n)


def adx(h, l, c, n: int = 14) -> pd.Series:
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = wilder(true_range(h, l, c), n)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=h.index), n) / tr
    minus_di = 100 * wilder(pd.Series(minus_dm, index=h.index), n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n)


def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    line = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def bollinger_pctb(close: pd.Series, n=20, k=2.0) -> pd.Series:
    m, sd = sma(close, n), close.rolling(n, min_periods=n).std(ddof=0)
    return (close - (m - k * sd)) / (2 * k * sd).replace(0, np.nan)


def stoch_k(h, l, c, n=14) -> pd.Series:
    hh, ll = h.rolling(n, min_periods=n).max(), l.rolling(n, min_periods=n).min()
    return 100 * (c - ll) / (hh - ll).replace(0, np.nan)


def cci(h, l, c, n=20) -> pd.Series:
    tp = (h + l + c) / 3
    arr = tp.to_numpy(dtype=float)
    mad = np.full(len(arr), np.nan)
    if len(arr) >= n:  # vectorised mean absolute deviation over sliding windows
        win = np.lib.stride_tricks.sliding_window_view(arr, n)
        mad[n - 1:] = np.abs(win - win.mean(axis=1, keepdims=True)).mean(axis=1)
    mad = pd.Series(mad, index=tp.index)
    return (tp - sma(tp, n)) / (0.015 * mad.replace(0, np.nan))


def mfi(h, l, c, v, n=14) -> pd.Series:
    tp = (h + l + c) / 3
    flow = tp * v
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def obv_slope(c, v, n=20) -> pd.Series:
    """20-day change of on-balance volume, scaled by 20-day average volume."""
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    return (obv - obv.shift(n)) / (sma(v, n) * n).replace(0, np.nan)


def streak(ret: pd.Series, sign: int) -> pd.Series:
    """Number of consecutive days up to and including t with sign(ret) == sign."""
    hit = (np.sign(ret) == sign).astype(int)
    groups = (hit == 0).cumsum()
    return hit.groupby(groups).cumsum()


def rolling_beta(r: pd.Series, m: pd.Series, n: int = 252, min_periods: int = 120) -> pd.Series:
    cov = r.rolling(n, min_periods=min_periods).cov(m)
    var = m.rolling(n, min_periods=min_periods).var()
    return cov / var.replace(0, np.nan)
