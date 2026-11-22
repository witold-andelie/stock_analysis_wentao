"""Turn model probabilities on the newest events into actionable, explainable signals.

Tiering borrows TradePulse's discipline: a tier is a research stance with an explicit cost
hurdle, the market regime downgrades conviction, and missing evidence means NO_CALL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from . import config, simulate

S = config.SETTINGS


def assign_tier(prob: float, thr: float, regime_state: str, expected_net: float | None,
                round_trip: float | None = None) -> str:
    """`round_trip` is the cap-aware round-trip cost of the stock's bucket (flat default when None)."""
    if prob is None or not np.isfinite(prob):
        return "NO_CALL"
    rt = S.costs.round_trip if round_trip is None else round_trip
    if prob >= thr + 0.10 and (expected_net is None or expected_net > rt):
        tier = "STRONG_BUY"
    elif prob >= thr and (expected_net is None or expected_net > 0):
        tier = "BUY"
    elif prob >= thr - 0.05:
        tier = "WATCH"
    elif prob <= 0.30:
        tier = "AVOID"
    else:
        tier = "SKIP"
    downgrade = {"stressed": {"STRONG_BUY": "BUY", "BUY": "WATCH"}, "elevated": {"STRONG_BUY": "BUY"}}
    return downgrade.get(regime_state, {}).get(tier, tier)


def expected_return_lookup(test_df: pd.DataFrame, prob_col: str):
    """Map a probability to the average realised net 10-day return of TEST events with a
    similar probability (deciles). Gives the agent a cost-aware number, not just a score."""
    d = test_df.dropna(subset=["growth_future_10d", prob_col])
    if len(d) < 200:
        return lambda p, c=None: None
    edges = np.unique(np.quantile(d[prob_col], np.linspace(0, 1, 11)))
    idx = np.clip(np.searchsorted(edges, d[prob_col], side="right") - 1, 0, len(edges) - 2)
    gross = d.assign(b=idx).groupby("b")["growth_future_10d"].mean() - 1      # gross 10-day edge per decile

    def lookup(p: float, cap_bucket: str | None = None):
        """Net expected return = decile gross edge - round-trip cost of the stock's cap bucket."""
        b = int(np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2))
        return float(gross.get(b, np.nan)) - simulate.round_trip_cost(cap_bucket)
    return lookup


def build_signals(ev: pd.DataFrame, prob: np.ndarray, thr: float, model_name: str, lookup) -> pd.DataFrame:
    """ev: event rows (latest dates) with ticker/date/close/etc. Returns the signals frame in
    the artifact contract."""
    st = S.strategy
    rs = config.research_settings()
    d = ev.copy()
    d["prob"] = prob
    caps = d["cap_bucket"] if "cap_bucket" in d else pd.Series(None, index=d.index)
    d["expected_net_return"] = [lookup(p, c) for p, c in zip(d["prob"], caps)]
    d["round_trip_cost"] = [simulate.round_trip_cost(c) for c in caps]
    d["tier"] = [assign_tier(p, thr, r, e, rt) for p, r, e, rt in
                 zip(d["prob"], d["regime_state"].fillna("normal"), d["expected_net_return"], d["round_trip_cost"])]
    # book composition (decided on VALIDATION): excluded families / EDGAR categories are never model BUYs
    ex = d["event_family"].isin(rs.book_exclude_families)
    if "edgar_category" in d:
        ex |= d["edgar_category"].isin(rs.book_exclude_categories)
    d["tier"] = np.where(ex & d["tier"].isin(["STRONG_BUY", "BUY", "WATCH"]), "SKIP", d["tier"])
    d["suggested_weight"] = np.where(d["tier"].isin(["STRONG_BUY", "BUY"]),
                                     np.clip((1 / st.max_positions) * (0.5 + (d["prob"] - thr) / max(1e-6, 1 - thr) * 2), 0.1, 0.3), 0.0)
    d["entry_rule"] = "market-on-open next session"
    d["entry_ref_price"] = d["close"]
    # breakouts use the strategy guide's template: stop 2 x ATR, target 3R, 8-session time stop
    bo = d["event_family"].eq("momentum_breakout")
    atr_stop = (2 * d.get("atr_pct", pd.Series(0.04, index=d.index)).fillna(0.04)).clip(0.02, 0.25)
    stop = np.where(bo, atr_stop, st.stop_loss)
    target = np.where(bo, 3 * atr_stop, st.take_profit)
    d["stop_loss_price"] = (d["close"] * (1 - stop)).round(2)
    d["take_profit_price"] = (d["close"] * (1 + target)).round(2)
    d["exit_by_date"] = [dt + BDay(9 if b else st.hold_days + 1) for dt, b in zip(d["date"], bo)]
    d["model_name"] = model_name
    d["threshold"] = thr
    keep = ["date", "ticker", "company", "sector", "cap_bucket", "event_family", "ret_1d", "abn_ret_1d", "prob",
            "tier", "expected_net_return", "suggested_weight", "entry_rule", "entry_ref_price", "stop_loss_price",
            "take_profit_price", "exit_by_date", "round_trip_cost", "edgar_items", "edgar_category", "sue", "regime_state", "model_name",
            "threshold", "growth_future_10d", "rsi_14", "dist_52w_high", "momentum_blend", "atr_pct",
            "rw_prior_strength", "rw_max_dev_20", "rw_release_20_3", "arm_b", "arm_c"]
    order = {"STRONG_BUY": 0, "BUY": 1, "WATCH": 2, "SKIP": 3, "AVOID": 4, "NO_CALL": 5}
    d = d[[c for c in keep if c in d]]
    return d.assign(_o=d["tier"].map(order)).sort_values(["date", "_o", "prob"], ascending=[False, True, False]).drop(columns="_o")
