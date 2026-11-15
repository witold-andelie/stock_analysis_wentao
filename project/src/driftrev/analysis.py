"""Deep-dive diagnostics: where do predictions work, where do they fail, and why."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import TARGET

DIMENSIONS = ["event_family", "edgar_category", "sector", "cap_bucket", "year", "month", "regime_state",
              "bin_vix", "bin_momentum", "bin_reaction", "bin_sue", "bin_attention", "prob_bucket"]


def with_dimensions(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["date"].dt.year.astype(str)
    d["month"] = d["date"].dt.month_name()
    for b in ("bin_vix", "bin_momentum", "bin_reaction", "bin_sue", "bin_attention"):
        cols = [c for c in d.columns if c.startswith(b + "_")]
        if b not in d and cols:  # rebuild the categorical from its dummies
            d[b] = d[cols].idxmax(axis=1).str.replace(b + "_", "", regex=False)
    d["prob_bucket"] = pd.cut(d[prob_col], [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]).astype(str)
    return d


def precision_breakdown(df: pd.DataFrame, pred_col: str, prob_col: str) -> pd.DataFrame:
    """For each dimension value: base rate, #selected, precision of the selected trades,
    and their average 10-day growth."""
    d = with_dimensions(df[df[TARGET].notna()], prob_col)
    rows = []
    for dim in DIMENSIONS:
        if dim not in d:
            continue
        for val, g in d.groupby(dim):
            sel = g[g[pred_col] == 1]
            rows.append({"dimension": dim, "value": str(val), "n": len(g), "base_rate": float(g[TARGET].mean()),
                         "n_selected": len(sel),
                         "precision": float(sel[TARGET].mean()) if len(sel) else np.nan,
                         "avg_growth": float(sel["growth_future_10d"].mean()) if len(sel) else np.nan,
                         "avg_growth_all": float(g["growth_future_10d"].mean())})
    out = pd.DataFrame(rows)
    out["lift"] = out["precision"] - out["base_rate"]
    return out


def worst_false_positives(df: pd.DataFrame, prob_col: str, n: int = 25) -> pd.DataFrame:
    """Most confident predictions that lost money: the starting point for debugging."""
    d = df[df[TARGET] == 0].sort_values(prob_col, ascending=False).head(n)
    cols = ["date", "ticker", "company", "event_family", "edgar_category", "edgar_items", prob_col,
            "abn_ret_1d", "growth_future_10d", "mae_10d", "regime_state", "sector"]
    return d[[c for c in cols if c in d]]


def calibration(df: pd.DataFrame, prob_col: str, bins: int = 10) -> pd.DataFrame:
    d = df[df[TARGET].notna()].copy()
    d["bucket"] = pd.qcut(d[prob_col], bins, duplicates="drop")
    return d.groupby("bucket", observed=True).agg(mean_prob=(prob_col, "mean"), hit_rate=(TARGET, "mean"),
                                                  avg_growth=("growth_future_10d", "mean"), n=(TARGET, "size")).reset_index()
