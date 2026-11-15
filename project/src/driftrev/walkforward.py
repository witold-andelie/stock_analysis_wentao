"""R2 Walk-forward retraining: the honest, stitched out-of-sample track record.

For every year Y (first_year .. last year in the data):

  1. threshold year:  fit on events whose label had closed before year Y-1 started, score year Y-1,
                      choose the probability threshold on that validation year (same rule as models.choose_threshold)
  2. prediction year: refit on events whose label had closed before year Y started (expanding window),
                      predict year Y. Nothing from year Y (or its last-days labels) is used to fit or choose.

"Label closed" = `date + purge_days < period start`: a 10-session target looks ~14 calendar days
ahead, so the last rows before a boundary are purged (no label leakage across the boundary).
Stitching the yearly predictions gives one out-of-sample prediction per event, which drives an
exact simulation (cap-aware costs) and is compared with SPY / IWM / the HW1 momentum rule over the same dates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, models, research, simulate
from .dataset import TARGET

log = logging.getLogger(__name__)
S = config.SETTINGS


def year_splits(dates: pd.Series, first_year: int | None = None, purge_days: int | None = None) -> list[dict]:
    """Boundaries only (no data): one dict per prediction year."""
    rs = S.research
    first_year = first_year or rs.wf_first_year
    purge = pd.Timedelta(days=rs.wf_purge_days if purge_days is None else purge_days)
    last = int(pd.Timestamp(dates.max()).year)
    out = []
    for y in range(first_year, last + 1):
        ys, vs = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y - 1}-01-01")
        out.append({"year": y, "pred_start": ys, "pred_end": pd.Timestamp(f"{y}-12-31"),
                    "val_start": vs, "val_end": pd.Timestamp(f"{y - 1}-12-31"),
                    "val_train_cutoff": vs - purge,         # rows with date < cutoff may train the threshold model
                    "pred_train_cutoff": ys - purge})       # rows with date < cutoff may train the year-Y model
    return out


def run(ml: pd.DataFrame, features: list[str], kind: str, params: dict, layout: str = "pooled",
        first_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (stitched OOS predictions of every event, per-year table).
    New columns: proba_wf, wf_thr, wf_score (= proba - thr + 0.5, so one fixed 0.5 cut reproduces
    the per-year thresholds in the exact simulator), wf_year."""
    rs = S.research
    known = ml[TARGET].notna()
    parts, table = [], []
    for sp in year_splits(ml["date"], first_year):
        tr_v = ml[(ml["date"] < sp["val_train_cutoff"]) & known]
        va = ml[(ml["date"] >= sp["val_start"]) & (ml["date"] <= sp["val_end"]) & known]
        tr_p = ml[(ml["date"] < sp["pred_train_cutoff"]) & known]
        pr = ml[(ml["date"] >= sp["pred_start"]) & (ml["date"] <= sp["pred_end"])]
        if len(tr_v) < 5000 or len(va) < 500 or pr.empty:
            continue
        pv = research.fit_predict(tr_v, va, features, kind, params, layout)
        thr = models.choose_threshold(va[TARGET].astype(int).to_numpy(), pv, dates=va["date"].to_numpy(),
                                      max_trades_per_day=rs.max_trades_per_day)
        pp = research.fit_predict(tr_p, pr, features, kind, params, layout)
        part = pr.assign(proba_wf=pp, wf_thr=thr, wf_year=sp["year"])
        part["wf_score"] = part["proba_wf"] - thr + 0.5
        parts.append(part)
        kn = part[part[TARGET].notna()]
        sel = kn[kn["proba_wf"] >= thr]
        y = kn[TARGET].astype(int).to_numpy()
        table.append({"year": sp["year"], "threshold": thr, "n_train": len(tr_p), "n_events": len(kn),
                      "n_selected": len(sel), "base_rate": float(y.mean()) if len(y) else np.nan,
                      "precision_selected": float(sel[TARGET].mean()) if len(sel) else np.nan,
                      "auc": float(roc_auc_score(y, kn["proba_wf"])) if len(np.unique(y)) > 1 else np.nan,
                      "within_date_corr": research.within_date_corr(kn["date"], y, kn["proba_wf"].to_numpy()) if len(y) else np.nan,
                      "val_year_auc": float(roc_auc_score(va[TARGET].astype(int), pv))})
        log.info("walk-forward %s thr=%.2f n_sel=%d prec=%.3f auc=%.3f", sp["year"], thr, len(sel),
                 table[-1]["precision_selected"], table[-1]["auc"])
    stitched = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return stitched, pd.DataFrame(table)


def simulate_stitched(stitched: pd.DataFrame, panel: pd.DataFrame, market: pd.DataFrame,
                      start: str | None = None) -> dict:
    """Exact simulation on the stitched OOS predictions + same-window benchmarks."""
    start = start or str(stitched["date"].min().date())
    spy = market[market["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    ex_f, ex_c = S.research.book_exclude_families, S.research.book_exclude_categories
    base = simulate.SimConfig(name="", prob_col="wf_score", min_prob=0.5, max_positions=S.strategy.max_positions,
                              hold_days=S.strategy.hold_days)
    cfgs = [simulate.replace(base, name="wf_model_top5", exclude_families=ex_f, exclude_categories=ex_c),
            simulate.replace(base, name="wf_model_top5_flat_cost", exclude_families=ex_f, exclude_categories=ex_c,
                             cost_model="flat"),
            simulate.replace(base, name="wf_model_crash_only", families=("earnings_crash", "shock_crash")),
            simulate.replace(base, name="wf_all_events_unconditional", min_prob=-1e9)]
    curves, strat = [], []
    for cfg in cfgs:
        eq, tr = simulate.exact_sim(stitched, panel, cfg, start=start)
        if eq.empty:
            continue
        m = simulate.perf_metrics(eq, cfg.name)
        m.update({"n_trades": len(tr), "win_rate": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan,
                  "avg_trade_ret": float(tr["ret_net"].mean()) if len(tr) else np.nan,
                  "report_card": simulate.report_card(tr, eq, spy)})
        strat.append(m)
        curves.append(eq.rename("equity").rename_axis("date").reset_index().assign(strategy=cfg.name))
    end = panel["date"].max()
    cap = S.strategy.initial_capital
    bench = [simulate.buy_and_hold(market, "SPY", start, end, cap), simulate.buy_and_hold(market, "IWM", start, end, cap),
             simulate.momentum_baseline(panel, start, end, cap)]
    benchmarks = []
    for b in bench:
        benchmarks.append(simulate.perf_metrics(b, b.name))
        curves.append(b.rename("equity").rename_axis("date").reset_index().assign(strategy=b.name))
    return {"strategies": strat, "benchmarks": benchmarks,
            "equity": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()}
