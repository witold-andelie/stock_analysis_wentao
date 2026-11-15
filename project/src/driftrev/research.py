"""Research layer: "which, not when" model comparison, book composition, holding-period study.

Everything that CHOOSES something is scored on VALIDATION (models fit on TRAIN only). TEST numbers
are computed in a second pass (refit on TRAIN+VALIDATION) and are only reported, never used to
select (see AGENTS.md invariant).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import roc_auc_score

from . import config, dataset, models, simulate
from .dataset import TARGET, fill_for_model

log = logging.getLogger(__name__)
RS = config.SETTINGS.random_state

FAMILY_GROUPS = models.FAMILY_GROUPS


def family_group(fam: pd.Series) -> pd.Series:
    inv = {f: g for g, fs in FAMILY_GROUPS.items() for f in fs}
    return fam.map(inv).fillna("other")


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------
def within_date_corr(dates: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    """Correlation of (score - date mean) with (label - date mean): the 'which' skill."""
    d = pd.DataFrame({"d": dates.to_numpy(), "y": y.astype(float), "p": p})
    g = d.groupby("d")
    yc, pc = d["y"] - g["y"].transform("mean"), d["p"] - g["p"].transform("mean")
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    return float((yc * pc).sum() / den) if den > 0 else float("nan")


def date_level_auc(dates: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    """AUC of the date-mean score alone: the 'when' skill (market timing share of the signal)."""
    d = pd.DataFrame({"d": dates.to_numpy(), "y": y, "p": p})
    pm = d.groupby("d")["p"].transform("mean")
    return float(roc_auc_score(y, pm)) if len(np.unique(y)) > 1 else float("nan")


def score_block(df: pd.DataFrame, p: np.ndarray, prefix: str) -> dict:
    y = df[TARGET].astype(int).to_numpy()
    return {f"{prefix}_auc": float(roc_auc_score(y, p)),
            f"{prefix}_prec_top10": models.precision_at_top(y, p, 0.10),
            f"{prefix}_prec_top20": models.precision_at_top(y, p, 0.20),
            f"{prefix}_within_date_corr": within_date_corr(df["date"], y, p),
            f"{prefix}_when_auc": date_level_auc(df["date"], y, p),
            f"{prefix}_base_rate": float(y.mean()), f"{prefix}_n": int(len(y))}


# --------------------------------------------------------------------------------------
# Fitting variants
# --------------------------------------------------------------------------------------
def _fit_rank(tr: pd.DataFrame, features: list[str]) -> LGBMRanker:
    """LightGBM lambdarank, one query (group) per event date: the objective only rewards putting
    the events that bounce above the others of the SAME date."""
    tr = tr.sort_values("date", kind="stable")
    sizes = tr.groupby("date", sort=True).size().to_numpy()
    m = LGBMRanker(objective="lambdarank", n_estimators=300, num_leaves=31, learning_rate=0.03, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.6, min_child_samples=100, random_state=RS, n_jobs=-1,
                   verbose=-1)
    return m.fit(fill_for_model(tr[features]), tr[TARGET].astype(int), group=sizes)


def fit_predict(tr: pd.DataFrame, pr: pd.DataFrame, features: list[str], kind: str, params: dict,
                layout: str = "pooled") -> np.ndarray:
    """Fit on `tr`, return scores for `pr`. kind: rf/xgb/lgbm/logreg/tree or 'rank'. layout: pooled/per_family."""
    def one(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
        if kind == "rank":
            return _fit_rank(a, features).predict(fill_for_model(b[features]))
        m = models.build(kind, params).fit(fill_for_model(a[features]), a[TARGET].astype(int))
        return m.predict_proba(fill_for_model(b[features]))[:, 1]

    if layout == "pooled":
        return one(tr, pr)
    gtr, gpr = family_group(tr["event_family"]), family_group(pr["event_family"])
    out = np.full(len(pr), np.nan)
    for g in FAMILY_GROUPS:
        sel = (gpr == g).to_numpy()
        if sel.any() and (gtr == g).sum() > 200:
            out[sel] = one(tr[gtr == g], pr[sel])
    return np.where(np.isnan(out), np.nanmean(out), out)


def variants() -> list[dict]:
    """The comparison grid: (a) current pooled, (b) pooled + cross-sectional (market kept / dropped),
    (c) per-family models, (d) lambdarank grouped by date."""
    v = [{"variant": "a_pooled", "set": "pooled", "layout": "pooled"},
         {"variant": "a2_pooled_month_wom", "set": "pooled_wom", "layout": "pooled"},
         {"variant": "b1_cs_keep_market", "set": "cs_keep_market", "layout": "pooled"},
         {"variant": "b2_cs_drop_market", "set": "cs_drop_market", "layout": "pooled"},
         {"variant": "c1_per_family_pooled_set", "set": "pooled", "layout": "per_family"},
         {"variant": "c2_per_family_cs_keep", "set": "cs_keep_market", "layout": "per_family"},
         {"variant": "c3_per_family_cs_drop", "set": "cs_drop_market", "layout": "per_family"},
         {"variant": "d1_lambdarank_cs_keep", "set": "cs_keep_market", "layout": "pooled", "rank": True},
         {"variant": "d2_lambdarank_cs_drop", "set": "cs_drop_market", "layout": "pooled", "rank": True}]
    return v


def model_comparison(ml: pd.DataFrame, params: dict[str, dict], learners: tuple[str, ...] = ("rf", "lgbm", "logreg"),
                     skip_test: bool = False) -> pd.DataFrame:
    """Full comparison table. Every row has val_* (train-only fit, used to CHOOSE) and test_*
    (train+validation refit, reported only)."""
    fs = dataset.feature_sets(ml)
    known = ml[TARGET].notna()
    tr = ml[(ml["split"] == "train") & known]
    va = ml[(ml["split"] == "validation") & known]
    trva = ml[ml["split"].isin(["train", "validation"]) & known]
    te = ml[(ml["split"] == "test") & known]
    rows = []
    for v in variants():
        feats = fs[v["set"]]
        for kind in (("rank",) if v.get("rank") else learners):
            t0 = time.time()
            prm = params.get(kind, {})
            row = {"variant": v["variant"], "learner": kind, "feature_set": v["set"], "layout": v["layout"],
                   "n_features": len(feats)}
            row.update(score_block(va, fit_predict(tr, va, feats, kind, prm, v["layout"]), "val"))
            if not skip_test:
                row.update(score_block(te, fit_predict(trva, te, feats, kind, prm, v["layout"]), "test"))
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            log.info("compare %s/%s val auc=%.4f p@10=%.3f wd=%.4f", v["variant"], kind, row["val_auc"],
                     row["val_prec_top10"], row["val_within_date_corr"])
    return pd.DataFrame(rows)


def pick_variant(cmp: pd.DataFrame, eligible: tuple[str, ...] = ("rf", "xgb", "lgbm")) -> pd.Series:
    """Winner = best VALIDATION rank-average of precision@top10% and AUC among the learners that can be
    the production model (models.choose_production: rf/xgb/lgbm/ensemble). Rankers have no probability
    and logreg is not a production candidate, so both are shown in the table but cannot win."""
    c = cmp[cmp["learner"].isin(eligible)].copy()
    c["score"] = c["val_prec_top10"].rank() + c["val_auc"].rank()
    return c.sort_values(["score", "val_prec_top10"], ascending=False).iloc[0]


# --------------------------------------------------------------------------------------
# HW2 Q3: holding period study
# --------------------------------------------------------------------------------------
def holding_period_study(ml: pd.DataFrame, selected: pd.Series | None = None) -> pd.DataFrame:
    """Median / mean growth and hit rate at 5 / 10 / 20 sessions per event family (and per
    sample). Uses only train+validation by default; test rows are labelled so they are never
    used to choose a horizon. `selected` (bool mask) adds a 'model_selected' sample."""
    rows = []
    samples = {"train": ml["split"] == "train", "validation": ml["split"] == "validation", "test": ml["split"] == "test"}
    if selected is not None:
        samples["validation_model_selected"] = (ml["split"] == "validation") & selected
        samples["test_model_selected"] = (ml["split"] == "test") & selected
    for sample, mask in samples.items():
        d = ml[mask]
        for fam, g in [("ALL", d)] + list(d.groupby("event_family")):
            for h in (5, 10, 20):
                x = g[f"growth_future_{h}d"].dropna() - 1
                if len(x) == 0:
                    continue
                rows.append({"sample": sample, "event_family": fam, "horizon": h, "n": len(x),
                             "median_growth": float(x.median()), "mean_growth": float(x.mean()),
                             "hit_rate": float((x > 0).mean()),
                             "mean_growth_per_day": float(x.mean() / h)})
    out = pd.DataFrame(rows)
    # best horizon per family by VALIDATION median growth (report only; the target stays 10 days)
    best = (out[out["sample"] == "validation"].sort_values("median_growth", ascending=False)
            .groupby("event_family").head(1)[["event_family", "horizon"]].rename(columns={"horizon": "best_horizon_val_median"}))
    return out.merge(best, on="event_family", how="left")


# --------------------------------------------------------------------------------------
# Book composition and capacity: evidence on VALIDATION (test shown, labelled, never used to choose)
# --------------------------------------------------------------------------------------
def _sim_row(ev: pd.DataFrame, panel: pd.DataFrame, cfg: simulate.SimConfig, start: str) -> dict:
    eq, tr = simulate.exact_sim(ev, panel, cfg, start=start)
    if eq.empty:
        return {}
    m = simulate.perf_metrics(eq, cfg.name)
    return {"cagr": m.get("cagr"), "sharpe": m.get("sharpe"), "max_drawdown": m.get("max_drawdown"),
            "total_return": m.get("total_return"), "n_trades": len(tr),
            "win_rate": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan}


COMPOSITIONS = {
    "all_families": ((), ()),
    "no_breakout": (("momentum_breakout",), ()),
    "no_mgmt_change": ((), ("mgmt_change",)),
    "no_offering": ((), ("offering",)),
    "no_breakout_no_mgmt": (("momentum_breakout",), ("mgmt_change",)),
    "no_breakout_no_offering": (("momentum_breakout",), ("offering",)),
    "no_breakout_no_mgmt_no_offering": (("momentum_breakout",), ("mgmt_change", "offering")),
}


def composition_study(out: pd.DataFrame, panel: pd.DataFrame, prod: str, thr: float) -> pd.DataFrame:
    """For every candidate book: model_top5 exact simulation on VALIDATION (probabilities from the
    train-only fit, validation threshold) and, labelled, on TEST. Plus precision lift per family /
    EDGAR category. Column `decision_sample` = 'validation' marks the rows used to choose."""
    va_end = pd.Timestamp(config.SETTINGS.split.valid_end)
    va_start = (pd.Timestamp(config.SETTINGS.split.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    te_start = (va_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    va = out[out["split"] == "validation"]
    te = out[out["split"] == "test"]
    panel_va = panel[panel["date"] <= va_end + pd.Timedelta(days=20)]
    st = config.SETTINGS.strategy
    rows = []
    for name, (ex_f, ex_c) in COMPOSITIONS.items():
        for sample, ev, col, pnl, start in (("validation", va, f"proba_val_{prod}", panel_va, va_start),
                                            ("test", te, f"proba_{prod}", panel, te_start)):
            cfg = simulate.SimConfig(name=name, prob_col=col, min_prob=thr, max_positions=st.max_positions,
                                     hold_days=st.hold_days, exclude_families=ex_f, exclude_categories=ex_c)
            sel = ev[ev[col] >= thr]
            sel = sel[~sel["event_family"].isin(ex_f) & ~sel["edgar_category"].isin(ex_c)]
            kn = sel[sel[dataset.TARGET].notna()]
            row = {"kind": "composition", "name": name, "sample": sample, "decision_sample": sample == "validation",
                   "n_selected": len(kn), "precision": float(kn[dataset.TARGET].mean()) if len(kn) else np.nan,
                   "avg_growth_10d": float(kn["growth_future_10d"].mean() - 1) if len(kn) else np.nan}
            row.update(_sim_row(ev, pnl, cfg, start))
            rows.append(row)
    # lift per family and EDGAR category (validation chosen, test shown)
    for sample, ev, col in (("validation", va, f"proba_val_{prod}"), ("test", te, f"proba_{prod}")):
        kn = ev[ev[dataset.TARGET].notna()]
        for dim in ("event_family", "edgar_category", "cap_bucket"):
            for val, g in kn.groupby(dim):
                sel = g[g[col] >= thr]
                rows.append({"kind": f"lift_{dim}", "name": str(val), "sample": sample, "decision_sample": sample == "validation",
                             "n_selected": len(sel), "n": len(g), "base_rate": float(g[dataset.TARGET].mean()),
                             "precision": float(sel[dataset.TARGET].mean()) if len(sel) else np.nan,
                             "avg_growth_10d": float(sel["growth_future_10d"].mean() - 1) if len(sel) else np.nan})
    return pd.DataFrame(rows)


def suggest_composition(study: pd.DataFrame) -> str:
    """Candidate book with the best VALIDATION Sharpe (tie-break CAGR); test is ignored."""
    c = study[(study["kind"] == "composition") & (study["sample"] == "validation")].dropna(subset=["sharpe"])
    return str(c.sort_values(["sharpe", "cagr"], ascending=False).iloc[0]["name"]) if len(c) else "all_families"


def capacity_study(out: pd.DataFrame, panel: pd.DataFrame, prod: str, caps=(None, 0.5, 1.0, 2.0, 3.0, 5.0)) -> pd.DataFrame:
    """HW4: for each capacity (avg trades per active day) the VALIDATION-chosen threshold, its
    validation precision / trades per day, and the validation exact-sim result."""
    va = out[(out["split"] == "validation") & out[dataset.TARGET].notna()]
    y, p, d = va[dataset.TARGET].astype(int).to_numpy(), va[f"proba_val_{prod}"].to_numpy(), va["date"].to_numpy()
    va_end = pd.Timestamp(config.SETTINGS.split.valid_end)
    va_start = (pd.Timestamp(config.SETTINGS.split.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    panel_va = panel[panel["date"] <= va_end + pd.Timedelta(days=20)]
    st = config.SETTINGS.strategy
    rows = []
    for cap in caps:
        # min_share lowered to 1% so a binding capacity can actually move the threshold
        thr = models.choose_threshold(y, p, min_share=0.05 if cap is None else 0.01, dates=d, max_trades_per_day=cap)
        sel = p >= thr
        cfg = simulate.SimConfig(name="cap", prob_col=f"proba_val_{prod}", min_prob=thr, max_positions=st.max_positions,
                                 hold_days=st.hold_days)
        row = {"max_trades_per_day": cap if cap is not None else np.nan, "threshold": thr, "val_precision": float(y[sel].mean()) if sel.any() else np.nan,
               "val_avg_trades_per_day": models.avg_trades_per_day(d, sel), "val_share": float(sel.mean())}
        row.update(_sim_row(out[out["split"] == "validation"], panel_va, cfg, va_start))
        rows.append(row)
    return pd.DataFrame(rows)
