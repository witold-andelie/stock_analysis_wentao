"""Modeling: hand rules, tuned tree models, boosted models, an ensemble, and decision rules.

Protocol (no peeking at TEST):
  1. Hyper-parameters and the probability threshold are chosen on VALIDATION
     (models fit on TRAIN only).
  2. The chosen configuration is refit on TRAIN+VALIDATION and scored once on TEST.
  3. Live rows (target unknown yet) get probabilities from the same refit model.

Prediction columns follow the course naming: pred{N}_{family}_{detail} with 0/1 values,
plus proba_{model} columns with raw probabilities.
"""

from __future__ import annotations

import logging
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import artifacts, config
from .dataset import TARGET, fill_for_model

log = logging.getLogger(__name__)
RS = config.SETTINGS.random_state


# --------------------------------------------------------------------------------------
# Hand rules (hypotheses from HW1 and HW5, before any ML)
# --------------------------------------------------------------------------------------
def manual_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    fam, cat = df["event_family"], df["edgar_category"]
    mom_up = (df["close_sma50"] > 1) & (df["close_sma200"] > 1)
    return {
        # HW1: post-earnings drift in the direction of the surprise, only with trend support
        "pred0_manual_pead_momentum": ((fam == "earnings_up") & mom_up & (df["abn_ret_1d"] > 0.03)).astype(int),
        # HW5 Q3: panic crash with no damaging filing, oversold
        "pred1_manual_panic_no_damage": (fam.isin(["shock_crash", "earnings_crash"])
                                         & ~cat.isin(["bankruptcy_delisting", "restatement_auditor", "offering"])
                                         & (df["rsi_14"] < 35)).astype(int),
        # unconditional: buy every event (the benchmark every model must beat)
        "pred2_manual_all_events": pd.Series(1, index=df.index),
        # strategy guide, Section 9.1 arms (rule-based, no ML)
        "pred_armA_momentum_breakout": df["arm_a"].astype(int),
        "pred_armB_plus_consolidation": df["arm_b"].astype(int),
        "pred_armC_plus_window_release": df["arm_c"].astype(int),
    }


# --------------------------------------------------------------------------------------
# Candidate models and their small, time-respecting grids
# --------------------------------------------------------------------------------------
def candidates() -> dict[str, list[dict]]:
    return {
        "tree": [{"max_depth": d, "min_samples_leaf": 50} for d in (3, 4, 5, 6, 8, 10)],
        "rf": [{"n_estimators": 300, "max_depth": d, "min_samples_leaf": leaf, "max_features": "sqrt"}
               for d in (8, 12) for leaf in (20, 50)],
        "xgb": [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05, "subsample": 0.8,
                 "colsample_bytree": 0.6, "min_child_weight": 20} for n in (200, 400) for d in (3, 5)],
        "lgbm": [{"n_estimators": n, "num_leaves": nl, "learning_rate": 0.03, "subsample": 0.8, "subsample_freq": 1,
                  "colsample_bytree": 0.6, "min_child_samples": 100} for n in (300, 600) for nl in (15, 31)],
        "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    }


def build(kind: str, params: dict):
    if kind == "tree":
        return DecisionTreeClassifier(random_state=RS, **params)
    if kind == "rf":
        return RandomForestClassifier(random_state=RS, n_jobs=-1, **params)
    if kind == "xgb":
        return XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist", **params)
    if kind == "lgbm":
        return LGBMClassifier(random_state=RS, n_jobs=-1, verbose=-1, **params)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **params))
    raise ValueError(kind)


def precision_at_top(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:k]
    return float(y[top].mean())


def tune(ml: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Fit every grid point on TRAIN, score on VALIDATION. Returns the tuning table."""
    tr = ml[(ml["split"] == "train") & ml[TARGET].notna()]
    va = ml[(ml["split"] == "validation") & ml[TARGET].notna()]
    Xtr, ytr = fill_for_model(tr[features]), tr[TARGET].astype(int)
    Xva, yva = fill_for_model(va[features]), va[TARGET].astype(int).to_numpy()
    rows = []
    for kind, grid in candidates().items():
        for params in grid:
            t0 = time.time()
            m = build(kind, params).fit(Xtr, ytr)
            p = m.predict_proba(Xva)[:, 1]
            rows.append({"model": kind, "params": params, "val_auc": roc_auc_score(yva, p),
                         "val_prec_top10": precision_at_top(yva, p, 0.10),
                         "val_prec_top20": precision_at_top(yva, p, 0.20),
                         "val_base_rate": float(yva.mean()), "fit_seconds": round(time.time() - t0, 1)})
            log.info("tune %s %s auc=%.4f p@10=%.3f", kind, params, rows[-1]["val_auc"], rows[-1]["val_prec_top10"])
    return pd.DataFrame(rows)


def best_params(tuning: pd.DataFrame) -> dict[str, dict]:
    """Per model family: the grid point with the best validation precision in the top decile
    (what the strategy actually trades), AUC as tie-breaker."""
    t = tuning.sort_values(["val_prec_top10", "val_auc"], ascending=False)
    return {k: g.iloc[0]["params"] for k, g in t.groupby("model")}


def choose_threshold(y: np.ndarray, p: np.ndarray, min_share: float = 0.05) -> float:
    """Highest-precision threshold on VALIDATION that still trades >= min_share of events."""
    best, best_prec = 0.5, -1.0
    for thr in np.round(np.arange(0.40, 0.81, 0.01), 2):
        sel = p >= thr
        if sel.mean() < min_share:
            break
        prec = y[sel].mean()
        if prec > best_prec + 1e-4:
            best, best_prec = float(thr), prec
    return best


def fit_final(ml: pd.DataFrame, features: list[str], params: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    """Validation pass (for thresholds) + refit on TRAIN+VALIDATION, predict every row."""
    known = ml[TARGET].notna()
    tr = ml[(ml["split"] == "train") & known]
    va = ml[(ml["split"] == "validation") & known]
    trva = ml[ml["split"].isin(["train", "validation"]) & known]
    X_all = fill_for_model(ml[features])
    out = ml.copy()
    for name, s in manual_rules(ml).items():
        out[name] = s.values
    info = {"thresholds": {}, "val_auc": {}, "test_auc": {}, "params": params}
    val_probs, models = {}, {}
    for kind, prm in params.items():
        m_val = build(kind, prm).fit(fill_for_model(tr[features]), tr[TARGET].astype(int))
        val_probs[kind] = m_val.predict_proba(fill_for_model(va[features]))[:, 1]
        m = build(kind, prm).fit(fill_for_model(trva[features]), trva[TARGET].astype(int))
        models[kind] = m
        out[f"proba_{kind}"] = m.predict_proba(X_all)[:, 1]
    # ensemble of the three strongest learners (simple average of probabilities)
    ens = [k for k in ("rf", "xgb", "lgbm") if k in models]
    val_probs["ensemble"] = np.mean([val_probs[k] for k in ens], axis=0)
    out["proba_ensemble"] = out[[f"proba_{k}" for k in ens]].mean(axis=1)

    yva = va[TARGET].astype(int).to_numpy()
    te_mask = (out["split"] == "test") & known.values
    yte = out.loc[te_mask, TARGET].astype(int).to_numpy()
    n = 3
    for kind in list(params) + ["ensemble"]:
        thr = choose_threshold(yva, val_probs[kind])
        info["thresholds"][kind] = thr
        info["val_auc"][kind] = float(roc_auc_score(yva, val_probs[kind]))
        info["test_auc"][kind] = float(roc_auc_score(yte, out.loc[te_mask, f"proba_{kind}"]))
        out[f"pred{n}_{kind}_best_rule_{int(round(thr * 100))}"] = (out[f"proba_{kind}"] >= thr).astype(int)
        n += 1
    # custom rule: only the single most confident event per day (higher-probability events)
    p = out["proba_ensemble"]
    daily_max = p.groupby(out["date"]).transform("max")
    out[f"pred{n}_ensemble_top1_daily"] = ((p == daily_max) & (p >= info["thresholds"]["ensemble"])).astype(int)
    info["models"] = models
    info["ensemble_members"] = ens
    return out, info


def evaluate_preds(out: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    d = out[(out["split"] == split) & out[TARGET].notna()]
    rows = []
    for c in [c for c in out.columns if c.startswith("pred")]:
        sel = d[c] == 1
        rows.append({"prediction": c, "n_trades": int(sel.sum()), "share": float(sel.mean()),
                     "precision": float(d.loc[sel, TARGET].mean()) if sel.any() else np.nan,
                     "avg_growth_10d": float(d.loc[sel, "growth_future_10d"].mean()) if sel.any() else np.nan,
                     "base_rate": float(d[TARGET].mean())})
    return pd.DataFrame(rows).sort_values("precision", ascending=False)


def feature_importance(info: dict, features: list[str]) -> pd.DataFrame:
    rows = []
    for kind in ("tree", "rf", "xgb", "lgbm"):
        m = info["models"].get(kind)
        if m is None or not hasattr(m, "feature_importances_"):
            continue
        imp = np.asarray(m.feature_importances_, dtype=float)
        imp = imp / imp.sum() if imp.sum() else imp
        rows += [{"model": kind, "feature": f, "importance": float(v)} for f, v in zip(features, imp)]
    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])


def choose_production(info: dict) -> str:
    """Production model = best VALIDATION AUC among rf/xgb/lgbm/ensemble (never test)."""
    pool = {k: v for k, v in info["val_auc"].items() if k in ("rf", "xgb", "lgbm", "ensemble")}
    return max(pool, key=pool.get)


def save_bundle(info: dict, features: list[str], name: str) -> None:
    members = info["ensemble_members"] if name == "ensemble" else [name]
    bundle = {"name": name, "features": features, "threshold": info["thresholds"][name],
              "models": {k: info["models"][k] for k in members}, "params": info["params"],
              "val_auc": info["val_auc"].get(name), "test_auc": info["test_auc"].get(name)}
    config.ensure_dirs()
    joblib.dump(bundle, config.ARTIFACTS_DIR / artifacts.MODEL)


def load_bundle() -> dict:
    return joblib.load(config.ARTIFACTS_DIR / artifacts.MODEL)


def predict_with_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = fill_for_model(df.reindex(columns=bundle["features"], fill_value=0))
    return np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"].values()], axis=0)
