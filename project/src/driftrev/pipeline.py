"""End-to-end workflow: download -> transform -> (re)train -> simulate -> score -> publish.

Two regimes (course requirement):

* ``--mode update``  incremental download from the APIs (Yahoo, FRED, SEC EDGAR, Wikimedia,
                     Wikipedia constituents) into the DuckDB warehouse, then transform.
* ``--mode file``    no downloads and no transformations: read the ready-made parquet
                     snapshot (data/snapshot/, or fetched once from DRIFTREV_SNAPSHOT_URL).

Usage::

    python -m driftrev.pipeline --mode update            # daily cron run (reuses the model)
    python -m driftrev.pipeline --mode update --retrain  # refit models + all backtests
    python -m driftrev.pipeline --mode update --retrain --tune   # also re-run the grid search
    python -m driftrev.pipeline --mode file --retrain    # reproduce everything from the snapshot
    python -m driftrev.pipeline snapshot                 # export the snapshot for sharing
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import (analysis, artifacts, attention, config, dataset, edgar, features, housekeeping, macro, models, prices,
               report, signals, simulate, store, universe)

log = logging.getLogger("driftrev.pipeline")
S = config.SETTINGS
SNAP_FILES = ("events", "panel", "market", "universe", "ctx")


# --------------------------------------------------------------------------------------
# Step 1: ingest
# --------------------------------------------------------------------------------------
def ingest(refresh_universe: bool = False) -> dict:
    uni = universe.load_or_fetch(refresh=refresh_universe or not store.table_exists("universe"))
    report = {"universe": len(uni)}
    report["macro"] = macro.update_macro()
    report["market"] = prices.update_market()
    report["prices"] = prices.update_prices(uni["ticker"].tolist())
    report["edgar"] = edgar.update_edgar(uni["cik"].dropna().astype(int).tolist())
    report["pageviews"] = attention.update_pageviews(uni["wikipedia_title"].dropna().tolist())
    report["prices"]["failed"] = report["prices"]["failed"][:20]
    report["edgar"]["errors"] = dict(list(report["edgar"]["errors"].items())[:10])
    return report


# --------------------------------------------------------------------------------------
# Step 2: transform (or load the snapshot)
# --------------------------------------------------------------------------------------
def transform() -> dict[str, pd.DataFrame]:
    uni = universe.load_or_fetch()
    ev, panel, ctx = features.build_all(
        uni, prices.load_prices(), prices.load_market(), macro.load_macro_wide(),
        edgar.load_filings(), edgar.sue_table(edgar.load_eps_facts()), attention.load_pageviews())
    ctx = ctx.rename_axis("date").reset_index()
    store.replace_table("events", ev)
    store.replace_table("daily_panel", panel)
    store.replace_table("market_ctx", ctx)
    return {"events": ev, "panel": panel, "market": prices.load_market(), "universe": uni, "ctx": ctx}


def load_from_warehouse() -> dict[str, pd.DataFrame]:
    return {"events": store.read_table("events"), "panel": store.read_table("daily_panel"),
            "market": prices.load_market(), "universe": universe.load_or_fetch(), "ctx": store.read_table("market_ctx")}


def export_snapshot(data: dict[str, pd.DataFrame] | None = None) -> str:
    """Write data/snapshot/*.parquet and a zip to upload to Google Drive / S3."""
    data = data or load_from_warehouse()
    config.ensure_dirs()
    for name in SNAP_FILES:
        data[name].to_parquet(config.SNAPSHOT_DIR / f"{name}.parquet", index=False)
    zpath = config.DATA_DIR / "driftrev_snapshot.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for name in SNAP_FILES:
            z.write(config.SNAPSHOT_DIR / f"{name}.parquet", f"{name}.parquet")
    return str(zpath)


def load_snapshot() -> dict[str, pd.DataFrame]:
    missing = [n for n in SNAP_FILES if not (config.SNAPSHOT_DIR / f"{n}.parquet").exists()]
    if missing:
        if not config.SNAPSHOT_URL:
            raise FileNotFoundError(f"snapshot files missing {missing}; set DRIFTREV_SNAPSHOT_URL or run --mode update")
        log.info("downloading snapshot from %s", config.SNAPSHOT_URL)
        resp = store.get(store.session(), config.SNAPSHOT_URL, timeout=600)
        resp.raise_for_status()
        config.ensure_dirs()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(config.SNAPSHOT_DIR)
    return {n: pd.read_parquet(config.SNAPSHOT_DIR / f"{n}.parquet") for n in SNAP_FILES}


# --------------------------------------------------------------------------------------
# Step 3-4: train + backtest
# --------------------------------------------------------------------------------------
def train_and_backtest(ml: pd.DataFrame, feats: list[str], panel: pd.DataFrame, market: pd.DataFrame,
                       tune: bool = False) -> dict:
    tuning_path = config.ARTIFACTS_DIR / "tuning.parquet"
    if tune or not tuning_path.exists():
        tuning = models.tune(ml, feats)
        tuning.assign(params=tuning["params"].map(json.dumps)).to_parquet(tuning_path, index=False)
    tuning = pd.read_parquet(tuning_path)
    tuning["params"] = tuning["params"].map(json.loads)
    best = models.best_params(tuning)
    out, info = models.fit_final(ml, feats, best)
    prod = models.choose_production(info)
    thr = info["thresholds"][prod]
    prob_col = f"proba_{prod}"
    models.save_bundle(info, feats, prod)

    test = out[(out["split"] == "test")]
    known_test = test[test[dataset.TARGET].notna()]
    pred_cols = [c for c in out.columns if c.startswith("pred")]
    pred_eval = models.evaluate_preds(out, "test")
    vec = simulate.vector_sim(known_test, pred_cols)

    # exact simulations on TEST
    start = (pd.Timestamp(S.split.valid_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy = market[market["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    curves, trades, strat, cfgs = [], [], [], {}
    for cfg in simulate.standard_strategies(prob_col, thr):
        eq, tr = simulate.exact_sim(test, panel, cfg, start=start)
        if eq.empty:
            continue
        m = simulate.perf_metrics(eq, cfg.name)
        m.update({"n_trades": len(tr), "win_rate": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan,
                  "avg_trade_ret": float(tr["ret_net"].mean()) if len(tr) else np.nan,
                  "exposure_note": f"max {cfg.max_positions} positions, min_prob {cfg.min_prob:.2f}"
                                   + (" | guide R framework" if cfg.atr_stop_mult else ""),
                  "report_card": simulate.report_card(tr, eq, spy)})
        strat.append(m)
        cfgs[cfg.name] = cfg
        curves.append(eq.rename("equity").rename_axis("date").reset_index().assign(strategy=cfg.name))
        trades.append(tr)
    guide = guide_analysis(test, ml, panel, start, cfgs, strat, spy)
    end = panel["date"].max()
    bench_series = [simulate.buy_and_hold(market, "SPY", start, end, S.strategy.initial_capital),
                    simulate.buy_and_hold(market, "IWM", start, end, S.strategy.initial_capital),
                    simulate.momentum_baseline(panel, start, end, S.strategy.initial_capital)]
    benchmarks = []
    for b in bench_series:
        benchmarks.append(simulate.perf_metrics(b, b.name))
        curves.append(b.rename("equity").rename_axis("date").reset_index().assign(strategy=b.name))

    prod_pred = next(c for c in pred_cols if f"_{prod}_best_rule_" in c)
    breakdown = analysis.precision_breakdown(known_test, prod_pred, prob_col)
    artifacts.save_frame("predictions", known_test[["date", "ticker", "event_family", "edgar_category", "sector", "cap_bucket",
                                                    prob_col, dataset.TARGET, "growth_future_10d"]]
                         .rename(columns={prob_col: "prob", dataset.TARGET: "y_true", "growth_future_10d": "growth_future"}))
    artifacts.save_frame("trades", pd.concat(trades, ignore_index=True) if trades else pd.DataFrame())
    artifacts.save_frame("equity", pd.concat(curves, ignore_index=True))
    artifacts.save_frame("importance", models.feature_importance(info, feats))
    artifacts.save_frame("breakdown", breakdown)
    artifacts.save_frame("vector_sims.parquet", vec)
    artifacts.save_frame("pred_eval.parquet", pred_eval)
    artifacts.save_frame("false_positives.parquet", analysis.worst_false_positives(known_test, prob_col))
    artifacts.save_frame("calibration.parquet", analysis.calibration(known_test, prob_col).astype({"bucket": str}))
    split_sizes = ml["split"].value_counts().to_dict()
    return {
        "model": {"name": prod, "threshold": thr, "val_auc": info["val_auc"][prod], "test_auc": info["test_auc"][prod],
                  "target": f"{dataset.TARGET} (growth_future_10d >= {S.target.threshold})",
                  "test_base_rate": float(known_test[dataset.TARGET].mean()),
                  "test_precision_at_threshold": float(known_test.loc[known_test[prob_col] >= thr, dataset.TARGET].mean()),
                  "n_features": len(feats), "split_sizes": split_sizes},
        "models": [{"name": k, "val_auc": info["val_auc"][k], "test_auc": info["test_auc"][k], "threshold": info["thresholds"][k],
                    "params": info["params"].get(k)} for k in info["val_auc"]],
        "strategies": strat, "benchmarks": benchmarks, "guide": guide,
        "best_strategy": max(strat, key=lambda s: s.get("cagr") or -9)["name"] if strat else None,
    }


def guide_analysis(test: pd.DataFrame, ml: pd.DataFrame, panel: pd.DataFrame, start: str,
                   cfgs: dict, strat: list[dict], spy: pd.Series) -> dict:
    """Strategy guide Section 9: four-arm comparison, key causal test, parameter sensitivity
    and cost stress test. Writes guide_arms / guide_sensitivity / cost_stress parquet files."""
    rng = np.random.default_rng(42)

    def stats(df: pd.DataFrame, label: str, sample: str) -> dict:
        g = df["growth_future_10d"].dropna() - 1
        if len(g) == 0:
            return {"arm": label, "sample": sample, "n": 0}
        boots = rng.choice(g.to_numpy(), size=(1000, len(g))).mean(axis=1)
        return {"arm": label, "sample": sample, "n": len(g), "hit_rate_2pct": float((g >= 0.02).mean()),
                "mean_10d": float(g.mean()), "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
                "mean_10d_vs_spy": float((df["growth_future_10d_vs_spy"].dropna() - 1).mean())}

    rows = []
    bo = ml[ml["event_family"] == "momentum_breakout"]
    for sample, d in (("all_years", bo), ("test", bo[bo["split"] == "test"])):
        for arm in ("arm_a", "arm_b", "arm_c"):
            rows.append(stats(d[d[arm] == 1], arm, sample))
        rows.append(stats(d[(d["arm_b"] == 1) & (d["arm_c"] == 0)], "arm_b_without_release", sample))
    arms = pd.DataFrame(rows)
    artifacts.save_frame("guide_arms.parquet", arms)

    # parameter sensitivity: release definition / threshold (guide asks for it explicitly)
    sens = []
    b = bo[bo["arm_b"] == 1]
    for col in ("rw_release_20_1", "rw_release_20_3", "rw_release_20_5", "rw_release_10_3"):
        for thr_pp in (0.0, 2.0, 5.0, 10.0):
            for sample, d in (("all_years", b), ("test", b[b["split"] == "test"])):
                st = stats(d[d[col] > thr_pp], f"{col}>{thr_pp:g}", sample)
                st.update({"release_col": col, "threshold_pp": thr_pp})
                sens.append(st)
    artifacts.save_frame("guide_sensitivity.parquet", pd.DataFrame(sens))

    # cost stress test (1x / 2x / 3x) for the guide arms and the best model strategy
    best = max(strat, key=lambda x: x.get("cagr") or -9)["name"] if strat else None
    stress = []
    for name in [n for n in cfgs if n.startswith("guide_")] + ([best] if best else []):
        for mult in (1.0, 2.0, 3.0):
            eq, tr = simulate.exact_sim(test, panel, simulate.replace(cfgs[name], cost_mult=mult), start=start)
            if not eq.empty:
                m = simulate.perf_metrics(eq, name)
                stress.append({"strategy": name, "cost_mult": mult, "cagr": m.get("cagr"), "sharpe": m.get("sharpe"),
                               "max_drawdown": m.get("max_drawdown"), "n_trades": len(tr)})
    artifacts.save_frame("cost_stress.parquet", pd.DataFrame(stress))

    by = {s_["name"]: s_ for s_ in strat}
    b_, c_ = by.get("guide_armB_plus_consolidation", {}), by.get("guide_armC_plus_window_release", {})

    def mean_of(arm: str, sample: str) -> float:
        row = arms[(arms["arm"] == arm) & (arms["sample"] == sample)]
        return float(row["mean_10d"].iloc[0]) if len(row) and row["n"].iloc[0] else float("nan")

    evidence = {f"{arm}_{smp}_mean_10d": mean_of(arm, smp) for arm in ("arm_b", "arm_c") for smp in ("all_years", "test")}
    sens_c = pd.DataFrame(sens)
    sens_c = sens_c[(sens_c["sample"] == "all_years") & (sens_c["release_col"] == "rw_release_20_3")]
    evidence["sensitivity_sign_stable"] = bool((sens_c["mean_10d"] > 0).all() or (sens_c["mean_10d"] < 0).all())
    better_sharpe = (c_.get("sharpe") or -9) > (b_.get("sharpe") or -9)
    better_all = evidence["arm_c_all_years_mean_10d"] > evidence["arm_b_all_years_mean_10d"]
    better_test = evidence["arm_c_test_mean_10d"] > evidence["arm_b_test_mean_10d"]
    adds_value = better_sharpe and better_all and better_test and evidence["sensitivity_sign_stable"]
    reasons = [f"Sharpe C {'>' if better_sharpe else '<='} B", f"all-years mean C {'>' if better_all else '<='} B",
               f"test mean C {'>' if better_test else '<='} B",
               "sign stable across release thresholds" if evidence["sensitivity_sign_stable"] else "sign flips across release thresholds"]
    return {"key_causal_test": "Does Arm C (+ rolling-window release) beat Arm B (momentum + consolidation)?",
            "verdict": ("release adds value (keep, re-test out of sample)" if adds_value
                        else "release does NOT add value -> discard the rollover factor (guide Section 9 rule)")
                       + " | " + "; ".join(reasons),
            "evidence": evidence,
            "arm_d": "not applicable: US exchanges have no cumulative-deviation threshold (LULD is a 5-minute band)",
            "arms_sharpe": {k: by.get(k, {}).get("sharpe") for k in ("guide_armA_momentum_breakout",
                                                                       "guide_armB_plus_consolidation",
                                                                       "guide_armC_plus_window_release")}}


def publish_live(ml: pd.DataFrame, panel: pd.DataFrame, uni: pd.DataFrame, bundle: dict) -> dict:
    data_through = panel["date"].max()
    recent = ml[ml["date"] >= data_through - pd.Timedelta(days=90)].copy()
    prob = models.predict_with_bundle(bundle, recent)
    preds = artifacts.load_frame("predictions")
    lookup = signals.expected_return_lookup(preds.rename(columns={"prob": "p"}).assign(**{"growth_future_10d": preds.get("growth_future")}), "p") \
        if not preds.empty else (lambda p: None)
    sig = signals.build_signals(recent, prob, bundle["threshold"], bundle["name"], lookup)
    latest = sig[sig["date"] == data_through]
    artifacts.save_frame("events", sig)
    artifacts.save_frame("signals", latest)
    pr = panel[panel["date"] >= data_through - pd.Timedelta(days=380)][["ticker", "date", "open", "high", "low", "close", "volume"]]
    artifacts.save_frame("prices_recent", pr)
    artifacts.save_frame("universe", uni[[c for c in ["ticker", "company", "sector", "sub_industry", "cap_bucket", "cik", "wikipedia_title", "index_name"] if c in uni]])
    return {"data_through": data_through.date().isoformat(), "n_signals_today": len(latest),
            "n_actionable_today": int(latest["tier"].isin(["STRONG_BUY", "BUY"]).sum()),
            "n_events_90d": len(sig)}


def regime_block(ctx: pd.DataFrame) -> dict:
    """Fragility monitor on the latest market day (TradePulse idea): 5 flags -> points -> state."""
    last = ctx.dropna(subset=["regime_points"]).sort_values("date").iloc[-1]
    flags = {k: int(last[k]) for k in ("flag_trend", "flag_vix_term", "flag_corr", "flag_breadth", "flag_credit") if k in last}
    return {"state": str(last["regime_state"]), "points": int(last["regime_points"]),
            "as_of": pd.Timestamp(last["date"]).date().isoformat(), "flags": flags,
            "vix": float(last.get("vix", np.nan)), "vix_term": float(last.get("vix_term", np.nan)),
            "breadth_above_200": float(last.get("breadth_above_200", np.nan)),
            "explain": "1 point per trigger: SPY<200d MA, VIX/VIX3M>1, sector-correlation pctile>0.9, "
                       "breadth<40% above 200d MA, Baa-10y spread +25bp/20d. 0-1 normal, 2-3 elevated, 4+ stressed."}


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def run_pipeline(mode: str = "update", retrain: bool = False, tune: bool = False) -> dict:
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    steps: dict = {}
    try:
        if mode == "update":
            steps["ingest"] = ingest()
            data = transform()
        elif mode == "file":
            data = load_snapshot()
        elif mode == "warehouse":           # transform already done; reuse warehouse tables
            data = load_from_warehouse()
        else:
            raise ValueError(f"unknown mode {mode!r}")
        steps["transform"] = {"events": len(data["events"]), "panel_rows": len(data["panel"])}
        ml, feats = dataset.prepare(data["events"])
        need_train = retrain or not (config.ARTIFACTS_DIR / artifacts.MODEL).exists()
        metrics = artifacts.load_metrics()
        if need_train:
            metrics.update(train_and_backtest(ml, feats, data["panel"], data["market"], tune=tune))
            steps["train"] = {"model": metrics["model"]["name"], "test_auc": metrics["model"]["test_auc"]}
        bundle = models.load_bundle()
        steps["publish"] = publish_live(ml, data["panel"], data["universe"], bundle)
        metrics.update({"generated_at": datetime.now(timezone.utc).isoformat(),
                        "data_through": steps["publish"]["data_through"],
                        "universe_size": int(data["universe"]["ticker"].nunique()),
                        "n_events": int(len(data["events"])), "n_panel_rows": int(len(data["panel"])),
                        "regime": regime_block(data["ctx"]), "mode": mode})
        artifacts.save_metrics(metrics)
        # SD7: dated copy of the signal book, static results site, retention of expired files
        housekeeping.archive_signals(metrics["data_through"])
        steps["report"] = report.build()
        steps["housekeeping"] = housekeeping.run()
        status = "success"
        err = None
    except Exception as exc:  # recorded, then re-raised for callers / cron exit code
        status, err = "failed", f"{type(exc).__name__}: {exc}"
        log.exception("pipeline failed")
    entry = {"started": started, "finished": datetime.now(timezone.utc).isoformat(), "mode": mode, "retrain": retrain,
             "status": status, "error": err, "seconds": round(time.time() - t0, 1), "steps": steps}
    artifacts.append_run_log(entry)
    if err:
        raise RuntimeError(err)
    return json.loads(json.dumps(entry, default=str))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Drift-or-Reversal pipeline")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "snapshot"])
    ap.add_argument("--mode", default="update", choices=["update", "file", "warehouse"])
    ap.add_argument("--retrain", action="store_true", help="refit models and recompute all backtests")
    ap.add_argument("--tune", action="store_true", help="re-run the hyper-parameter grid search")
    a = ap.parse_args()
    if a.command == "snapshot":
        print(export_snapshot())
        return
    print(json.dumps(run_pipeline(a.mode, a.retrain, a.tune), indent=2))


if __name__ == "__main__":
    main()
