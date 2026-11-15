"""Generate the project notebooks from code (so they stay in sync with the modules).

    PYTHONPATH=src .venv/bin/python scripts/build_notebooks.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys, warnings, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)
plt.rcParams.update({"figure.figsize": (11, 4.5), "axes.grid": True, "grid.alpha": 0.3})
from driftrev import config, store, artifacts, dataset, features, models, simulate, analysis, pipeline
print("data dir:", config.DATA_DIR)"""


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


# ------------------------------------------------------------------------------------
write("00_end_to_end_workflow.ipynb", [
    md("""# 00 · End-to-end workflow (one notebook that drives the .py modules)

Every step below is a function from `src/driftrev/`. The same sequence runs headless via
`python -m driftrev.pipeline --mode update` (cron / Docker scheduler).

| step | function | regime |
|---|---|---|
| 1 download (incremental) | `pipeline.ingest()` | `update` only |
| 2 transform → one table | `pipeline.transform()` / `pipeline.load_snapshot()` | `update` / `file` |
| 3 feature sets + split | `dataset.prepare()` | both |
| 4 train / tune / backtest | `pipeline.train_and_backtest()` | `--retrain` |
| 5 score newest events → signals | `pipeline.publish_live()` | both |
| 6 serve to agents / broker | `service.*`, `broker.preview_orders()` | both |
"""),
    code(SETUP),
    md("## 1 · Download (set `FETCH = True` to pull the latest data incrementally)"),
    code("""FETCH = False          # True: Yahoo + FRED + SEC EDGAR + Wikimedia, only the missing days
if FETCH:
    report = pipeline.ingest()
    print(json.dumps(report, indent=1, default=str)[:2000])
with store.connect() as con:
    sizes = {t: con.execute(f"select count(*) from {t}").fetchone()[0]
             for t in ["universe", "prices", "market", "macro", "filings", "eps_facts", "pageviews", "events", "daily_panel"]}
pd.Series(sizes, name="rows").to_frame()"""),
    md("## 2 · Transform (or load the snapshot in `file` regime)"),
    code("""TRANSFORM = False      # True: rebuild events + daily panel from the raw tables (~3 min)
data = pipeline.transform() if TRANSFORM else pipeline.load_from_warehouse()
events, panel = data["events"], data["panel"]
print(f"events: {len(events):,}   daily panel: {len(panel):,} rows   tickers: {panel.ticker.nunique():,}")
events[["date", "ticker", "event_family", "ret_1d", "edgar_category", "sue", "growth_future_10d"]].tail()"""),
    md("## 3 · Feature sets and temporal split"),
    code("""ml, feats = dataset.prepare(events)
print("TO_PREDICT:", dataset.TO_PREDICT)
print(f"NUMERIC: {len(dataset.numeric_features())}  DUMMIES from {dataset.CATEGORICAL}")
print(f"total model features: {len(feats)}")
ml.groupby("split").agg(rows=("ticker", "size"), start=("date", "min"), end=("date", "max"),
                        base_rate=(dataset.TARGET, "mean"))"""),
    md("## 4 · Train + backtest (`RETRAIN=True` refits everything; otherwise reuse the saved model)"),
    code("""RETRAIN = False
if RETRAIN:
    m = pipeline.train_and_backtest(ml, feats, panel, data["market"], tune=False)
metrics = artifacts.load_metrics()
bundle = models.load_bundle()
print("production model:", bundle["name"], "threshold:", bundle["threshold"],
      "val AUC %.3f  test AUC %.3f" % (bundle["val_auc"], bundle["test_auc"]))
pd.DataFrame(metrics["strategies"] + metrics["benchmarks"])[["name", "cagr", "sharpe", "max_drawdown", "n_trades"]]"""),
    md("## 5 · Score the newest events → today's new trades"),
    code("""live = pipeline.publish_live(ml, panel, data["universe"], bundle)
print(live)
sig = artifacts.load_frame("signals")
recent = artifacts.load_frame("events")
cols = ["date", "ticker", "company", "event_family", "ret_1d", "prob", "tier", "expected_net_return",
        "entry_ref_price", "stop_loss_price", "take_profit_price", "exit_by_date", "edgar_category"]
print("latest session:", live["data_through"])
(sig if len(sig) else recent.head(15))[cols]"""),
    md("## 6 · What a 24/7 agent sees (same functions as the REST API / MCP server)"),
    code("""from driftrev import service, broker
print(service.latest_signals(limit=5)["summary"])
print(json.dumps(service.regime(), indent=1, default=str)[:800])
preview = broker.preview_orders(max_positions=5, capital=10_000)
pd.DataFrame(preview.get("orders", []))"""),
])

# ------------------------------------------------------------------------------------
write("01_data_sources_and_eda.ipynb", [
    md("""# 01 · Data sources, the unified dataset and EDA

**Question:** after an information shock (an earnings release, or a ≥10% crash), does a US
stock keep moving the same way (**drift** — HW1 idea: momentum + PEAD) or snap back
(**reversal** — HW5 Q3: panic vs permanent damage)? Target: `growth_future_10d ≥ 1.02`
(enter at the next open, exit at the close 10 sessions later, ≥ +2% ≈ costs + profit).
"""),
    code(SETUP),
    code("""events = store.read_table("events")
with store.connect() as con:
    sizes = {t: con.execute(f"select count(*) from {t}").fetchone()[0]
             for t in ["prices", "market", "macro", "filings", "eps_facts", "pageviews", "daily_panel", "events"]}
print({k: f"{v:,}" for k, v in sizes.items()}); print("total records:", f"{sum(sizes.values()):,}")"""),
    md("### Events per family and year"),
    code("""tab = events.pivot_table(index=events.date.dt.year, columns="event_family", values="ticker", aggfunc="size", fill_value=0)
tab.plot(kind="bar", stacked=True, title="Events per year"); plt.show(); tab"""),
    md("### Target: distribution of the 10-day forward return by family"),
    code("""ml, feats = dataset.prepare(events)
known = ml[ml[dataset.TARGET].notna()]
fig, ax = plt.subplots()
for fam, g in known.groupby("event_family"):
    (g.growth_future_10d - 1).clip(-0.4, 0.4).plot(kind="hist", bins=80, alpha=0.4, density=True, ax=ax, label=fam)
ax.axvline(0.02, color="k", ls="--"); ax.legend(); ax.set_title("growth_future_10d - 1 (clipped ±40%)"); plt.show()
known.groupby("event_family").agg(n=("ticker", "size"), base_rate=(dataset.TARGET, "mean"),
    mean_10d=("growth_future_10d", lambda s: s.mean() - 1), median_10d=("growth_future_10d", lambda s: s.median() - 1),
    mean_10d_vs_spy=("growth_future_10d_vs_spy", lambda s: s.mean() - 1)).round(4)"""),
    md("### Event study: drift or reversal? (mean forward return vs SPY at 5/10/20 days)"),
    code("""es = known.groupby("event_family")[["growth_future_5d_vs_spy", "growth_future_10d_vs_spy", "growth_future_20d_vs_spy"]].mean() - 1
es.columns = ["+5d", "+10d", "+20d"]
es.T.plot(marker="o", title="Mean abnormal return after the event (vs SPY), by family"); plt.axhline(0, color="k"); plt.show()
es.round(4)"""),
    md("""### EDGAR taxonomy test (HW5 Q3, stage 1): do crash causes have different base rates?"""),
    code("""crash = known[known.event_family.isin(["shock_crash", "earnings_crash"])]
crash.groupby("edgar_category").agg(n=("ticker", "size"), bounce_rate=(dataset.TARGET, "mean"),
    mean_10d=("growth_future_10d", lambda s: s.mean() - 1), mean_mae=("mae_10d", "mean")).sort_values("n", ascending=False).round(3)"""),
    md("### PEAD (HW1): post-earnings drift by surprise (SUE) and by blended momentum"),
    code("""earn = known[known.event_family.str.startswith("earnings")].copy()
earn["sue_bucket"] = pd.qcut(earn["sue"], 5, duplicates="drop")
earn["mom_bucket"] = pd.qcut(earn["momentum_blend"], 5, duplicates="drop")
a = earn.groupby("sue_bucket", observed=True)["growth_future_10d_vs_spy"].mean() - 1
b = earn.groupby("mom_bucket", observed=True)["growth_future_10d_vs_spy"].mean() - 1
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
a.plot(kind="bar", ax=ax[0], title="10d abnormal return by SUE quintile")
b.plot(kind="bar", ax=ax[1], title="10d abnormal return by momentum quintile"); plt.tight_layout(); plt.show()"""),
    md("### Attention (Wikipedia pageviews) and market regime"),
    code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
eda = dataset.add_engineered(events); eda = eda[eda[dataset.TARGET].notna()]   # bins before one-hot encoding
eda.groupby("bin_attention")[dataset.TARGET].mean().plot(kind="bar", ax=ax[0], title="hit rate by attention spike")
eda.groupby("regime_state")[dataset.TARGET].mean().plot(kind="bar", ax=ax[1], title="hit rate by regime"); plt.tight_layout(); plt.show()"""),
    md("### Correlation of features with the target"),
    code("""num = [c for c in dataset.numeric_features() if c in known]
corr = known[num].corrwith(known[dataset.TARGET]).dropna().sort_values()
corr_g = known[num].corrwith(known.growth_future_10d).dropna()
top = pd.concat([corr.head(12), corr.tail(12)])
top.plot(kind="barh", figsize=(8, 7), title="corr(feature, is_positive_growth_10d)"); plt.show()
pd.DataFrame({"corr_target": corr, "corr_growth": corr_g}).sort_values("corr_target", key=abs, ascending=False).head(20).round(4)"""),
    md("### Dummy correlations (one-hot sets)"),
    code("""dums = [c for c in feats if c not in num]
dc = known[dums].corrwith(known[dataset.TARGET]).dropna().sort_values(key=abs, ascending=False)
dc.head(20).round(4)"""),
])

# ------------------------------------------------------------------------------------
write("02_modeling.ipynb", [
    md("""# 02 · Modeling

Protocol: hyper-parameters and probability thresholds are chosen on **VALIDATION** (fit on
TRAIN), then the chosen configuration is refit on TRAIN+VALIDATION and evaluated once on
**TEST** (2024 → today). Models: hand rules, DecisionTree, RandomForest (lecture) +
XGBoost, LightGBM, LogisticRegression (new) + an RF/XGB/LGBM probability ensemble.
"""),
    code(SETUP),
    code("""events = store.read_table("events"); ml, feats = dataset.prepare(events)
ml.groupby("split")[dataset.TARGET].agg(["size", "mean"])"""),
    md("### Hyper-parameter tuning table (validation)"),
    code("""tuning = pd.read_parquet(config.ARTIFACTS_DIR / "tuning.parquet")
tuning.sort_values(["model", "val_prec_top10"], ascending=[True, False]).groupby("model").head(3)"""),
    code("""tree = tuning[tuning.model == "tree"].copy(); tree["depth"] = tree.params.map(lambda p: json.loads(p)["max_depth"])
tree.plot(x="depth", y=["val_auc", "val_prec_top10"], marker="o", title="DecisionTree depth tuning"); plt.show()"""),
    md("### Model comparison (validation vs test AUC) and the production choice"),
    code("""metrics = artifacts.load_metrics()
print("production:", metrics["model"]["name"], "| threshold", metrics["model"]["threshold"])
pd.DataFrame(metrics["models"])[["name", "val_auc", "test_auc", "threshold"]].sort_values("val_auc", ascending=False)"""),
    md("### Decision rules on TEST (hand rules vs models vs custom rules)"),
    code("""artifacts.load_frame("pred_eval.parquet")"""),
    md("### Feature importance"),
    code("""imp = artifacts.load_frame("importance")
for m in ["rf", "xgb", "lgbm"]:
    s = imp[imp.model == m].head(20).set_index("feature")["importance"][::-1]
    s.plot(kind="barh", figsize=(7, 6), title=f"{m}: top-20 features"); plt.show()"""),
    md("### Calibration of the production probabilities on TEST"),
    code("""cal = artifacts.load_frame("calibration.parquet"); cal"""),
])

# ------------------------------------------------------------------------------------
write("03_simulation_and_analysis.ipynb", [
    md("""# 03 · Trading simulation, profitability vs benchmarks, deep analysis

Costs: 15 bps per side (30 bps round trip). Capital $10,000. TEST window 2024-01-01 → today.
"""),
    code(SETUP),
    code("""metrics = artifacts.load_metrics()
trades, eq = artifacts.load_frame("trades"), artifacts.load_frame("equity")"""),
    md("### Vector simulations (`sim1_` hold-to-horizon, `sim2_` with stop-loss / take-profit)"),
    code("""artifacts.load_frame("vector_sims.parquet").head(20).round(4)"""),
    md("### Exact simulations (daily loop, reinvestment, max concurrent positions) vs benchmarks"),
    code("""res = pd.DataFrame(metrics["strategies"] + metrics["benchmarks"])
res[["name", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "rolling_1y_median", "n_trades", "win_rate", "final_equity"]].sort_values("cagr", ascending=False).round(3)"""),
    code("""w = eq.pivot_table(index="date", columns="strategy", values="equity").ffill()
w.plot(logy=True, figsize=(12, 6), title="Equity curves on TEST (log scale)"); plt.show()
(w / w.cummax() - 1).plot(figsize=(12, 4), title="Drawdowns", legend=False); plt.show()
w.pct_change(126).plot(figsize=(12, 4), title="Rolling 6-month return", legend=False); plt.show()"""),
    md("### Trade anatomy of the best strategy"),
    code("""best = metrics.get("best_strategy"); print("best strategy:", best)
t = trades[trades.strategy == best]
display(t.groupby("exit_reason").agg(n=("pnl", "size"), avg_ret=("ret_net", "mean"), pnl=("pnl", "sum")).round(4))
display(t.groupby("event_family").agg(n=("pnl", "size"), win=("pnl", lambda s: (s > 0).mean()), avg_ret=("ret_net", "mean")).round(4))
t.sort_values("pnl").head(10)"""),
    md("### Where does the model work? Precision by event type, cause, sector, size, year, regime"),
    code("""bd = artifacts.load_frame("breakdown")
for dim in ["event_family", "edgar_category", "cap_bucket", "year", "regime_state", "sector", "prob_bucket"]:
    display(bd[bd.dimension == dim].sort_values("n", ascending=False).round(3))"""),
    md("### Debugging the most confident wrong predictions"),
    code("""artifacts.load_frame("false_positives.parquet")"""),
])

# ------------------------------------------------------------------------------------
write("04_rolling_window_pressure_release.ipynb", [
    md("""# 04 · Rolling-Window Pressure & Release (strategy guide, Variant B)

Source: `Global_Rolling_Window_Trading_Strategy_Guide.docx` (v1.0, 15 Nov 2026). This project covers
**US equities only**. US exchanges have no cumulative-deviation threshold rule (NYSE/Nasdaq LULD is a
5-minute band), so we implement the guide's **statistical variant (Variant B)** and treat "release"
purely as a feature. The guide's exchange-rule variant (Variant A, Arm D) is out of scope.

Indicators (Section 3), benchmark B = SPY:

* `D(t,L) = 100·(P_t/P_{t-L} − B_t/B_{t-L})` (percentage points)
* `M(t,W) = max_{L≤W} D(t,L)` for W = 10, 20
* flat-path release `Release(j) = M(t,W) − max(0, M(t,W−j))` — price and benchmark frozen at t,
  calendar advanced j sessions (closed form, so no future data enters)

Setup (Section 4 research defaults): 20-session median dollar volume ≥ $10m, prior 20d relative
strength top decile, 3–8 sessions since the impulse high, pullback < 2 ATR, then confirmation:
close > previous 5-session high, close in the top 25% of the day's range, volume ≥ 1.5× median.
"""),
    code(SETUP),
    code("""events = store.read_table("events"); ml, feats = dataset.prepare(events)
bo = ml[ml.event_family == "momentum_breakout"]
print(f"momentum_breakout events: {len(bo):,}  | arm A {int(bo.arm_a.sum()):,}  arm B {int(bo.arm_b.sum()):,}  arm C {int(bo.arm_c.sum()):,}")
bo[["rw_dev_20", "rw_max_dev_20", "rw_best_lag_20", "rw_release_20_1", "rw_release_20_3", "rw_release_20_5",
    "rw_days_since_high", "rw_pullback_atr", "rw_volume_mult_med", "rw_prior_strength"]].describe().round(2)"""),
    md("### Worked example: one Arm-C setup (window math on real prices)"),
    code("""from driftrev import rolling_window as rw, prices
ex = bo[bo.arm_c == 1].sort_values("date").iloc[-1]
px = prices.load_prices([ex.ticker], start=str((ex.date - pd.Timedelta(days=70)).date()))
spy = prices.load_market(); spy = spy[spy.ticker == "SPY"].set_index("date")["close"]
px = px.set_index("date"); b = spy.reindex(px.index)
wf = rw.window_features(px["close"].reset_index(drop=True), b.reset_index(drop=True)); wf.index = px.index
print(ex.ticker, ex.date.date(), "| M(t,20) = %.1f pp, best lag %d, Release(3) = %.1f pp" % (ex.rw_max_dev_20, ex.rw_best_lag_20, ex.rw_release_20_3))
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
px["close"].plot(ax=ax[0], title=f"{ex.ticker} close"); ax[0].axvline(ex.date, color="r", ls="--")
wf[["rw_max_dev_20", "rw_release_20_3"]].plot(ax=ax[1], title="M(t,20) and flat-path Release(3)"); ax[1].axvline(ex.date, color="r", ls="--")
plt.tight_layout(); plt.show()"""),
    md("### Four-arm comparison (Section 9.1) — event level, with bootstrap 95% CI of the mean 10-day return"),
    code("""arms = artifacts.load_frame("guide_arms.parquet"); arms.round(4)"""),
    code("""a = arms[arms["sample"] == "all_years"].set_index("arm")
plt.errorbar(a.index, a.mean_10d, yerr=[a.mean_10d - a.ci_low, a.ci_high - a.mean_10d], fmt="o", capsize=6)
plt.axhline(0, color="k"); plt.title("Mean 10-day return by arm (all years, 95% bootstrap CI)"); plt.show()"""),
    md("### Same execution & risk framework for every arm (Sections 6–7): exact simulations on TEST"),
    code("""metrics = artifacts.load_metrics()
g = [s for s in metrics["strategies"] if s["name"].startswith("guide_")]
pd.DataFrame([{**{k: s[k] for k in ("name", "cagr", "sharpe", "max_drawdown", "n_trades")},
               **{k: s["report_card"].get(k) for k in ("fill_rate", "avg_r_multiple", "profit_factor", "win_rate",
                                                         "top5_profit_share", "beta_vs_spy", "mean_ret_ci95")}} for s in g]).round(3)"""),
    md("### Key causal test and parameter sensitivity"),
    code("""print(json.dumps(metrics["guide"], indent=1))
sens = artifacts.load_frame("guide_sensitivity.parquet")
sens.pivot_table(index=["release_col", "threshold_pp"], columns="sample", values=["n", "mean_10d", "hit_rate_2pct"]).round(4)"""),
    md("### Cost stress test (1× / 2× / 3× the 15 bps one-way cost)"),
    code("""artifacts.load_frame("cost_stress.parquet").pivot_table(index="strategy", columns="cost_mult", values="cagr").round(4)"""),
    md("""### Interpretation

The guide's own rule decides: *if adding window release does not materially improve risk-adjusted
results over the same momentum + consolidation + breakout strategy, discard the rollover factor
rather than rebranding ordinary momentum.* The verdict printed above is computed, not hand-written;
the README reports it with the numbers. The release features stay in the ML feature set either way,
so the model can use (or ignore) them — their importance is visible in notebook 02.
"""),
])

# ------------------------------------------------------------------------------------
write("05_research_improvements.ipynb", [
    md("""# 05 · Research improvements: survivorship, which-not-when, walk-forward, homework borrowings

This notebook reads the tables produced by the second development wave (OPM SD1/SD2/SD3 changes):

| Topic | Table | Origin |
|---|---|---|
| Survivorship bias: current members vs point-in-time universe | `survivorship.parquet` | roadmap R1, HW1 Q1 change tables |
| Model variants: pooled vs cross-sectional vs per-family vs ranking | `model_comparison.parquet` | README §8 diagnosis |
| Book composition (exclude breakouts / `mgmt_change`?) | `composition.parquet` | README §8 diagnosis |
| Walk-forward yearly refits, stitched OOS track record | `walkforward.parquet`, `walkforward_equity.parquet` | roadmap R2 |
| Holding period by family | `holding_period.parquet` | HW2 Q3 |
| Uniquely-correct predictions | `uniquely_correct.parquet` | HW3 Q3 |
"""),
    code(SETUP),
    code("""metrics = artifacts.load_metrics()
def show(name):
    df = artifacts.load_frame(name)
    print(f"{name}: {'missing' if df.empty else f'{len(df)} rows'}")
    return df
print(json.dumps({k: metrics.get(k) for k in ("versions", "walkforward")}, indent=1, default=str)[:2500])"""),
    md("## 1 · Survivorship bias (R1)"),
    code("""show("survivorship.parquet")"""),
    md("## 2 · Which, not when: model variants (VALIDATION decides; TEST shown for reference)"),
    code("""show("model_comparison.parquet")"""),
    md("## 3 · Book composition"),
    code("""show("composition.parquet")"""),
    md("## 4 · Walk-forward track record (R2)"),
    code("""wf = show("walkforward.parquet"); wfe = show("walkforward_equity.parquet")
if not wfe.empty:
    piv = wfe.pivot_table(index="date", columns="strategy", values="equity") if "strategy" in wfe else wfe.set_index("date")["equity"]
    piv.plot(logy=True, figsize=(12, 5), title="Walk-forward equity (stitched out-of-sample)"); plt.show()
wf"""),
    md("## 5 · Holding period (HW2 Q3) and uniquely-correct predictions (HW3)"),
    code("""display(show("holding_period.parquet")); show("uniquely_correct.parquet")"""),
    md("## 5b · Capacity-aware threshold (HW4) and cap-aware costs"),
    code("""display(show("capacity_study.parquet")); show("cost_models.parquet")"""),
    md("## 5c · SEC filing text (R4): does the text separate panic from permanent damage?"),
    code("""ft = show("filing_text_study.parquet"); ft"""),
    md("## 5d · Walk-forward by year"),
    code("""pd.DataFrame((metrics.get("walkforward_detail") or {}).get("by_year", []))"""),
    md("## 6 · Final strategies vs benchmarks after the improvements"),
    code("""pd.DataFrame(metrics.get("strategies", []) + metrics.get("benchmarks", []))[
    ["name", "cagr", "sharpe", "max_drawdown", "n_trades", "win_rate"]].sort_values("sharpe", ascending=False).round(3)"""),
])
