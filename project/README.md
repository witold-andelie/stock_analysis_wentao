# Drift or Reversal? An event-driven ML trading system for the S&P 1500

Final project for the [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).

> **One question, asked every evening for every S&P 1500 stock that just had an information shock:**
> will the move **drift** (continue) or **revert** over the next 10 sessions, after costs?

Three hypotheses I developed during the course are tested side by side on one dataset:

| Event family | Origin | Hypothesis |
|---|---|---|
| `earnings_up` / `earnings_down` / `earnings_crash` | **HW1 capstone idea**: short-horizon momentum + post-earnings drift (PEAD) | Prices under-react to earnings news. The drift is stronger with trend support and a real EPS surprise (SUE). |
| `shock_crash` | **HW5 Q3**: panic vs permanent damage | After a ≥10% one-day fall on ≥2× volume, the bounce depends on *what was filed* with the SEC, not on how the tape moved. |
| `momentum_breakout` | **Strategy guide** *Rolling-Window Pressure & Release* (Variant B) | A top-decile relative-strength stock that consolidates, then breaks out, continues. Old gains "aging out" of the rolling window may add fuel. |

Everything runs end to end: data → features → models → simulations → daily signals → API and agent tools → CI/CD. It is designed **model-first** with an OPM architecture model ([`docs/architecture/opm/`](docs/architecture/opm/OPL.md)).

---

## Contents
1. [Problem and how the result is used](#1-problem-and-how-the-result-is-used)
2. [Results at a glance](#2-results-at-a-glance)
3. [Data sources](#3-data-sources)
4. [Features](#4-features-200-model-inputs)
5. [Transformations and EDA](#5-transformations-and-eda)
6. [Modeling](#6-modeling)
7. [Trading simulation](#7-trading-simulation)
8. [Deep analysis and what to improve](#8-deep-analysis-and-what-to-improve)
9. [Automation, CI/CD and data retention](#9-automation-cicd-and-data-retention)
10. [Agent interface, broker and dashboard](#10-agent-interface-for-a-247-bot-broker-and-dashboard)
11. [How to reproduce](#11-how-to-reproduce)
12. [Architecture (OPM) and repository layout](#12-architecture-opm-and-repository-layout)
13. [Limitations](#13-limitations-read-before-trusting-any-number)
14. [Evaluation-criteria map](#14-evaluation-criteria-map)

---

## 1. Problem and how the result is used

The course's base setup predicts 30-day growth for the largest stocks, every day. This project changes the
**unit of decision**. A row is an **event** rather than a ticker-day: an earnings release dated to the
minute from SEC EDGAR, a crash, or a momentum breakout. The prediction is made only when new information
has arrived. Events cover **large, mid and small caps** (S&P 500/400/600) and three economically
different mechanisms, so the analysis can ask *where* any edge lives.

**Target:** `is_positive_growth_10d = close[t+10] / open[t+1] ≥ 1.02`. Entry is at the next open, because the
signal is only known at the close. The exit is the close 10 sessions later. The +2% hurdle covers the
30 bps round-trip cost and leaves a profit.

**How it is used.** After each US close the system:
1. pulls only the missing data;
2. scores the session's events;
3. publishes tiered signals (`STRONG_BUY / BUY / WATCH / SKIP / AVOID / NO_CALL`) with an expected net
   return, a stop, a target and an exit date;
4. serves them to a personal 24/7 agent (Grok bot or any MCP / OpenAPI client), a dashboard, a static
   results site and an optional Alpaca paper-trading account.

## 2. Results at a glance

TEST period = **2024-01-02 → 2026-11-13**, strictly out of sample. Every hyper-parameter, threshold and
production-model choice was made on VALIDATION (2022–23). Capital is $10,000 and costs are 15 bps per side.

| Strategy (exact simulation, reinvestment) | CAGR | Sharpe | Max DD | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| `model_crash_only`: model ≥ 0.47 on crash events, top 5 | **37.5%** | **1.58** | **−9.7%** | 54 | 70% |
| `model_top5`: model ≥ 0.47, any family, max 5 positions | 35.2% | 1.28 | −14.9% | 92 | 59% |
| `model_top5_risk_managed`: + stop 15% / TP 25%, prob-weighted size, regime pause | 19.5% | 1.39 | −10.9% | 100 | 59% |
| `all_events_unconditional`: buy every event, first come, max 5 | 40.6% | 1.03 | −49.6% | 360 | 53% |
| `model_pead_only` (HW1 idea, earnings only) | 9.2% | 0.67 | −16.6% | 52 | 63% |
| `model_breakout_only` | 5.3% | 0.36 | −25.6% | 61 | 48% |
| `guide_model_all_events_R`: guide risk template (0.5% risk/trade) | 2.6% | 0.78 | −3.2% | 93 | 53% |
| **Benchmark: SPY buy & hold** | 19.1% | 1.19 | −18.8% | | |
| Benchmark: IWM buy & hold | 13.4% | 0.71 | −27.5% | | |
| Benchmark: HW1 momentum rule (top quintile, 10-day rebalance) | 23.0% | 1.05 | −24.1% | | |

**Reading this honestly:**

* **Precision gain:** the production model (RandomForest, test AUC **0.541**) lifts precision from a
  **39.9% base rate to 49.2%** at its validation-chosen threshold. Calibration is monotonic: the bottom
  probability decile hits 34.7%, the top decile 47.7%. The edge is **real but small**.
* **Model vs benchmark:** `model_top5` and `model_crash_only` beat SPY on CAGR, Sharpe *and* drawdown.
  However, ~50% of `model_top5` profit comes from 5 trades (bootstrap 95% CI of the mean trade return:
  +1.4% … +8.9%).
* **Survivorship bias:** even the dumb `all_events_unconditional` makes 40.6% CAGR. That is a red flag for
  **survivorship bias**: the universe is *today's* S&P 1500, so companies that crashed and left the index
  are missing. Treat every absolute return as an upper bound; the *relative* comparisons are what this
  project can defend (see [Limitations](#13-limitations-read-before-trusting-any-number)).
* **Strategy-guide causal test:** **discard the rolling-window release factor.** Arm C (with release)
  beats Arm B on the 2024+ test sample (+1.58% vs +0.81% mean 10-day return, Sharpe 0.56 vs 0.30). Over
  all years it is worse (+0.07% vs +0.23%), all bootstrap CIs include 0, and the sign flips across release
  thresholds. The guide's own go/no-go rule (stable across periods and parameters) therefore rejects it.
  Arm D (exchange-rule events) does not apply to US stocks.

The daily results site (GitHub Pages, produced by CI) shows today's signals; see §9.

## 3. Data sources

| Source | What | Use | New vs lectures | Rows (2014-01 → 2026-11) |
|---|---|---|---|---:|
| **Yahoo Finance** (`yfinance`) | daily OHLCV + dividends + splits for 1,502 stocks; SPY, IWM, MDY, ^VIX, ^VIX3M, 11 sector ETFs | prices, targets, technicals, benchmarks | lecture source | 4,426,339 + 50,226 |
| **FRED** (fredgraph CSV) | DGS10, DGS2, DFF, BAA10Y, T10Y2Y | rates, curve, credit-spread regime flag | lecture source | 15,136 |
| **SEC EDGAR** submissions API | every filing: form, **8-K item codes**, acceptance time (UTC → ET) | minute-accurate earnings dates; crash-cause taxonomy; offering, late-filing, 13D and Form 4 activity | **new** | **1,896,174** |
| **SEC EDGAR** XBRL `companyconcept` | diluted EPS facts (first-reported) | point-in-time SUE | **new** | 334,462 |
| **Wikimedia** pageviews API | daily views of each company's Wikipedia article | abnormal public attention | **new** | **4,522,878** |
| **Wikipedia** S&P 500/400/600 lists | constituents, GICS sector, article title | universe, sector, size bucket | **new** (scraped) | 1,502 |

**≈ 11.2 million raw records.** The derived daily panel has 4,425,338 ticker-days, and the event dataset
has 176,802 rows × 202 model inputs. No data is committed to git (see §9 for where it lives).

## 4. Features (202 model inputs)

All features at row *t* use information available by the close of *t*. The only exception is filings
accepted before 09:00 ET on *t+1*, since the trade enters at that open. FRED and pageviews are lagged one
row. `tests/test_core.py::test_features_have_no_look_ahead` enforces this.

**Event reaction (new):** `abn_ret_1d` (beta-adjusted), `ret_vs_sector` (vs the sector SPDR), `gap_open`,
`range_pct`, `close_loc`, `ret_open_close`, `volume_mult` (vs the prior 20 sessions), `volume_z`
(log-volume z-score).

**Momentum, from the HW1 idea:** `growth_{3,5,10,20,60,120,252}d` (total return), `mom_12_1`,
`close_sma{20,50,200}`, `sma50_sma200`, `rs_vs_spy_60d`, `dist_52w_high/low`, `up/down_streak_prior`, and
**`momentum_blend`** = the HW1 score (price vs the 20/50/200-day MAs + 12-1 momentum).

**Technicals** (TA-Lib formulas reimplemented in pandas so Docker needs no C library): RSI-14, MACD
histogram, Bollinger %b, Stochastic %K, CCI-20, MFI-14, ADX-14, OBV slope, ATR %.

**Risk and liquidity:** `vol_20d`, `vol_60d`, `vol_ratio_20_60`, `beta_252`, `idio_vol_60`, `log_dollar_vol`.

**Earnings history (new):** `prev_ear` (previous earnings-day abnormal return), `prev_pead_20d` (previous
quarter's realised drift), `beats_last4`, `days_since_prev_event`, `n_crash_prior_252`.

**SEC EDGAR (new source):**
- `sue` (seasonal random-walk standardized unexpected earnings from *first-reported* diluted EPS, with
  Q4 = FY − Q1..Q3), `eps_yoy_pos`, `sue_age_days`;
- 14 **8-K item flags** in the event window (1.01 agreement, 1.02 termination, 1.03 bankruptcy,
  2.01 acquisition, 2.02 results, 2.05 restructuring, 2.06 impairment, 3.01 delisting notice,
  3.02 unregistered equity, 4.01 auditor change, 4.02 non-reliance/restatement, 5.02 officer change,
  7.01 Reg FD, 8.01 other);
- form flags (offering 424B/S-1/S-3, NT 10-K/Q late filing, periodic report) and prior-activity counts
  (`n_offering_365d`, `n_8k_90d`, `n_form4_30d` insider filings, `n_13d_180d` activist stakes,
  `n_late_filing_365d`);
- **`edgar_category`**, a priority taxonomy (most damaging cause wins): bankruptcy_delisting >
  restatement_auditor > offering > earnings > impairment_restructuring > mgmt_change > deal >
  regfd_other_8k > none.

**Attention (new source):** `att_ratio_7_90` (log views over 7 days vs the prior 90), `att_spike_1d`,
`att_level`.

**Rolling-window state (strategy guide):**
- `rw_dev_{10,20}` = D(t,L) = 100·(P_t/P_{t−L} − B_t/B_{t−L});
- `rw_max_dev_{10,20}` = M(t,W) = max over L ≤ W, plus `rw_best_lag_*` and `rw_proximity_*`;
- `rw_release_{10,20}_{1,3,5}` = flat-path release M(t,W) − max(0, M(t,W−j)). This is a closed form: price
  and benchmark are frozen at *t*, so no future data enters;
- `rw_prior_strength` (cross-sectional percentile of D(t,20) among liquid names, maximum over the last
  10 sessions), `rw_days_since_high`, `rw_pullback_atr`, `rw_range_compression`, `rw_breakout_5d`,
  `rw_volume_mult_med`, and arm flags `arm_a/b/c`.

**Market and regime (fragility monitor idea from my TradePulse project):** SPY/IWM returns, drawdown,
above-200-day flag, small-vs-large spread, VIX level/change, **VIX/VIX3M term structure**, average
pairwise correlation of the 11 sector ETFs, the FRED series, `breadth_above_200`, `crash_breadth`
(share of the universe crashing today), and `regime_points` → `regime_state`.

**Dummies (7 categoricals + 7 engineered bins):** `event_family`, `edgar_category`, `sector`, `cap_bucket`,
`month`, `weekday`, `regime_state`, plus binned `bin_reaction`, `bin_rsi`, `bin_sue`, `bin_52w`, `bin_vix`,
`bin_attention`, `bin_momentum`.

## 5. Transformations and EDA

`features.build_all` builds **one unified event table** from all sources (notebook `01`). `dataset.prepare`
defines the course feature sets (`TO_PREDICT`, `NUMERIC`, `CATEGORICAL` → `DUMMIES`, `TO_DROP`) and the
temporal split: train ≤ 2021 (99,735 events), validation 2022–23 (30,236), test 2024 → (46,831).

Point-in-time engineering that matters:
- **Earnings dating:** an 8-K Item 2.02 accepted after 16:00 ET belongs to the *next* session. Yahoo
  earnings calendars are not point-in-time, so EDGAR acceptance times are the ground truth.
  (`acceptanceDateTime` is genuine UTC: Apple's 16:30 ET releases appear as 20:30Z in summer and 21:30Z
  in winter.)
- **SUE:** uses the first-reported value of each quarter, so later restatements cannot leak.
- **Total return:** built from stored dividends instead of Yahoo's *Adj Close*, which rewrites history on
  every dividend and would silently corrupt incremental loads.

**Key EDA findings** (all years, 2014–2026, notebook `01`):

| Family | Events | Base rate (≥ +2% in 10 days) | Mean 10-day return | Mean vs SPY, +10d / +20d |
|---|---:|---:|---:|---:|
| `shock_crash` | 3,256 | **46.2%** | +1.68% | **+1.41% / +1.58%**: reversal |
| `earnings_crash` | 3,771 | 41.9% | +0.70% | +0.14% / +0.32% |
| `earnings_up` | 36,015 | 38.5% | +0.58% | −0.03% / +0.03% |
| `earnings_down` | 29,862 | 37.8% | +0.57% | −0.04% / −0.01% |
| `momentum_breakout` | 102,818 | 36.8% | +0.26% | −0.07% / −0.12% |

* **Reversal is real after non-earnings crashes; unconditional PEAD is not** (in this universe and period).
  Earnings events drift ≈ 0 vs SPY on average. This is why the HW1 hand rule is weak and the
  conditioning has to come from the model.
* **The EDGAR cause matters (HW5 Q3, stage 1).** Crashes accompanied by an **offering** have a *negative*
  mean 10-day return (−1.0%), while crashes with no filing bounce +1.6%. Dilution behaves like
  "permanent damage", as hypothesised.
* **Breakouts slightly lag SPY** at every horizon, which already hints at the four-arm verdict.

EDA (notebook `01`) covers:
- target distribution per family;
- an event study of the abnormal return vs SPY at +5/+10/+20 days per family;
- crash base rates by EDGAR cause (HW5 Q3, stage 1);
- PEAD by SUE quintile and by momentum quintile (HW1);
- hit rate by attention spike and by regime;
- correlations of numeric features and dummies with the target.

## 6. Modeling

* **Hand rules first:**
  - `pred0_manual_pead_momentum` (HW1);
  - `pred1_manual_panic_no_damage` (crash + RSI < 35 + no damaging filing, HW5);
  - `pred2_manual_all_events`;
  - the guide arms A/B/C.
* **Lecture models:** DecisionTree (depth 3–10) and RandomForest (300 trees, depth × leaf grid).
* **New models:** XGBoost, LightGBM, LogisticRegression (scaled), and an **RF + XGB + LGBM probability
  ensemble**.
* **Tuning:** each grid point is fit on TRAIN and scored on VALIDATION by precision in the top decile
  (what the strategy trades), with AUC as tie-breaker. Results are in `data/artifacts/tuning.parquet` and
  notebook `02`.
* **Decision rules:**
  - each model's threshold is the highest-precision cut-off on VALIDATION that still trades ≥ 5% of
    events (`pred*_best_rule_NN`);
  - custom rule `pred9_ensemble_top1_daily` takes only the single most confident event per day;
  - the strategies add max-concurrent-position rules.

TEST results (precision, base rate 39.9%):

| Prediction | Trades | Precision | Avg 10-day growth |
|---|---:|---:|---:|
| `pred4_logreg_best_rule_71` | 53 | **56.6%** | +6.6% |
| `pred9_ensemble_top1_daily` | 423 | 50.8% | +3.2% |
| `pred5_rf_best_rule_47` (production) | 992 | 49.2% | +2.1% |
| `pred8_ensemble_best_rule_47` | 7,204 | 46.0% | +1.8% |
| `pred1_manual_panic_no_damage` (HW5 rule) | 1,587 | 43.4% | +1.2% |
| `pred2_manual_all_events` | 45,751 | 39.9% | +0.7% |
| `pred0_manual_pead_momentum` (HW1 rule) | 3,815 | 38.9% | +0.3% |

| Model | Validation AUC | Test AUC |
|---|---:|---:|
| RandomForest (production: best validation AUC among rf/xgb/lgbm/ensemble) | 0.548 | **0.541** |
| Ensemble | 0.516 | 0.539 |
| LightGBM | 0.509 | 0.537 |
| XGBoost | 0.509 | 0.535 |
| LogisticRegression | 0.550 | 0.531 |
| DecisionTree | 0.509 | 0.519 |

## 7. Trading simulation

* **Vector simulations** (`simulate.vector_sim`) use the course's `sim1_` convention: $100 per positive
  prediction, fees, capital = $100 × 10 days × Q75(daily bets). `sim2_` adds a stop-loss and take-profit
  from each event's intraperiod MAE/MFE. They run for every `pred*` column. The best is
  `pred9_ensemble_top1_daily` with an approximate CAGR of 33.1% (sim1).
* **Exact simulations** (`simulate.exact_sim`) are a day-by-day loop with a cash account and
  **reinvestment of gains**:
  - **entry and exit:** entry at the next open, stop / target / time exits on the real daily OHLC path
    (a gap through the stop fills at the open);
  - **position limits:** max concurrent positions, skipping tickers already held;
  - **sizing:** probability-weighted sizes, and the **regime pause** (no new entries when the fragility
    monitor is *stressed*);
  - **long-short:** a long-short book that shorts events with prob ≤ 0.30;
  - **guide risk template:** stop at 2×ATR, target at 3R, 8-session time stop, 0.5% of equity risked per
    trade, ≤ 10% notional per name, ≤ 2% total open risk, skip gap-ups above +3%, and no new entries after
    3 stop-outs in a day.
* **13 strategies vs 3 benchmarks:** SPY, IWM and the HW1 momentum rule (§2). Metrics are CAGR, Sharpe,
  Sortino, max drawdown, Calmar, rolling 1-year returns, win rate and average trade.
* **Report card** (strategy guide §9.3) for every strategy: fill rate, profit factor, average R, top-5
  profit share, beta vs SPY, bootstrap 95% CI of the mean trade return, turnover.
* **Cost stress:** results at 2× and 3× costs.
* **Four-arm causal test:** with a parameter-sensitivity grid (notebook `04`).

## 8. Deep analysis and what to improve

Notebook `03`, `precision_breakdown.parquet` and `false_positives.parquet` (TEST, production rule):

| Slice | Base rate | Precision of selected | Lift |
|---|---:|---:|---:|
| `earnings_down` / `earnings_up` | 39.8% / 40.0% | 58.7% / 58.4% | **+18.8 / +18.4 pp** |
| `shock_crash` | 47.3% | 62.5% | +15.2 pp |
| `momentum_breakout` | 39.5% | 36.3% | **−3.2 pp** |
| large / mid / small caps | 40.5 / 39.7 / 39.6% | 54.3 / 51.6 / 45.2% | +13.7 / +11.9 / +5.7 pp |
| 2024 / 2025 | 39.7 / 39.4% | 34.2 / 58.8% | −5.5 / +19.4 pp |
| EDGAR `mgmt_change` | 41.4% | 8.3% (12 trades) | −33 pp |

What the diagnostics say:
1. **The model learns *when* more than *which*.** The top RF importances are market-level (DGS10, the
   credit spread, VIX term structure, sector correlation, breadth), and precision swings from 34% (2024)
   to 59% (2025). *Next step:* demean features cross-sectionally by date, or add a per-date ranking
   objective (LightGBM `lambdarank`), so the model must pick stocks rather than time the market.
2. **Breakouts dilute the book.** The model has negative lift on `momentum_breakout`, and the guide's
   Arm C fails its stability test. *Next step:* drop breakouts from the model book (or train per-family
   models) and keep the guide arms as a separate rule-based sleeve.
3. **Panic vs damage works where the thesis says.** Crash events have the highest base rate (47%) and the
   model adds +15 pp. Events tagged `mgmt_change` almost never bounce, which is a candidate exclusion rule
   in the spirit of HW5 Q3.
4. **Small caps carry the least edge** and the highest costs. *Next step:* a cap-aware cost model
   (higher bps for small caps) and a minimum dollar-volume tier for live trading.
5. **Concentration:** about half the profit of `model_top5` comes from 5 trades. The fill rate is only
   8.5% (capacity-bound), so a ranking-quality improvement matters more than the threshold.

## 9. Automation, CI/CD and data retention

```bash
python -m driftrev.pipeline --mode update                 # daily: incremental download → signals → report
python -m driftrev.pipeline --mode update --retrain       # + refit models and all backtests
python -m driftrev.pipeline --mode update --retrain --tune   # + grid search
python -m driftrev.pipeline --mode file --retrain         # reproduce from the snapshot, no downloads
python -m driftrev.pipeline snapshot                      # export data/driftrev_snapshot.zip
python -m driftrev.housekeeping [--dry-run]               # delete expired data files
```

* **Two regimes.** `update` downloads from the APIs. `file` reads the parquet snapshot (`events`,
  `panel`, `market`, `universe`, `ctx`) from `data/snapshot/`, or fetches it once from
  `DRIFTREV_SNAPSHOT_URL`. CI publishes that zip weekly as the GitHub Release asset `data-snapshot`.
* **Incremental loading into DuckDB** (`data/warehouse.duckdb`, git-ignored):
  - each loader asks for its last stored date and downloads only the gap (prices: 7-day overlap, and a
    split triggers a full re-download of that ticker);
  - FRED re-pulls the last 60 days because values get revised;
  - EDGAR re-reads only the `recent` block for known companies;
  - pageviews are fetched from each article's last date.
* **Cron:** `scripts/run_daily.sh` + `scripts/crontab.example` (22:30 New York, Mon–Fri), or the
  `scheduler` container. Every run is recorded in `run_log.jsonl`.
* **One notebook runs the whole workflow:** `notebooks/00_end_to_end_workflow.ipynb` calls the `.py`
  functions in order (download → transform → predict → simulate → latest new trades → agent view).

**CI/CD on GitHub Actions** (OPM `SD7_cicd.dot`):

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | push / PR on `project/**` | pytest (API, core invariants, retention, CI contract), every OPM diagram renders and has an OPL section, notebooks valid, `docker compose config`, `docker build` + import check |
| `.github/workflows/daily.yml` | 22:30 ET Mon–Fri; Sunday retrain; manual | restore the warehouse from the Actions cache → `pipeline --mode update` → job summary, **GitHub Pages results site** (`index.html`, `signals.json`, `metrics.json`), run artifact (expires after 30 days), save cache; Sunday: release the snapshot asset |
| `.github/workflows/cleanup.yml` | weekly; manual (dry-run option) | `scripts/gh_cleanup.sh`: delete Actions caches > 3 days (keep the newest), artifacts > 30 days, workflow runs > 30 days, old snapshot assets |

**Data retention** (`config.Retention`, run after every pipeline run):
- EDGAR root JSONs older than 2 days and history pages older than 180 days are deleted;
- the daily signal archive is kept for 60 days;
- only the newest snapshot zip is kept;
- the run log is capped at 1,000 lines and logs at 5 MB;
- the warehouse is checkpointed afterwards.

`tests/test_cicd.py` fails if the workflow retention numbers drift from `config.py`.

**One-time GitHub setup:** Settings → Pages → Source = *GitHub Actions*. Add the repository secret
`SEC_USER_AGENT` = `"Your Name your@email"` (the SEC requires a contact). The first scheduled run with an
empty cache bootstraps the full history (about 1 hour).

## 10. Agent interface for a 24/7 bot, broker and dashboard

* **REST API** (`driftrev.api`, FastAPI, bearer token, OpenAPI with clean operation IDs):
  - `/health`;
  - `/v1/signals/latest`, `/v1/events/recent`, `/v1/tickers/{t}`;
  - `/v1/backtest/{summary,trades,equity}`, `/v1/model`, `/v1/regime`, `/v1/runs`;
  - `POST /v1/pipeline/run` → `/v1/jobs/{id}`;
  - `POST /v1/orders/preview`, `GET /v1/portfolio`;
  - `/v1/agent/tools`.
* **Function calling:** `agent/tools.json` uses the OpenAI/xAI format, so a Grok bot can load it directly.
  `agent/grok_agent_example.py` is a working tool loop against `https://api.x.ai/v1`.
* **MCP server:** `python -m driftrev.mcp_server --transport stdio|streamable-http` exposes the same tools.
* See [`agent/README.md`](agent/README.md) for how to expose it safely (token, Tailscale or Cloudflare
  Tunnel) and for a daily-brief prompt.
* **Broker (Alpaca, paper by default):**
  - `broker.preview_orders` sizes bracket orders from the signal book;
  - `submit_orders` refuses unless `confirm=true` **and** `DRIFTREV_ALLOW_ORDERS=true`, and it is not
    exposed as an agent tool;
  - `close_expired_positions` exits after `exit_by_date`.
* **Dashboard** (`streamlit run dashboard/app.py`): overview KPIs vs SPY, latest signals, equity curves,
  trades explorer, model importance and breakdowns, rolling-window guide report card, pipeline runs.
  The GitHub Pages site is the always-on read-only version.
* **Containers:** `docker compose up -d` starts `api` (:8000), `mcp` (:8001), `dashboard` (:8501) and
  `scheduler` on a shared `/data` volume.

## 11. How to reproduce

```bash
git clone https://github.com/witold-andelie/stock_analysis_wentao.git
cd stock_analysis_wentao/project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # pinned: scikit-learn 1.7.2, numpy 1.26.4, pyarrow 18.1 …
cp .env.example .env                            # set SEC_USER_AGENT="Your Name your@email"
export PYTHONPATH=src

# quick smoke run on 50 tickers (~3 min)
DRIFTREV_MAX_TICKERS=50 python -m driftrev.pipeline --mode update --retrain --tune

# full S&P 1500 history: ~20 min download + ~6 min transform + ~25 min tuning/backtests on 8 cores
python -m driftrev.pipeline --mode update --retrain --tune

# or without any download, from the CI snapshot:
DRIFTREV_SNAPSHOT_URL=https://github.com/witold-andelie/stock_analysis_wentao/releases/download/data-snapshot/driftrev_snapshot.zip \
  python -m driftrev.pipeline --mode file --retrain

python -m pytest tests -q                       # 27 tests
jupyter notebook notebooks/                     # 00 … 04, executed with outputs
```

## 12. Architecture (OPM) and repository layout

Development is **model-first** ([`AGENTS.md`](AGENTS.md)): every feature is first added to the OPM model
(ISO 19450, Graphviz DOT + the OPL text twin), then implemented together with a test.

| Diagram | Scope |
|---|---|
| [`SD0_system`](docs/architecture/opm/svg/SD0_system.svg) | system function *Signalling Event Trades*, environment, top-level processes |
| [`SD1_acquiring`](docs/architecture/opm/svg/SD1_acquiring.svg) | incremental loading of the 5 sources into 8 warehouse tables |
| [`SD2_transforming`](docs/architecture/opm/svg/SD2_transforming.svg) | features, event detection and specialisation, EDGAR taxonomy, SUE, attention, regime |
| [`SD3_researching`](docs/architecture/opm/svg/SD3_researching.svg) | split, tuning, final fit, thresholds, simulations, benchmarks, four-arm test |
| [`SD4_signalling`](docs/architecture/opm/svg/SD4_signalling.svg) | scoring, net edge, tiering with regime downgrade, sizing and exits, publishing |
| [`SD5_serving_trading`](docs/architecture/opm/svg/SD5_serving_trading.svg) | REST / tools.json / MCP, auth, jobs, dashboard, order guard |
| [`SD6_roadmap`](docs/architecture/opm/svg/SD6_roadmap.svg) | planned R1–R7 |
| [`SD7_cicd`](docs/architecture/opm/svg/SD7_cicd.svg) | CI, daily CD, Pages, snapshot release, housekeeping, GitHub garbage collection |

```
project/
  src/driftrev/   config · store · universe · prices · macro · edgar · attention        (ingestion)
                  indicators · rolling_window · features · dataset                      (transformation)
                  models · simulate · analysis                                          (research)
                  signals · pipeline · artifacts · report · housekeeping                (signalling + ops)
                  api · service · security · mcp_server · agent_tools · broker · scheduler (serving / execution)
  notebooks/      00 end-to-end · 01 data & EDA · 02 modeling · 03 simulation & analysis · 04 rolling-window guide
  dashboard/      Streamlit monitor            agent/   tools.json, routes.json, Grok example, agent README
  docs/architecture/opm/   *.dot, OPL.md, svg/  scripts/ build_notebooks, render_opm, gh_cleanup, run_daily
  tests/          27 tests                      Dockerfile · docker-compose.yml · requirements.txt · .env.example
../.github/workflows/   ci.yml · daily.yml · cleanup.yml
```

## 13. Limitations (read before trusting any number)

* **Survivorship bias (largest).** The universe is the *current* S&P 1500. Delisted or demoted crash
  victims are missing, which inflates crash-reversal and unconditional returns. Roadmap R1 adds the
  historical add/remove tables.
* **Small edge, concentrated P&L.** Test AUC is 0.54, and about half the best strategy's profit comes from
  5 trades. One market regime (2025) carries most of the result.
* **Costs.** A flat 15 bps per side is optimistic for small caps right after a crash. The cost-stress
  table shows the strategies at 2× and 3×.
* **Execution model.** Daily bars only: stops are checked against the low (gap fills at the open), and
  there is no partial-fill or short-borrow model.
* **Pageviews:** 6 of 1,247 articles were rate-limited during the first ingest and are filled
  incrementally later.
* **Disclaimer:** research and education only, not investment advice. Orders are paper-only and disabled
  by default.

## 14. Evaluation-criteria map

| Criterion | Where |
|---|---|
| Problem description, context, new problem definition, step-by-step findings | §1, §2, §5–§8; notebooks 01–04 |
| Lecture data + 20+ new features with descriptions | §3, §4 (YF/FRED + ~150 new features) |
| New data source (not YF/FRED) | SEC EDGAR filings + XBRL, Wikimedia pageviews, Wikipedia constituents |
| Dataset > 1M records | 11.2M raw records; 4.43M-row daily panel |
| Combined dataframe, feature sets, ≥ 5 new transformations, EDA | `features.build_all`, `dataset.py`, notebook 01 |
| Lecture models, several models, custom decision rules, tuning, new models | §6, `models.py`, notebook 02 |
| Vector sims, ≥ 2 strategies, exact sims with reinvestment | §7, `simulate.py`, notebook 03 |
| Profitability vs benchmark (CAGR, Sharpe, DD, rolling returns) | §2, notebook 03 |
| Advanced risk management, new strategies (max positions, regime, long-short, guide R) | §7 |
| Exceptional profitability vs SPY | `model_top5`, `model_crash_only` (with the survivorship caveat, §13) |
| Deep analysis by family / size / year / regime / cause, debug wrong predictions, ideas | §8, notebook 03 |
| Notebooks exported to scripts + one end-to-end notebook | `src/driftrev/*`, notebook 00 |
| Dependencies managed | `requirements.txt`, `.venv`, Dockerfile |
| Cron re-run for the latest data | `run_daily.sh`, `crontab.example`, scheduler container, `daily.yml` |
| Two regimes (file vs download) | `--mode file` / `--mode update` |
| Incremental loading with storage outside GitHub | DuckDB warehouse, Actions cache, Release snapshot |
| Bonus: code design | OPM model + module boundaries + 27 tests + CI |
| Bonus: broker API | `broker.py` (Alpaca paper, bracket orders, order guard) |
| Bonus: monitoring | Streamlit dashboard, GitHub Pages results site, REST/MCP agent interface |
| Bonus: containerization | Dockerfile + docker-compose (4 services) |
| Bonus: cloud deployment | GitHub Actions daily CD + GitHub Pages |
