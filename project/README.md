# Drift or Reversal? An event-driven ML trading system for the S&P 1500

Final project for the [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).
Scope: **US equities** and **German equities on Deutsche Börse Xetra** (DAX + MDAX + SDAX; `DRIFTREV_MARKET=de`), with the German extension documented in [§15](#15-extension-german-equities-on-xetra).

> **The question, asked every evening for every S&P 1500 stock that just had an information shock:**
> will the move **drift** (continue) or **revert** over the next 10 sessions, after costs?

Three hypotheses I developed during the course are tested side by side on one point-in-time dataset:

| Event family | Origin | Hypothesis |
|---|---|---|
| `earnings_up` / `earnings_down` / `earnings_crash` | **HW1 capstone idea**: short-horizon momentum + post-earnings drift (PEAD) | Prices under-react to earnings news. The drift is stronger with trend support, a real EPS surprise (SUE / analyst consensus) and guidance language. |
| `shock_crash` | **HW5 Q3**: panic vs permanent damage | After a ≥10% one-day fall on ≥2× volume, the bounce depends on *what was filed* with the SEC (8-K items and the text of the filing), not on how the tape moved. |
| `momentum_breakout` | **Strategy guide** *Rolling-Window Pressure & Release* (Variant B) | A top-decile relative-strength stock that consolidates, then breaks out, continues. Old gains "aging out" of the rolling window may add fuel. |

The system runs end to end: data → features → models → simulations → daily signals → REST / MCP / tools for a
24/7 agent → CI/CD on GitHub Actions. It is built **model-first** against an OPM architecture model
([`docs/architecture/opm/`](docs/architecture/opm/OPL.md)), and its research numbers passed an independent
leakage and accounting audit (§8).

---

## Contents
1. [Problem and how the result is used](#1-problem-and-how-the-result-is-used)
2. [Results at a glance: the honest version](#2-results-at-a-glance-the-honest-version)
3. [Data sources](#3-data-sources)
4. [Features](#4-features-269-model-inputs)
5. [Transformations and EDA](#5-transformations-and-eda)
6. [Modeling](#6-modeling)
7. [Trading simulation](#7-trading-simulation)
8. [Deep analysis, audit and what to improve](#8-deep-analysis-audit-and-what-to-improve)
9. [Automation, CI/CD and data retention](#9-automation-cicd-and-data-retention)
10. [Agent interface, notifications, broker and dashboard](#10-agent-interface-notifications-broker-and-dashboard)
11. [How to reproduce](#11-how-to-reproduce)
12. [Architecture (OPM) and repository layout](#12-architecture-opm-and-repository-layout)
13. [Limitations](#13-limitations)
14. [Evaluation-criteria map](#14-evaluation-criteria-map)
15. [Extension: German equities on Xetra](#15-extension-german-equities-on-xetra)

---

## 1. Problem and how the result is used

The course's base setup predicts 30-day growth of the largest stocks every day. This project changes the
**unit of decision**. A row is an **event** rather than a ticker-day: an earnings release dated to the
minute from SEC EDGAR, a crash, or a momentum breakout. Events cover **large, mid and small caps**
(S&P 500/400/600), only while the stock was actually in the index (point-in-time membership), and span
three economically different mechanisms, so the analysis can ask *where* any edge lives.

**Target:** `is_positive_growth_10d = close[t+10] / open[t+1] ≥ 1.02`. Entry is at the next open, because
the signal is only known at the close. The exit is the close 10 sessions later. The +2% hurdle covers
round-trip costs (8 / 15 / 25 bps per side for large / mid / small caps) and leaves a profit.

**How it is used.** After each US close the system:
1. pulls only the missing data;
2. scores the session's events;
3. publishes tiered signals (`STRONG_BUY / BUY / WATCH / SKIP / AVOID / NO_CALL`) with an expected net
   return, stop, target and exit date;
4. pushes them to Telegram or a webhook, serves them to a personal 24/7 agent (Grok bot or any MCP /
   OpenAPI client), a dashboard and a GitHub Pages results site, and optionally sizes paper orders on Alpaca.

## 2. Results at a glance: the honest version

The project's most important result is **how much the first version overstated returns**, and why.

**Out-of-sample TEST, 2024-01-02 → 2026-11-13** (exact simulation, $10,000, reinvestment, cap-aware costs).
Every hyper-parameter, threshold, feature set and book rule was chosen on VALIDATION (2022–23).

| Strategy | First version: today's S&P 1500, no audit | **Final: point-in-time universe + audit fixes** | Sharpe (final) | Max DD (final) |
|---|---:|---:|---:|---:|
| Buy every event, no model (`all_events_unconditional`) | CAGR +40.6% | **−8.5%** | −0.08 | −57.9% |
| `model_top5`: ensemble ≥ 0.52, ≤ 5 positions | +35.2% | **+8.4%** | 0.43 | −18.0% |
| `model_crash_only`: same model, crash events only | +37.5% | **+16.4%** | 0.71 | −19.2% |
| **SPY buy & hold** | +19.1% | **+19.1%** | 1.19 | −18.8% |
| IWM buy & hold | +13.4% | +13.4% | 0.71 | −27.5% |
| HW1 momentum rule (top quintile, point-in-time) | +23.0% | +17.0% | 0.86 | −23.7% |

**Walk-forward 2018 → 2026** (yearly expanding refits, 16-day purge, threshold from the prior year;
113,414 out-of-sample events):

| Strategy | CAGR | Sharpe | Max DD | Trades |
|---|---:|---:|---:|---:|
| `wf_model_top5` | 10.6% | 0.47 | −59.4% | 486 |
| `wf_model_crash_only` | 11.0% | 0.50 | −56.8% | 293 |
| `wf_all_events_unconditional` | 9.7% | 0.43 | −52.8% | 1,110 |
| SPY buy & hold | 14.1% | 0.79 | −33.7% | |
| HW1 momentum rule | 9.8% | 0.53 | −39.2% | |

**What survives:**

* **The model has real, but small, stock-picking skill.** Test AUC is 0.546 and walk-forward AUC 0.556.
  Precision at the threshold is **50.5% against a 39.2% base rate**, and calibration is monotonic: the
  bottom probability decile hits 33.3%, the top decile 46.4%. On the *same* events, buying everything
  loses 8.5% a year on test, while the model's book makes +8.4%.
* **It does not beat buy-and-hold SPY**, neither on the 2024+ test nor in the 2018–2026 walk-forward.
  The bootstrap 95% CI of `model_top5`'s mean trade return is −1.8% … +4.4%, which includes zero.
* **The impressive first-version numbers were mostly survivorship bias.** The universe was *today's*
  S&P 1500, so a stock that crashed and later dropped out of the index was missing, and a stock that later
  *joined* the index was included before it qualified. With the point-in-time universe (R1) and the
  audit fixes, buy-everything falls from +40.6% to −8.5% CAGR on test. The table in §5 quantifies it.
* **Where the edge is:** crash events (HW5 Q3) and `earnings_down`, larger caps, and stressed or elevated
  regimes. There is no edge on `earnings_up`, and almost none on breakouts (§8).
* **Strategy-guide causal test:** the rolling-window *release* factor is **discarded**. Arm C beats Arm B on
  Sharpe and on the all-years and test means, but the sign flips across release thresholds, which fails
  the guide's stability rule.

The daily results site (GitHub Pages, produced by CI) shows the latest signals; see §9.

## 3. Data sources

| Source | What | Use | New vs lectures |
|---|---|---|---|
| **Yahoo Finance** (`yfinance`) | daily OHLCV + dividends + splits for 1,678 stocks (1,502 current + 176 recovered former members); SPY, IWM, MDY, ^VIX, ^VIX3M, 11 sector ETFs | prices, targets, technicals, benchmarks | lecture source |
| **Yahoo Finance** earnings dates | analyst EPS estimate, reported EPS, surprise % (HW1 Q4) | consensus surprise for earnings events | new use |
| **FRED** (fredgraph CSV) | DGS10, DGS5, DGS2, DFF, BAA10Y, T10Y2Y | rates, curve, credit-spread regime flag, HW3 macro rule | lecture source |
| **SEC EDGAR** submissions API | every filing: form, **8-K item codes**, acceptance time (UTC → ET) | minute-accurate earnings dates; crash-cause taxonomy; offering, late-filing, 13D and Form 4 activity | **new** |
| **SEC EDGAR** XBRL `companyconcept` | diluted EPS facts (first-reported) | point-in-time SUE | **new** |
| **SEC EDGAR** filing documents | EX-99.1 press releases / primary documents of 8-K, 424B, NT 10-K/Q filings around events | **R4 text layer**: guidance cut/raise, going concern, offering priced, investigations, clinical failures… + Loughran–McDonald tone | **new** |
| **Wikimedia** pageviews API | daily views of each company's Wikipedia article | abnormal public attention | **new** |
| **Wikipedia** S&P 500/400/600 lists + *historical components* | constituents, GICS sector, article title, **add/remove history** (HW1 Q1) | point-in-time universe (R1), size bucket | **new** (scraped) |

Sizes (2014-01 → 2026-11): 4,837,123 price rows, 2,021,569 SEC filings, 4,544,238 pageview rows,
370,038 XBRL EPS facts, 25,302 consensus EPS records, 26,091 classified filing texts and 2,982 membership
intervals. That is **11.9 million raw records**. The derived daily panel has 4,835,791 ticker-days, and the event dataset
has 149,055 rows × 269 model inputs. No data is committed to git (§9).

## 4. Features (269 model inputs)

All features at row *t* use information available by the close of *t*. The only exception is filings
accepted before 09:00 ET on *t+1*, since the trade enters at that open. FRED and pageviews are lagged one row.
`tests/test_audit_leakage.py::test_build_all_has_no_look_ahead` replaces all data after a cut date with
random numbers and asserts that no earlier feature changes.

* **Event reaction:** `abn_ret_1d` (beta-adjusted), `ret_vs_sector`, `gap_open`, `range_pct`, `close_loc`,
  `ret_open_close`, `volume_mult`, `volume_z`.
* **Momentum (HW1):** `growth_{3,5,10,20,60,120,252}d`, `mom_12_1`, `close_sma{20,50,200}`, `sma50_sma200`,
  `rs_vs_spy_60d`, `dist_52w_high/low`, streaks, and **`momentum_blend`** (the HW1 score).
* **Technicals** (TA-Lib formulas in pandas): RSI-14, MACD histogram, Bollinger %b, Stochastic %K, CCI-20,
  MFI-14, ADX-14, OBV slope, ATR %.
* **Risk and liquidity:** `vol_20d`, `vol_60d`, `vol_ratio_20_60`, `beta_252`, `idio_vol_60`, `log_dollar_vol`.
* **Earnings history:** `prev_ear`, `prev_pead_20d` (used only once fully realised), `beats_last4`,
  `days_since_prev_event`, `n_crash_prior_252`.
* **Analyst consensus (HW1 Q4):** `eps_surprise_pct`, `eps_beat`, `eps_est_available`,
  `eps_release_age_sessions`. A release counts only if stamped by the event day's 16:00 ET close; 91% of
  earnings events are covered.
* **SEC EDGAR items:**
  - SUE from first-reported diluted EPS;
  - 14 **8-K item flags** in the event window (1.01 agreement … 2.02 results, 2.06 impairment,
    3.01 delisting notice, 4.02 restatement, 5.02 officer change, 7.01, 8.01);
  - form flags (offering, late filing, periodic report) and prior-activity counts (offerings 365d,
    8-Ks 90d, Form 4 insider filings 30d, 13D stakes 180d, late filings 365d);
  - **`edgar_category`**, a priority taxonomy in which the most damaging cause wins.
* **SEC filing text (R4):**
  - 16 lexicon flags `ft_guidance_cut / raise / reaffirm`, `ft_going_concern`, `ft_offering_priced`,
    `ft_restatement`, `ft_material_weakness`, `ft_investigation`, `ft_clinical_failure`,
    `ft_exec_departure`, `ft_covenant`, `ft_impairment`, `ft_layoffs`, `ft_bankruptcy`, `ft_beat`, `ft_miss`;
  - Loughran–McDonald negative / positive / uncertainty / net tone, and the offering discount when stated;
  - **`edgar_category_text`**, which splits `earnings` into guidance cut / raise / mixed / other, and marks
    going-concern, clinical-failure, offering-priced and investigation events.
* **Attention:** `att_ratio_7_90`, `att_spike_1d`, `att_level`.
* **Rolling-window state (strategy guide):** `rw_dev_*` = D(t,L), `rw_max_dev_*` = M(t,W), flat-path
  `rw_release_{10,20}_{1,3,5}` = M(t,W) − max(0, M(t,W−j)), prior strength, consolidation and breakout
  state, arm flags.
* **Market and regime** (fragility monitor from my TradePulse project, plus the HW1 Q3 correction state):
  SPY/IWM returns and drawdown, VIX and **VIX/VIX3M**, sector-ETF correlation, FRED series, breadth,
  crash breadth, `in_correction` / `correction_depth` / `days_in_correction`, and regime points and state.
* **Which, not when:** `cs_rank_*` / `cs_dm_*` are per-date percentile ranks and date-demeaned copies of
  32 stock-level features. The production feature set **drops the market-level inputs** (§6).
* **Dummies** (8 categoricals + 7 engineered bins): `event_family`, `edgar_category`,
  `edgar_category_text`, `sector`, `cap_bucket` (point-in-time), `month`, `weekday`, `regime_state`, and the
  binned reaction, RSI, SUE, 52-week, VIX, attention and momentum.

## 5. Transformations and EDA

`features.build_all` builds **one unified event table** from all sources (notebook `01`). `dataset.prepare`
defines the course feature sets (`TO_PREDICT`, `NUMERIC`, `CATEGORICAL` → `DUMMIES`, `TO_DROP`) and the
**purged** temporal split: train ≤ 2021 (79,544), validation 2022–23 (25,616), test 2024 → (43,015).
880 rows whose 10-day label crosses a split boundary are purged.

**Survivorship bias, quantified** (R1, `survivorship.parquet`, notebook `05`). These are all events with a
known target.

| Universe | Events | Base rate | Mean 10d | `shock_crash` mean 10d |
|---|---:|---:|---:|---:|
| Today's constituents, any date (first version) | 173,705 | 37.6% | +0.42% | **+1.68%** |
| **Point-in-time: in the index on the event date** | 147,973 | 37.2% | +0.31% | **+1.06%** |
| ↳ names that later left the index | 9,927 | 36.5% | **−0.28%** | +0.10% |
| Today's names *before* they joined (now excluded) | 35,659 | 39.0% | +0.65% | **+2.68%** |

Most of the first version's crash bounce came from stocks that were *about to be added* to an index. That is
a look-ahead in the universe itself.

**Event study by family** (point-in-time, all years):

| Family | Events | Base rate | Mean 10d | Mean vs SPY |
|---|---:|---:|---:|---:|
| `shock_crash` | 2,849 | **44.7%** | +1.06% | **+0.82%**: reversal |
| `earnings_crash` | 3,743 | 41.4% | +0.38% | −0.17% |
| `earnings_up` | 31,670 | 38.2% | +0.48% | −0.13% |
| `earnings_down` | 26,823 | 37.6% | +0.51% | −0.12% |
| `momentum_breakout` | 82,888 | 36.1% | +0.16% | −0.17% |

* **Reversal after non-earnings crashes is real; unconditional PEAD is not** in this universe and period.
* **What the SEC text says** (R4, `filing_text_study.parquet`, crash events, all years):
  - Crashes with an **offering filing** mean −3.0% over 10 days (95% CI −5.7% … −0.5%, n=96, hit rate 31%):
    permanent damage, as HW5 Q3 predicted.
  - Crashes whose press release contains **guidance-cut language** mean **+2.8%** (+1.95% vs SPY,
    CI +1.1% … +4.4%, n=171): the market over-reacts to guidance cuts.
  - For post-earnings drift (2022+), guidance cut vs raise makes no reliable difference.
* Notebook `01` also covers the target distribution, the event study at +5/+10/+20 days, crash base rates
  by EDGAR cause, PEAD by SUE and momentum quintile, attention and regime, and feature and dummy correlations.

## 6. Modeling

* **Hand rules:**
  - HW1 PEAD + momentum;
  - HW5 panic-no-damage;
  - HW3 macro rule `DGS10 ≤ 4.5 & DGS5 ≤ 4`;
  - buy all events;
  - guide arms A/B/C.
* **Lecture models:** DecisionTree and RandomForest. **New models:** XGBoost, LightGBM, LogisticRegression,
  an RF + XGB + LGBM ensemble, and LightGBM LambdaRank.
* **Tuning:** each grid is fit on TRAIN and scored on VALIDATION by precision in the top decile, with AUC
  as tie-breaker (`tuning.parquet`).
* **Which, not when** (`model_comparison.parquet`, VALIDATION decides):
  - The first model's top features were market-level (rates, credit spread, VIX term structure), so it was
    timing the market. Variants tested: pooled; pooled + HW3 week-of-month dummies; cross-sectional
    features keeping or **dropping** market inputs; per-family models; LambdaRank by date.
  - Dropping the market inputs gives the most stable result: validation AUC 0.555 and test AUC 0.545 for
    RF/LGBM, against 0.547 / 0.532 for pooled.
  - The pooled + week-of-month RF scores marginally higher on validation (0.558) but generalises worse
    (test 0.535).
  - Per-family models and LambdaRank did not help.
* **Production model:** the ensemble on `cs_drop_market` (best validation AUC 0.557; test AUC 0.546).
* **Decision rules:**
  - the threshold is the highest-precision cut on VALIDATION subject to **≤ 5 trades per active day**
    (HW4 capacity rule), giving **0.52**;
  - `pred9_ensemble_top1_daily` takes one event per day;
  - book composition, decided on VALIDATION, excludes `momentum_breakout` and EDGAR `offering` events
    (`composition.parquet`).

| Prediction (TEST, base rate 39.2%) | Trades | Precision | Avg 10-day growth |
|---|---:|---:|---:|
| `pred4_logreg_best_rule_69` | 26 | 73.1% | +8.1% |
| `pred3_lgbm_best_rule_55` | 160 | 54.4% | +3.3% |
| `pred5_rf_best_rule_51` | 132 | 53.0% | +3.3% |
| `pred_book_ensemble_rule_52` (production book) | 158 | 51.3% | +2.5% |
| `pred8_ensemble_best_rule_52` | 190 | 50.5% | +2.4% |
| `pred_manual_dgs10_dgs5` (HW3 rule) | 19,496 | 42.1% | +1.3% |
| `pred1_manual_panic_no_damage` (HW5 rule) | 1,587 | 42.0% | +0.6% |
| `pred2_manual_all_events` | 41,933 | 39.2% | +0.5% |
| `pred0_manual_pead_momentum` (HW1 rule) | 3,466 | 38.4% | +0.2% |

Library versions are logged in `metrics.json`. A `check_sklearn_version()` guard (HW4 lesson) warns on
scikit-learn ≠ 1.7.x.

## 7. Trading simulation

* **Vector simulations** follow the course's `sim1_` ($100 per positive prediction, capital = $100 × 10 ×
  Q75 of daily bets) and `sim2_` (stop-loss / take-profit from each event's MAE/MFE) for every `pred*`.
* **Exact simulations** (`simulate.exact_sim`) are a daily loop with a cash account and **reinvestment**:
  - entry at the next open; stops, targets and time exits on real OHLC paths (a gap through the stop fills
    at the open);
  - max concurrent positions; probability-weighted sizing; **regime pause**; a long-short book;
  - **cap-aware costs**;
  - the **guide's R framework**: stop at 2×ATR, 3R target, 8-session time stop, 0.5% risk per trade,
    ≤ 10% notional, ≤ 2% open risk, gap-up skip, and no new entries after 3 stop-outs in a day.
* **13 strategies vs 3 benchmarks** (SPY, IWM and the point-in-time HW1 momentum rule), with CAGR, Sharpe,
  Sortino (downside deviation), max drawdown, Calmar, rolling 1-year returns, win rate and average trade.
* **Report card** per strategy: fill rate, profit factor, average R, top-5 profit share, beta vs SPY,
  bootstrap CI and turnover.
* **Cost stress:** `model_top5` makes 8.4% CAGR at 1× costs, 5.5% at 2× and 2.7% at 3×.
* **Walk-forward** (R2): yearly expanding refits 2018–2026 with stitched out-of-sample equity (§2,
  `walkforward.parquet`).
* **Pre-declared headline:** the strategy shown on the results site is fixed in advance as `model_top5`.
  It is not picked as the best on TEST.

## 8. Deep analysis, audit and what to improve

**Where the model works** (TEST, production rule, `precision_breakdown.parquet`):

| Slice | Base rate | Precision of selected | Lift |
|---|---:|---:|---:|
| `earnings_down` | 39.2% | 65.6% (n=32) | **+26.4 pp** |
| `shock_crash` | 44.5% | 55.4% (n=56) | +10.8 pp |
| `momentum_breakout` | 38.6% | 48.3% (n=29) | +9.6 pp |
| `earnings_up` | 39.9% | 37.8% (n=45) | −2.1 pp |
| large / mid / small caps | 38.8 / 39.0 / 39.8% | 57.5 / 51.0 / 47.5% | +18.7 / +12.0 / +7.7 pp |
| 2024 / 2025 / 2026 | 38.5 / 38.7 / 40.5% | 33.3 / 58.1 / 48.1% | −5.2 / +19.5 / +7.5 pp |
| regime normal / elevated / stressed | 38.3 / 48.9 / 55.3% | 42.6 / 71.4 / 78.6% | +4.2 / +22.5 / +23.3 pp |

**Independent correctness audit.** A separate reviewer ran the audit; every fix comes with a failing test
first (`tests/test_audit_*.py`).
* **Verified correct:**
  - no look-ahead in the feature build (randomised-future test);
  - cross-sectional features use same-date rows only;
  - walk-forward never fits on the year it predicts;
  - thresholds, models, feature set and book rules are chosen on VALIDATION only;
  - simulator accounting is correct: next-open entry, gap fills, cash never negative, two-sided costs;
  - API auth is fail-closed with a constant-time compare, and orders need two switches.
* **Fixed:**
  1. purge between splits (labels crossed into validation);
  2. `cap_bucket` was today's index, so 16.9% of events had the wrong size bucket and cost; it is now
     point-in-time;
  3. the HW1 benchmark could hold not-yet-members;
  4. the headline strategy was chosen on TEST;
  5. the Sortino definition.
* Also fixed after the audit:
  - NaN in `metrics.json` broke `/v1/backtest/summary` (HTTP 500);
  - broker brackets are GTC instead of DAY, so stops survive the close;
  - duplicate same-day orders are blocked;
  - expiry uses the latest signal per ticker;
  - the public `/health` no longer echoes error text.

**Debugging insights and ideas:**
1. **The model is partly a volatility selector.** Its top feature is the ATR-based stop distance, followed by
   SUE age and distance from the 52-week high. The +2% target favours volatile names. *Next:* a
   volatility-only baseline, and a volatility-normalised target (e.g. return / ATR).
2. **2024 was a bad year** (precision 33%) and 2025 a good one (58%). Regime dependence is large. *Next:*
   a regime-conditional threshold or position size, decided on validation.
3. **Earnings-up events carry no edge**, while earnings-down and crash events do, which supports the
   reversal side of the thesis. *Next:* drop `earnings_up` from the book (validate first).
4. **Text helps classify crashes** (offering means damage, guidance cut means panic) but cannot be learned
   for earnings before 2022 (no text). *Next:* extend the text fetch to 2014–2021 earnings.
5. **Concentration and capacity:** with ~0.4% of events selected, results rest on a few hundred trades. A
   ranking-quality gain matters more than a threshold tweak.

## 9. Automation, CI/CD and data retention

```bash
python -m driftrev.pipeline --mode update                 # daily: incremental download → signals → report → notify
python -m driftrev.pipeline --mode update --retrain       # + refit models and all backtests + walk-forward
python -m driftrev.pipeline --mode update --retrain --tune   # + grid search
python -m driftrev.pipeline --mode file --retrain         # reproduce from the snapshot, no downloads
python -m driftrev.pipeline snapshot                      # export data/driftrev_snapshot.zip
python -m driftrev.housekeeping [--dry-run]               # delete expired data files
python -m driftrev.filing_text ingest --max-seconds 1800  # continue the resumable SEC text fetch
```

* **Two regimes.** `update` downloads from the APIs. `file` reads the parquet snapshot (`events`, `panel`,
  `market`, `universe`, `ctx`) from `data/snapshot/`, or fetches it from `DRIFTREV_SNAPSHOT_URL`. CI
  publishes it weekly as the Release asset `data-snapshot`.
* **Incremental loading into DuckDB** (`data/warehouse.duckdb`, git-ignored):
  - every loader fetches only the gap (prices: 7-day overlap; a split triggers a full re-download);
  - FRED re-pulls the last 60 days;
  - EDGAR re-reads only the recent block per company;
  - pageviews and consensus EPS are fetched from each series' last date;
  - SEC filing text resumes with a per-run time budget;
  - membership is refreshed weekly.
* **Cron:** `scripts/run_daily.sh` + `scripts/crontab.example` (22:30 New York, Mon–Fri), or the `scheduler`
  container. Every run is recorded in `run_log.jsonl`.
* **One notebook runs the whole workflow:** `notebooks/00_end_to_end_workflow.ipynb` calls the `.py`
  functions in order (download → transform → predict → simulate → latest new trades → agent view).

**CI/CD on GitHub Actions** (OPM `SD7_cicd.dot`):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR on `project/**` | ruff lint, pytest (236 tests: API, core invariants, audit, retention, CI contract, dashboard render), every OPM diagram renders and has an OPL section, notebooks valid, `docker compose config`, `docker build` + import check |
| `daily.yml` | 22:30 ET Mon–Fri; Sunday retrain; manual | restore the warehouse from the Actions cache → `pipeline --mode update` → job summary, **GitHub Pages results site** (`index.html` with an SVG equity chart, `signals.json`, `metrics.json`), run artifact (30-day expiry), save cache, optional Telegram/webhook push; Sunday: snapshot Release asset |
| `image.yml` | push on `project/**`; manual | build and push `ghcr.io/<owner>/driftrev` (`latest`, `sha-*`), then pull and check `/health` |
| `cleanup.yml` | weekly; manual (dry-run) | delete Actions caches > 3 days (keeping the newest), artifacts and workflow runs > 30 days, and old snapshot assets |

**Data retention** (`config.Retention`, after every pipeline run):
- EDGAR root JSONs older than 2 days and history pages older than 180 days are deleted;
- **raw filing text** is deleted after 30 days (its flags and tone stay in the warehouse);
- the signal archive is kept for 60 days;
- only the newest snapshot zip is kept;
- the run log is capped at 1,000 lines and logs at 5 MB.

`tests/test_cicd.py` keeps the workflow retention numbers in sync with `config.py`.

**One-time GitHub setup:** Settings → Pages → Source = *GitHub Actions*. Add the repository secret
`SEC_USER_AGENT` = `"Your Name your@email"`; `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` /
`DRIFTREV_WEBHOOK_URL` are optional. The first run with an empty cache bootstraps the full history
(1–2 hours).

## 10. Agent interface, notifications, broker and dashboard

* **REST API** (`driftrev.api`, FastAPI, bearer token, OpenAPI with clean operation IDs):
  - `/health`;
  - `/v1/signals/latest`, `/v1/events/recent`, `/v1/tickers/{t}`;
  - `/v1/backtest/{summary,trades,equity}`, `/v1/model`, `/v1/regime`;
  - `/v1/research?table=…` (model comparison, walk-forward, survivorship, filing-text study, holding
    period, uniquely-correct, guide arms…), `/v1/runs`;
  - `POST /v1/pipeline/run` → `/v1/jobs/{id}`;
  - `POST /v1/orders/preview`, `GET /v1/portfolio`;
  - `/v1/agent/tools`.
* **Function calling:** `agent/tools.json` (14 tools, OpenAI/xAI format) can be loaded directly by a Grok
  bot. `agent/grok_agent_example.py` is a working tool loop. The **MCP server**
  (`python -m driftrev.mcp_server --transport stdio|streamable-http`) exposes the same tools. See
  [`agent/README.md`](agent/README.md).
* **Notifications (R3):** after each run, `driftrev.notify` pushes the day's BUY/STRONG_BUY signals with
  stop, target and exit date to Telegram and/or a JSON webhook. It does nothing when not configured.
* **Broker (Alpaca, paper by default):**
  - `preview_orders` sizes GTC bracket orders;
  - `submit_orders` refuses unless `confirm=true` **and** `DRIFTREV_ALLOW_ORDERS=true`, skips held tickers
    and open orders, and is not an agent tool;
  - `close_expired_positions` exits after the latest signal's `exit_by_date`.
* **Dashboard** (`streamlit run dashboard/app.py`): overview vs SPY, signals, equity curves, trades, model,
  **research** (walk-forward, model variants, composition, capacity, costs, holding period, filing text,
  survivorship), the rolling-window guide, and pipeline runs. It is rendered headlessly in the tests.
* **Containers:** `docker compose up -d` starts `api` (:8000), `mcp` (:8001), `dashboard` (:8501) and
  `scheduler` on a shared `/data` volume. Deployment templates for Cloud Run, Fly.io and Render are in
  [`deploy/`](deploy/README.md); they are not deployed.

## 11. How to reproduce

```bash
git clone https://github.com/witold-andelie/stock_analysis_wentao.git
cd stock_analysis_wentao/project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt     # pinned: scikit-learn 1.7.2, numpy 1.26.4 …; dev = ruff
cp .env.example .env                                        # set SEC_USER_AGENT="Your Name your@email"
export PYTHONPATH=src

# quick smoke run on 50 tickers (~2–3 min)
DRIFTREV_MAX_TICKERS=50 python -m driftrev.pipeline --mode update --retrain --tune

# full history: ~30 min download + ~10 min transform + ~25 min tuning/backtests on 8 cores
python -m driftrev.pipeline --mode update --retrain --tune

# or without any download, from the CI snapshot:
DRIFTREV_SNAPSHOT_URL=https://github.com/witold-andelie/stock_analysis_wentao/releases/download/data-snapshot/driftrev_snapshot.zip \
  python -m driftrev.pipeline --mode file --retrain

python -m pytest tests -q && ruff check .                   # 236 tests, lint
jupyter notebook notebooks/                                 # 00 … 06, executed with outputs (06 needs data/de)

# Docker: api :8000, mcp :8001, dashboard :8501, scheduler. Set DRIFTREV_API_TOKEN in .env first.
docker compose up -d --build
curl localhost:8000/health        # "no_data" until the pipeline has run once inside the volume:
docker compose exec scheduler python -m driftrev.pipeline --mode update --retrain
docker compose down
```

A peer-review style reproducibility audit (fresh clone, every step timed) is in
[`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md).

## 12. Architecture (OPM) and repository layout

Development is **model-first** ([`AGENTS.md`](AGENTS.md)). Every feature is first added to the OPM model
(ISO 19450, Graphviz DOT + the OPL text twin), then implemented together with a test.

| Diagram | Scope |
|---|---|
| [`SD0_system`](docs/architecture/opm/svg/SD0_system.svg) | system function *Signalling Event Trades*, environment, top-level processes |
| [`SD1_acquiring`](docs/architecture/opm/svg/SD1_acquiring.svg) | incremental loading incl. historical membership and consensus EPS |
| [`SD2_transforming`](docs/architecture/opm/svg/SD2_transforming.svg) | features, point-in-time universe, event detection, EDGAR taxonomy, SUE, attention, regime |
| [`SD2.1_classifying_filing_text`](docs/architecture/opm/svg/SD2.1_classifying_filing_text.svg) | R4: exhibit selection, lexicon flags, LM tone, refined category |
| [`SD3_researching`](docs/architecture/opm/svg/SD3_researching.svg) | purged split, model variants, book composition, tuning, thresholds, simulations, walk-forward, guide arms |
| [`SD4_signalling`](docs/architecture/opm/svg/SD4_signalling.svg) | scoring, cap-aware net edge, tiering with regime downgrade, sizing and exits, publishing |
| [`SD5_serving_trading`](docs/architecture/opm/svg/SD5_serving_trading.svg) | REST / tools.json / MCP, auth, research tables, jobs, notifications, dashboard, order guard |
| [`SD6_roadmap`](docs/architecture/opm/svg/SD6_roadmap.svg) | R1–R5 implemented; R6 (intraday entry) planned; R7 (German equities on Xetra) largely implemented |
| [`SD7_cicd`](docs/architecture/opm/svg/SD7_cicd.svg) | CI, daily CD, Pages, GHCR image, snapshot release, housekeeping, GitHub garbage collection |

```
project/
  src/driftrev/   config · store · universe · membership · prices · macro · edgar · attention · consensus · filing_text   (ingestion)
                  indicators · rolling_window · features · dataset · research                                          (transformation)
                  models · walkforward · simulate · analysis                                                           (research)
                  signals · pipeline · artifacts · report · housekeeping · notify                                      (signalling + ops)
                  api · service · security · mcp_server · agent_tools · broker · scheduler                             (serving / execution)
  notebooks/      00 end-to-end · 01 data & EDA · 02 modeling · 03 simulation & analysis · 04 rolling-window guide · 05 research improvements · 06 German extension
  dashboard/  agent/  deploy/  docs/architecture/opm/  docs/REPRODUCIBILITY_AUDIT.md  scripts/  tests/ (236)
  Dockerfile · docker-compose.yml · requirements{,-runtime,-dev}.txt · ruff.toml · .env.example
../.github/workflows/   ci.yml · daily.yml · image.yml · cleanup.yml
```

| Extension module | What it does |
|---|---|
| [`de_membership.py`](src/driftrev/de_membership.py) | reconstructs German index membership from Wikipedia changes, redirects and evidenced aliases |
| [`de_universe.py`](src/driftrev/de_universe.py) | resolves company names to Yahoo symbols and loads Xetra or regional prices |
| [`de_market.py`](src/driftrev/de_market.py) | loads German index levels and Bundesbank Bund yields |
| [`de_context.py`](src/driftrev/de_context.py) | builds German market context for the shared feature and regime code |
| [`de_pipeline.py`](src/driftrev/de_pipeline.py) | ingests and transforms German data, then runs the shared research pipeline |
| [`research_pooled.py`](src/driftrev/research_pooled.py) | compares German-only, pooled and US-transfer models on German events |
| [`broker_ibkr.py`](src/driftrev/broker_ibkr.py) | bridges German signals to Interactive Brokers with paper and order guards |

## 13. Limitations

* **Survivorship is reduced, not removed.**
  - 176 of 737 names that left the indices after 2014 were recovered; 66% are gone from Yahoo.
  - S&P 600 history starts in 2019-12, so earlier small-cap membership is assumed.
  - Renamed tickers are dropped.
* **Selection on validation.** Hyper-parameters, feature set, book rules and the capacity cap were all
  chosen on 2022–23. Walk-forward years 2018–2023 are therefore not fully hyper-parameter-out-of-sample;
  only 2024 onward is clean.
* **Filing text for earnings** starts in 2022. Train sees no earnings text (`ft_text_available` flags this),
  which is a distribution shift, not a leak.
* **Sector** is today's GICS classification, which is a mild leak.
* **Live vs backtest timing:** training windows include filings accepted before 09:00 ET on t+1. A live run
  at 22:30 ET on t may not have them yet.
* **Execution:** daily bars only, with no partial fills or borrow costs. Strategy P&L is price-only, while
  the SPY benchmark includes dividends (conservative for the strategy).
* **Small samples:** about 100–150 trades per test strategy, and bootstrap CIs of the mean trade include zero.
* **German extension** ([§15](#15-extension-german-equities-on-xetra)):
  - SDAX has no full change-history table. Only 38 of ~50 members have usable price history for 2014. The
    reconstructed SDAX count is also overstated early on (58 members on 2014-12-31 vs 50 in the index): a name whose
    first SDAX event is a move up to MDAX is assumed to have been a member since inception (47 of those 58).
  - Linde is missing from the DAX history (29 instead of 30 members, 2014–2021): Linde AG and Linde plc are not
    matched as one company, and no evidenced alias exists yet.
    The 62 SDAX orphans have unknown exit dates; their open SDAX intervals are dropped (not closed at a guessed date).
  - 21 delisted companies have no Yahoo prices on any German venue, leaving survivorship bias.
  - Yahoo earnings timestamps appear one session late for ~9% of releases. Correcting them from prices
    would introduce look-ahead.
  - There is no ad-hoc notice source, so German crashes have no filing-based cause taxonomy.
  - The IBKR bridge is tested offline only. It has not been verified in a paper account.
* Research and education only, not investment advice. Orders are paper-only and disabled by default.

## 14. Evaluation-criteria map

| Criterion | Where |
|---|---|
| Problem description, context, new problem definition, step-by-step findings | §1, §2, §5–§8; notebooks 01–05; German extension in notebook 06 (§15) |
| Lecture data + 20+ new features with descriptions | §3, §4 (YF/FRED + ~250 new features) |
| New data source (not YF/FRED) | SEC EDGAR filings, XBRL and filing text; Wikimedia pageviews; Wikipedia constituents and history; Bundesbank API, German Wikipedia + redirect API (§15, notebook 06); Yahoo search is an additional Yahoo use |
| Dataset > 1M records | 11.9M raw records; 4.84M-row daily panel |
| Combined dataframe, feature sets, ≥ 5 new transformations, EDA | `features.build_all`, `dataset.py`, notebook 01, §5; shared multi-market feature pipeline (§15, notebook 06) |
| Lecture models, several models, custom decision rules, tuning, new models | §6, `models.py`, `research.py`, notebook 02; German-only and pooled US + DE comparison (§15, notebook 06) |
| Vector sims, ≥ 2 strategies, exact sims with reinvestment | §7, `simulate.py`, notebook 03 |
| Profitability vs benchmark (CAGR, Sharpe, DD, rolling returns) | §2, notebook 03; walk-forward in notebook 05; German research vs DAX (§15, notebook 06) |
| Advanced risk management, new strategies | §7 (regime pause, R framework, max positions, long-short, capacity rule) |
| Exceptional profitability vs SPY | **not claimed**: after removing survivorship bias the strategies do not beat SPY (§2); the German strategy does not beat the DAX (§15) |
| Deep analysis by family / size / year / regime / cause, debug wrong predictions, ideas | §8, notebooks 03 and 05; German events and earnings-date alignment in notebook 06 (§15) |
| Notebooks exported to scripts + one end-to-end notebook | `src/driftrev/*`, notebook 00 |
| Dependencies managed | `requirements*.txt`, `.venv`, Dockerfile |
| Cron re-run for the latest data | `run_daily.sh`, `crontab.example`, scheduler container, `daily.yml`; market-aware scheduler for Germany, no German daily CD job yet (§15) |
| Two regimes (file vs download) | US: `--mode file` / `--mode update`; DE: `--mode warehouse` / `--mode update`, no `file` mode (§15) |
| Incremental loading with storage outside GitHub | DuckDB warehouse, Actions cache, Release snapshot; separate German data directory (§15) |
| Bonus: code design | OPM model, module boundaries, shared multi-market pipeline, 236 tests, ruff, CI, independent audit |
| Bonus: broker API | `broker.py` (Alpaca paper, GTC brackets, order guard); `broker_ibkr.py` (IBKR paper by default, offline-tested only; §15) |
| Bonus: monitoring | dashboard, GitHub Pages site, Telegram/webhook notifications, REST/MCP agent interface |
| Bonus: containerization | Dockerfile + docker-compose (4 services), GHCR image |
| Bonus: cloud deployment | GitHub Actions daily CD + Pages; Cloud Run / Fly / Render templates |

## 15. Extension: German equities on Xetra

**Why and scope.** This extension tests the same drift-or-reversal question on German large, mid and
small caps: DAX 40 + MDAX 50 + SDAX 70 on Deutsche Börse Xetra, traded in EUR.
Select it with `DRIFTREV_MARKET=de`. It uses `DATA_ROOT/de`; the US default keeps `DATA_ROOT` and its
behaviour is unchanged. China A-shares are out of scope. The source and research record is
[OPL R7](docs/architecture/opm/OPL.md#r7--german-equities-on-deutsche-börse-xetra).
Notebook [06](notebooks/06_german_extension.ipynb) reads the German tables and research artifacts for an end-to-end review.

**Data and point-in-time universe.** German Wikipedia supplies the DAX, MDAX and SDAX current tables
and the DAX and MDAX change tables, including `MDAX/Zusammensetzungshistorie`.
The parser expands rowspan/colspan grids. It infers SDAX moves from MDAX history rows marked `SDAX`.
The Wikipedia redirect API resolves article titles. Company identity joins these titles and normalised
names, with evidenced aliases in `resources/de_aliases.csv`. Current-table dates seed missing open
membership intervals.
The 2026-11-22 fixtures contain 625 intervals, with exactly 40 / 50 / 70 members on 2026-11-20.

Cached Yahoo search and evidenced symbol overrides resolve names, with Xetra preferred over Hamburg.
The 2026-11-22 run covers 201 companies since 2014: 180 resolved, including all 160 current members on
Xetra. There are 21 delisted names without Yahoo prices on any German venue. Some other delisted names
have Hamburg prices. Their liquidity is based on index membership because regional volume is not
representative. They stay in event history; the German signal book contains only tradable `.DE` symbols.

Yahoo supplies `.DE` prices, index levels (`^GDAXI`, `^MDAXI`, `^SDAXI`), earnings dates and estimated
and reported EPS. The Bundesbank time-series API supplies the daily 10-year Bund yield.
Yahoo does not supply VDAX-NEW or VSTOXX, so the volatility gauge uses realised DAX volatility.

**Shared pipeline, market-aware settings.** German data uses the same `features.build_all`, models,
simulations, signal publishing, reports and notifications. Settings follow the active market.
TEST is reported only and never used to choose models or book rules.

| Setting | US | Germany |
|---|---|---|
| Primary / small-cap benchmarks | SPY / IWM | DAX / SDAX; MDAX levels are also loaded |
| Volatility gauge | VIX and VIX/VIX3M | annualised 20-session DAX realised volatility; 20/60-session ratio |
| One-way costs, large / mid / small | US cap-aware costs (§1) | 10 / 20 / 35 bps (assumed) |
| Event liquidity floor, average turnover | USD 2m | EUR 1m |
| Breakout ranking floor, median turnover | USD 10m | EUR 5m (assumed) |
| Local close | 16:00 New York | 17:30 Berlin |
| Scheduled run | 22:30 New York | 18:30 Berlin |
| Book rules | US VALIDATION decisions | no family or category exclusions; German VALIDATION keeps all families |
| Broker | Alpaca | Interactive Brokers, IBIS exchange, EUR |

The scheduler and daily pipeline support Germany. The GitHub Actions workflow `daily_de.yml` runs it after the
Xetra close (17:30 UTC, Monday to Friday; Sunday retrain). The results site shows the US at the root and Germany under `/de/`.

**Results (honest).** These runs were recorded on 2026-11-22. AUCs below are measured on German events.
The price-event walk-forward uses yearly refits from 2018; its OOS period is 2024-01-02 → 2026-11-20.
An em dash means the metric is not reported in R7 for that run.

| Research run | VALIDATION AUC | TEST AUC | Walk-forward OOS AUC | TEST model book CAGR | Walk-forward CAGR |
|---|---:|---:|---:|---:|---:|
| German-only price events, LightGBM (7,993 events) | 0.534 | 0.549 | 0.560 | −0.5% | 8.5% |
| Pooled US + DE, LightGBM (VALIDATION winner) | 0.555 | 0.544 | — | — | — |
| German events with earnings, RF (11,933 events) | 0.528 | 0.546 | 0.534 | — | −1.2% |
| DAX buy-and-hold, price-event OOS period | — | — | — | — | 14.5% |

* The German signal is weak. In the price-event run, the VALIDATION threshold of 0.63 leaves only
  11 TEST trades in the model books.
* The German walk-forward strategy does not beat the DAX. Adding earnings does not improve it.
* Pooling does not help beyond noise. The VALIDATION gain of +0.021 AUC is about 1.3 standard errors
  at n = 1,354 and does not carry through to TEST. German production stays German-only.
* No German post-earnings drift is visible with these features. The 10-day hit rates for `earnings_up`
  and `earnings_down` are similar: 37.5% vs 36.0%.

**Earnings-date alignment check.** Yahoo timestamps are converted from US Eastern time to Berlin.
A release at or after 17:30 maps to the next session. Median absolute return is 2.31% on the mapped
session, against 1.08–1.47% on the two sessions before and after. The mapped session has the largest
move in 42% of the five-session windows. Releases stamped 18:00–23:59 Berlin (424, ~9%) appear one
session late. This error is retained because shifting dates based on prices would introduce look-ahead.

**Trading.** `service.broker_for_market()` routes German signals to the Interactive Brokers bridge.
It requires TWS or IB Gateway. It does not connect unless `IBKR_ENABLED=true`. Paper mode is the default
and accepts only a `DU…` account. Submission needs both `confirm=true` and `DRIFTREV_ALLOW_ORDERS=true`.
The bracket has a DAY market BUY parent, intended for the next Xetra open when sent after the close,
with a GTC take-profit limit and a GTC stop. Order references identify the symbol and signal date.
Expiry closing sells long positions only and uses daily close-order references. It skips an already-open
close order and requests cancellation of bracket exits before the SELL. A cancellation exception blocks
the SELL; the code does not wait for cancellation acknowledgements.

The bridge is **offline-tested only**, using a fake IB. Before any real use, verify these in a paper account:

* IBIS contract qualification and permissions;
* after-close DAY parent handling;
* bracket acceptance after disconnect;
* GTC child activation and sibling cancellation;
* the cancellation-acknowledgement race before the expiry SELL.

**How to run.** From `project/`, use the environment from §11 with `PYTHONPATH=src`.
The first command downloads inputs and transforms them. Research reads the already-transformed warehouse.

```bash
# Download German inputs and build events, the daily panel and market context.
DRIFTREV_MARKET=de python -m driftrev.de_pipeline

# Rebuild from existing raw warehouse inputs, without downloading.
DRIFTREV_MARKET=de python -m driftrev.de_pipeline --transform-only

# Train and backtest the transformed warehouse; optionally rerun the tuning grid.
DRIFTREV_MARKET=de python -m driftrev.de_pipeline --research
DRIFTREV_MARKET=de python -m driftrev.de_pipeline --research --tune

# Daily shared pipeline: download and transform, then publish signals and reports.
DRIFTREV_MARKET=de python -m driftrev.pipeline --mode update

# Reuse transformed tables, then publish signals and reports without downloading.
DRIFTREV_MARKET=de python -m driftrev.pipeline --mode warehouse

# Compare explicit US and German warehouses; write a separate comparison parquet.
python -m driftrev.research_pooled \
  --us-warehouse data/warehouse.duckdb \
  --de-warehouse data/de/warehouse.duckdb \
  --out data/de/artifacts/pooled_comparison.parquet
```

The pooled command uses the default directory paths shown above; adjust them if `DATA_ROOT` differs.
It reads both warehouses without changing them. German `--mode file` is not supported.
