# OPL: Object-Process Language for the Drift-or-Reversal model

OPL is the text twin of the OPM diagrams in this folder (ISO/PAS 19450 notation drawn with
Graphviz, the same convention as TradePulse). **Every sentence below corresponds to one link in
a `.dot` file. If a diagram, this file and the code disagree, fix all three in the same commit.**

Render: `.venv/bin/python scripts/render_opm.py` → `svg/*.svg`. The legend is in `legend.dot`.

Owner tags (fill colour in the diagrams):
**[ING]** ingestion: `universe`, `prices`, `macro`, `edgar`, `attention`, `store` ·
**[TRF]** transformation: `features`, `indicators`, `rolling_window`, `dataset` ·
**[RES]** research: `models`, `simulate`, `analysis` ·
**[SIG]** signalling: `signals`, `pipeline`, `artifacts` ·
**[SRV]** serving: `api`, `mcp_server`, `service`, `security`, `agent_tools`, `dashboard/` ·
**[OPS]** execution & ops: `broker`, `scheduler`, `scripts/`, Docker.
Status tags: ✅ implemented · 🟡 implemented with a known limitation · ⬜ PLANNED.

---

## SD0 · System diagram (`SD0_system.dot`)

- **Drift-or-Reversal** exhibits **Signalling Event Trades** (the system's function): for every
  information shock in the S&P 1500 (earnings release, crash, momentum breakout), it states whether
  the stock should **drift** (continue) or **revert** over the next 10 sessions, net of costs. ✅
- **GitHub Actions + Pages** handles **Scheduling** through `daily.yml` (SD7). ✅
- **Market Calendar** handles **Scheduling**. **Scheduling** invokes **Acquiring** [`--mode update`]. [OPS] ✅
- **Acquiring** requires **Public Data Providers** and changes **Warehouse** from *stale* to *current*. [ING] ✅
- **Transforming** requires **Warehouse** (or **Snapshot** in `--mode file`) and yields **Event Dataset**
  (with **Daily Panel** and **Market Context**). It also yields **Snapshot** on `pipeline snapshot`. [TRF] ✅
- **Researcher** handles **Researching** [`--retrain`]. **Researching** requires **Event Dataset** and yields
  **Model Bundle** and **Backtest Report**. [RES] ✅
- **Signalling Event Trades** requires **Event Dataset** and **Model Bundle** and yields **Signal Book** and **Run Log**. [SIG] ✅
- **Personal 24/7 Agent** handles **Serving**. **Serving** requires **Agent Interface**, **Signal Book**,
  **Backtest Report** and **Run Log**. It invokes **Acquiring** [POST /v1/pipeline/run] and **Trading** [POST /v1/orders/*]. [SRV] ✅
- **Researcher** handles **Monitoring**, which requires **Signal Book** and **Backtest Report**. [SRV] ✅
- **Trading** requires **Signal Book** and **Alpaca Broker** and yields **Order Set**. [OPS] ✅ (paper only, never live by default)

## SD1 · Acquiring in-zoomed (`SD1_acquiring.dot`)

**Acquiring** zooms into **Refreshing Universe**, **Loading Macro**, **Loading Market Series**,
**Loading Prices**, **Loading Filings & EPS** and **Loading Attention**, in that sequence.
**Warehouse** consists of **universe**, **prices**, **market**, **macro**, **filings**, **eps_facts**,
**pageviews** and **edgar_fetch_log**.

- **Refreshing Universe** requires **Wikipedia S&P lists** and **SEC EDGAR** (ticker→CIK) and affects **universe**. ✅ 🟡 current constituents only (survivorship bias, see R1)
- **Loading Macro** requires **FRED** and affects **macro**; it re-pulls the last 60 days because FRED revises. ✅
- **Loading Market Series** requires **Yahoo Finance** and affects **market**. ✅
- **Loading Prices** requires **universe** and **Yahoo Finance** and affects **prices**. It downloads only from
  each ticker's last date minus 7 days. A split inside the new window triggers a full reload of that ticker.
  Total return comes from stored dividends, not Yahoo's *Adj Close*, which rewrites history. ✅
- **Loading Filings & EPS** requires **universe**, **SEC EDGAR**, **Contact User-Agent** and **Rate Limiters**,
  and affects **filings**, **eps_facts** and **edgar_fetch_log**. It skips CIKs refreshed less than 12 h ago,
  and older submission pages are cached forever. ✅
- **Loading Attention** requires **universe**, **Wikimedia pageviews API**, **Contact User-Agent** and
  **Rate Limiters**, and affects **pageviews**. ✅

## SD2 · Transforming in-zoomed (`SD2_transforming.dot`)

Invariant: a value on row *t* uses only information available by the close of *t*. The exceptions
are filings accepted up to 09:00 ET on *t+1* (the trade is entered at the *t+1* open). FRED and
pageviews are lagged one row. Targets use *t+1 … t+H* only.

- **Building Market Context** requires **market** and **macro** and yields **Market Context**. ✅
- **Ranking Prior Strength** requires **prices** and **Market Context**. It takes the cross-sectional
  percentile of D(t,20) among names with a 20-session median dollar volume of at least $10m, then the
  maximum over *t−10 … t−1*. ✅
- **Computing Ticker Features** requires **prices**, **Market Context** and **universe**, and yields **Daily Panel**. ✅
- **Computing Rolling-Window State** (strategy guide, Variant B) produces D(t,L), M(t,W), the flat-path
  Release(j) = M(t,W) − max(0, M(t,W−j)), the setup state and arm flags A/B/C. ✅
- **Dating Earnings Releases** requires **filings**. An 8-K Item 2.02 accepted at or after 16:00 ET maps to
  the next session; otherwise it maps to the same session (weekend releases go to the next session). ✅
- **Detecting Events** requires **Settings** and yields **Event Dataset** rows where
  *(earnings ∨ crash ∨ arm A) ∧ liquid*. ✅
- **Event** specialises into **earnings_up**, **earnings_down**, **earnings_crash**, **shock_crash** and
  **momentum_breakout**. Priority: earnings → crash → breakout. ✅
- **Adding Earnings History** affects **Event Dataset** (previous EAR, previous 20-day PEAD once realised, beats in the last 4). ✅
- **Tagging EDGAR Causes** requires **filings** and affects **Event Dataset**. It adds 14 item flags and form
  flags, plus a priority taxonomy: bankruptcy_delisting > restatement_auditor > offering > earnings >
  impairment_restructuring > mgmt_change > deal > regfd_other_8k > none. It also adds prior-activity counts. ✅
- **Measuring Attention** requires **pageviews** and affects **Event Dataset**. ✅
- **Attaching Point-in-time SUE** requires **eps_facts** and affects **Event Dataset**. It uses the
  first-reported EPS, Q4 = FY − Q1..Q3, and only filings dated on or before *t*. ✅
- **Assessing Regime** requires **Daily Panel** (breadth) and changes **Market Context** to *normal*,
  *elevated* or *stressed*. One point each for: SPY below its 200-day MA, VIX/VIX3M > 1, sector-correlation
  percentile > 0.9, breadth below 40%, and a Baa−10y spread rise of more than 25 bp in 20 days. ✅

## SD3 · Researching in-zoomed (`SD3_researching.dot`)

- **Preparing Feature Sets** consumes **Event Dataset** and yields **ML Frame**, with columns TO_PREDICT,
  NUMERIC and DUMMIES (7 categoricals plus 7 engineered bins), split into *train* (to 2021),
  *validation* (2022–23), *test* (2024 →) and *live*. ✅
- **Applying Hand Rules** requires **ML Frame** and yields **Predictions**: HW1 PEAD + momentum,
  HW5 panic-no-damage, all events, and guide arms A/B/C. ✅
- **Tuning** requires **ML Frame** (TRAIN fit, VALIDATION score) and yields **Tuning Table**. ✅
- **Fitting Final Models** requires **Tuning Table** and **ML Frame** and yields **Predictions**. It refits
  on TRAIN+VALIDATION: DecisionTree, RandomForest, XGBoost, LightGBM, LogisticRegression, and an
  RF/XGB/LGBM ensemble. ✅
- **Choosing Thresholds** affects **Predictions**. Each threshold is the highest-precision cut-off on
  VALIDATION that still trades at least 5% of events. ✅
- **Selecting Production Model** yields **Model Bundle** (best VALIDATION AUC; TEST is never used to choose). ✅
- **Vector Simulating** requires **Predictions** and **Cost Model** and yields **Backtest Report** (sim1 and sim2). ✅
- **Exact Simulating** requires **Predictions**, **Daily Panel** and **Cost Model** and yields
  **Backtest Report**. It runs a daily loop with cash, reinvestment, a position cap, stops, targets,
  time exits, the regime pause, a long-short variant and the guide's R framework. ✅
- **Benchmarking** requires **market** and **Daily Panel** and yields **Backtest Report**: SPY and IWM
  buy-and-hold, and the HW1 momentum top quintile. ✅
- **Testing Guide Arms** requires **ML Frame** and the exact simulations. It yields **Backtest Report**
  (four-arm table, sensitivity grid, 1×/2×/3× cost stress) and changes **Window-Release Factor** to
  *kept* or *discarded*. Arm D is not applicable to US equities. ✅
- **Diagnosing** requires **Predictions** and yields **Backtest Report** (precision breakdown, false positives, calibration). ✅

## SD4 · Signalling Event Trades in-zoomed (`SD4_signalling.dot`)

- **Scoring Events** requires **ML Frame** (last 90 days) and **Model Bundle** and yields **Signal**.*prob*. ✅
- **Estimating Net Edge** requires **TEST Predictions** and affects **Signal**.*expected_net_return*
  (the TEST probability-decile mean minus the round-trip cost). ✅
- **Tiering** requires **Market Context** and the **Model Bundle** threshold. It changes **Signal** to one of
  *STRONG_BUY, BUY, WATCH, SKIP, AVOID, NO_CALL*. An *elevated* regime lowers STRONG_BUY by one notch;
  a *stressed* regime lowers STRONG_BUY and BUY by one notch. ✅
- **Sizing & Setting Exits** requires **Strategy Settings** and affects **Signal** (weight, stop,
  take-profit, exit_by_date). Breakouts use 2×ATR / 3R / 8 sessions. ✅
- **Publishing Artifacts** consumes **Signal** and yields **Signal Book** and the other artifacts.
  **Artifact Store** consists of **Signal Book**, **prices_recent / universe / metrics.json** and **Run Log**. ✅
- **Logging Run** yields **Run Log**. ✅

## SD5 · Serving / Monitoring / Scheduling / Trading in-zoomed (`SD5_serving_trading.dot`)

Invariant: consumers read the **Artifact Store** only and never open the **Warehouse**, because DuckDB
allows a single writer process.

- **Agent Interface** consists of **REST API + OpenAPI**, **tools.json + routes.json** and **MCP server**. ✅
- **Personal 24/7 Agent** handles **Briefing**, which requires **xAI chat completions** and **tools.json**
  and invokes **Authenticating** through HTTP tool calls. ✅ (live xAI call not yet verified)
- **Authenticating** changes **Request** to *authorised* or *rejected* (bearer token, constant-time compare). ✅
- **Answering Queries** occurs if **Request** is *authorised*. It requires **REST API**, **MCP server** and
  **Artifact Store**. It invokes **Running Pipeline** and **Previewing Orders**. ✅
- **Scheduling** is handled by **Market Calendar** and invokes **Running Pipeline** at 22:30 ET, Monday to Friday. ✅
- **Running Pipeline** changes **Pipeline Job** through *queued → running → success | failed* and affects
  **Artifact Store**. ✅
- **Researcher** handles **Monitoring**, which requires **Dashboard** and **Artifact Store**. ✅
- **Previewing Orders** requires **Artifact Store** and yields **Order Set** in state *previewed*. ✅
- **Submitting Orders** occurs only if **Order Guard** holds (`confirm=true` and `DRIFTREV_ALLOW_ORDERS=true`).
  It requires **Alpaca API** (paper) and changes **Order Set** from *previewed* to *submitted* or *refused*.
  It is not exposed as an agent tool. ✅ (not run against a live Alpaca account)
- **Closing Expired Positions** requires **Artifact Store**, **Alpaca API** and **Order Guard**. ✅

## SD7 · CI/CD and Housekeeping (`SD7_cicd.dot`)

Invariant: no data file is committed to git. Data persists in the **Data Cache** (Actions cache),
in **Run Artifacts** that expire, and in one rolling **Snapshot Release Asset**.

- **Developer** handles **Integrating (CI)**, which requires **GitHub Actions runner** and
  **Source Code + OPM model** and changes **Build Status** to *passing* or *failing*
  (pytest, OPM SVGs re-rendered with no diff, `docker compose config`, `docker build`). [OPS] ✅
- **GitHub Scheduler** handles **Delivering (CD daily)** at 22:30 ET Monday to Friday; on Sunday it also
  retrains. Delivering occurs only if **Build Status** is *passing* on `main`. It requires **Repository
  Secrets** and **Public Data Providers**. It affects **Data Cache**: *missing* means a full bootstrap
  ingest, *fresh* means an incremental update. It invokes **Signalling Event Trades**. [OPS] ✅
- **Reporting** (`driftrev.report`) yields the HTML and markdown results. **Publishing Results** yields
  **Results Site** (GitHub Pages), **Job Summary** and a **Run Artifact** in state *retained* (30 days). [SIG/OPS] ✅
- **Releasing Snapshot** (Sunday) replaces the single **Snapshot Release Asset** used by `--mode file`. [OPS] ✅
- **Housekeeping Data Files** (`python -m driftrev.housekeeping`) requires **Retention Policy** and changes
  **Data Files** from *current* to *deleted*: EDGAR root JSONs older than 2 days, EDGAR history pages
  older than 180 days, the daily signal archive older than 60 days, snapshot zips except the newest,
  run-log lines beyond 1000, and cron logs beyond 5 MB. It is invoked after every pipeline run. [OPS] ✅
- **GitHub Scheduler** handles **Garbage-collecting GitHub** weekly (`cleanup.yml`). It requires
  **Retention Policy**, changes **Data Cache** and **Run Artifact** entries older than their retention to
  *expired* (deleted), deletes workflow runs older than 30 days, and keeps only the latest snapshot asset. [OPS] ✅

## SD6 · Roadmap (`SD6_roadmap.dot`), PLANNED

- ⬜ **R1 Loading Historical Membership** yields **membership_history** and affects **universe**
  (removes survivorship bias; required before claiming the crash results generalise).
- ⬜ **R2 Walk-forward Retraining** affects **Backtest Report** (yearly expanding refit, stitched OOS equity).
- ⬜ **R3 Notifying** occurs after **Signalling** and requires **Signal Book** (Telegram or e-mail push).
- ⬜ **R4 Classifying Filing Text** feeds **Transforming** (8-K EX-99.1 guidance language, going-concern phrases).
- ⬜ **R5 Cloud Deploying the API** affects **Serving** (container VM / Cloud Run). Batch CD and the
  Pages results site are already implemented in SD7.
- ⬜ **R6 Confirming Intraday Entry** feeds **Signalling** (guide Entry B with intraday bars).
- ⬜ **R7 Extending to Other Markets** feeds **Transforming** (guide Variant A for eligible China A-shares).
