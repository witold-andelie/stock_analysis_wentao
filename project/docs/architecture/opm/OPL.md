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
**Loading Prices**, **Loading Filings & EPS**, **Loading Attention** and **Loading Consensus EPS**.
**Loading Historical Membership** runs right after **Refreshing Universe**.
**Warehouse** consists of **universe**, **membership_history**, **prices**, **market**, **macro**, **filings**,
**eps_facts**, **pageviews**, **consensus_eps** and **edgar_fetch_log**.

- **Refreshing Universe** requires **Wikipedia S&P lists** and **SEC EDGAR** (ticker→CIK) and affects **universe**
  (*is_current_member* = true for today's constituents). ✅
- **Loading Historical Membership** (R1, `membership.py`) requires **Wikipedia S&P lists** (the "Historical components of
  the S&P 500/400/600" change tables), **SEC EDGAR** and **Yahoo Finance**. It reconstructs **membership_history**
  (ticker, index_name, start_date, end_date; member iff start ≤ date < end) backwards from today's constituents and
  the added/removed rows, adds removed names to **universe** with *is_current_member* = false (cap bucket from the index
  they left, sector from SEC SIC, CIK via company_tickers then the EDGAR company search with a name check against
  ticker reuse), and affects **prices** for removed names whose ticker Yahoo still serves (clipped to the membership
  window + 45 days). ✅ 🟡 reduces but does not remove survivorship bias: the S&P 600 table starts 2019-12, the S&P 400
  table 2012, delisted names Yahoo no longer serves stay missing, and renamed tickers are unresolved.
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
- **Loading Consensus EPS** (HW1 Q4, `consensus.py`) requires **universe** and **Yahoo Finance** (`get_earnings_dates`)
  and affects **consensus_eps** (release timestamp, EPS estimate, reported EPS, surprise %). It is incremental
  (tickers refreshed in the last 7 days are skipped) and tolerates rate limits, so coverage is partial by design. ✅

- **Respecting Fetch Cadence** (`config.FetchCadence`, warehouse table `fetch_log`): sources that change a few times a
  quarter are fetched at most every 7 days — US and DE index-membership pages, the Wikipedia redirect API and the US
  removed-name EDGAR searches — and a known German company's Yahoo symbol search is repeated only after 90 days (results
  in the warehouse table `yahoo_search_de`). CI keeps only the warehouse between runs, so the fetch state lives there,
  not in `cache/`. `--refresh-sources` forces a refetch. Daily incremental sources (prices, market series, rates,
  EDGAR submissions, pageviews; consensus EPS with its own 7-day rule) are unchanged. Reason: shared GitHub runner IPs
  get rate-limited (owner decision 2026-11-22). [ING] ✅

## SD2 · Transforming in-zoomed (`SD2_transforming.dot`)

Invariant: a value on row *t* uses only information available by the close of *t*. The exceptions
are filings accepted up to 09:00 ET on *t+1* (the trade is entered at the *t+1* open). FRED and
pageviews are lagged one row. Targets use *t+1 … t+H* only.

- **Building Market Context** requires **market** and **macro** and yields **Market Context**. ✅
- **Ranking Prior Strength** requires **prices** and **Market Context**. It takes the cross-sectional
  percentile of D(t,20) among names with a 20-session median dollar volume of at least $10m, then the
  maximum over *t−10 … t−1*. ✅
- **Computing Ticker Features** requires **prices**, **Market Context** and **universe**, and yields **Daily Panel**. The
  event's *cap_bucket* is the index the stock sat in on that date (from **membership_history**), not today's index. ✅
- **Flagging Index Membership** (R1) requires **membership_history** and **Settings** (`Membership.restrict_events`,
  default on). It adds the point-in-time *in_index* flag (member of any of the three indices on that date) to
  **Daily Panel** and **Event Dataset**, and **Detecting Events** keeps only rows where *in_index* holds. ✅
- **Tracking Market Correction** (HW1 Q3) requires **market** and affects **Market Context**: SPY *in_correction*
  (close ≥ 5% below its running all-time high), *correction_depth* and *days_in_correction*, all point-in-time. ✅
- **Attaching Consensus Surprise** (HW1 Q4) requires **consensus_eps** and affects **Event Dataset**: *eps_surprise_pct*,
  *eps_beat*, *eps_est_available* from the earnings release within ±3 sessions of the event, using only releases
  stamped on or before the event day's close. ✅
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

- **Classifying Filing Text** (in-zoomed in SD2.1) requires **filing_text** and **filings** and affects
  **Event Dataset** with `ft_*` flags, Loughran–McDonald tone and `edgar_category_text`, which refines
  `edgar_category` (e.g. earnings → guidance cut / raise; going concern; offering priced). It is wired into
  `features.build_all`, and its fetch runs in `pipeline.ingest` with a time budget (`DRIFTREV_FILING_TEXT_MAX_SECONDS`). ✅

## SD2.1 · Classifying Filing Text in-zoomed (`SD2.1_classifying_filing_text.dot`)

R4, `driftrev.filing_text`. A deterministic, explainable text layer (regex lexicon + Loughran-McDonald tone, no LLM,
no paid API) on top of the 8-K item codes of SD2: it separates "earnings + guidance cut" from "earnings +
beat-and-raise" and says whether a 424B prospectus actually priced an offering (and at what discount).
Invariant: only filings accepted inside the event window (previous close 16:00 ET .. next morning 09:00 ET) are used.

- **Selecting Target Filings** requires **filings**, **Event Dataset** and **Text Scope**, and reads **filing_text**
  (already fetched accessions are skipped, so a run is incremental and resumable). Scope: every 8-K, 8-K/A, 424B* and
  NT 10-K/Q accepted in the window of a *shock_crash* or *earnings_crash* event (all years), plus the 8-K Item 2.02 of
  *earnings_up* / *earnings_down* events from 2022-01-01 (validation + test). A per-run budget caps the number of filings. ✅
- **Fetching Press Releases** requires **SEC EDGAR Archives**, **Contact User-Agent** and **Rate Limiter** (≤ 8 req/s,
  404-tolerant, retries). It reads the filing index, picks EX-99.1, else EX-99 (or another EX-99.x), else the primary
  document, and downloads it (8-Ks with substantive items also read the primary document). ✅
- **Converting HTML to Text** yields **Exhibit Text Cache** (gzip, capped length). ✅
- **Flagging Lexicon Phrases** requires **Exhibit Text Cache** and affects **filing_text**: the **Lexicon Flags**
  guidance_cut, guidance_raise, guidance_reaffirm, going_concern, offering_priced (+ stated price, last price,
  discount), restatement, material_weakness, investigation, clinical_failure, exec_departure, covenant, impairment,
  layoffs, bankruptcy, beat and miss, each with a negation guard and a matched snippet. ✅
- **Scoring Tone** requires **Exhibit Text Cache** and **Loughran-McDonald Dictionary** (downloaded once; a curated
  fallback list is embedded) and affects **filing_text**: **Tone Scores** lm_negative_pct, lm_positive_pct,
  lm_uncertainty_pct and lm_net_tone of the main exhibit. ✅
- **filing_text** exhibits **Lexicon Flags** and **Tone Scores**; a row with status *no_index*, *no_doc* or *error*
  records a failed fetch (only *error* is retried). ✅
- **Attaching Text Flags to Events** requires **filing_text**, **Event Dataset** and **filings** and yields the
  *ft_** columns (any flag across the window's filings, tone of the main exhibit) and **edgar_category_text**
  (earnings split into earnings_guidance_cut / earnings_guidance_raise / earnings_guidance_mixed / earnings_other,
  plus going_concern, clinical_failure, offering_priced, investigation). `features.build_all` joins it by index (integration step). ✅
- **Studying Text Flags** requires **filing_text** and **Event Dataset** and yields **filing_text_study**: n, base rate,
  mean / median 10-day return, mean vs SPY and a bootstrap 95% CI per flag, split by family and period. ✅

## SD3 · Researching in-zoomed (`SD3_researching.dot`)

- **Engineering Cross-sectional Features** requires **ML Frame** and adds, per event date, the percentile rank
  (`cs_rank_*`) and the date-demeaned value (`cs_dm_*`) of about 30 stock-level numerics. Only same-date
  information is used, so it is point-in-time. A *market-level set* (SPY/VIX/rates/breadth, month, weekday,
  regime and `month_wom` dummies) can be dropped so the model must pick *which* stock, not *when*. ✅
- **Preparing Feature Sets** consumes **Event Dataset** and yields **ML Frame**, with columns TO_PREDICT,
  NUMERIC and DUMMIES (7 categoricals plus 7 engineered bins), split into *train* (to 2021),
  *validation* (2022–23), *test* (2024 →) and *live*. The split is **purged**: rows whose 10-session label window
  (16 calendar days, `Split.purge_days`) reaches past the end of train or validation are labelled *purged* and used
  by nothing (fitting, thresholds, metrics), so no label is realised inside the next block. ✅
- **Applying Hand Rules** requires **ML Frame** and yields **Predictions**: HW1 PEAD + momentum,
  HW5 panic-no-damage, all events, and guide arms A/B/C. ✅
- **Tuning** requires **ML Frame** (TRAIN fit, VALIDATION score) and yields **Tuning Table**. ✅
- **Fitting Final Models** requires **Tuning Table** and **ML Frame** and yields **Predictions**. It refits
  on TRAIN+VALIDATION: DecisionTree, RandomForest, XGBoost, LightGBM, LogisticRegression, and an
  RF/XGB/LGBM ensemble. ✅
- **Choosing Thresholds** affects **Predictions**. Each threshold is the highest-precision cut-off on
  VALIDATION that still trades at least 5% of events and, when a capacity is configured (HW4), whose
  average trades per day does not exceed it. ✅
- **Comparing Model Variants** requires **ML Frame** and yields **Model Comparison**: pooled model, pooled +
  cross-sectional features (market-level features kept or removed), per-family models, and a LightGBM
  `lambdarank` ranker grouped by date. Every row is scored on VALIDATION (train-only fit; this picks the winner)
  and on TEST (refit on train+validation; reported only, never used to choose). ✅
- **Composing the Book** requires **Predictions** and yields **Book Rules** (excluded event families and EDGAR
  categories for the model book). Each exclusion is decided on VALIDATION precision lift and VALIDATION exact
  simulation; the guide arms remain a separate rule-based sleeve. ✅
- **Walk-forward Retraining** (R2) requires **ML Frame**, **Daily Panel** and **Cost Model** and yields
  **Walk-forward Track Record**. For every year Y it refits on events whose 10-day label closed before Y
  (expanding window), takes the threshold from the previous year (fit on data before it), predicts Y, and the
  stitched out-of-sample predictions drive an exact simulation. This is the honest track record. ✅
- **Studying Holding Periods and Uniquely-correct Predictions** requires **ML Frame** and **Predictions** and
  yields the HW2-Q3 style median/mean growth at 5/10/20 sessions per family and the HW3 uniquely-correct table.
  It is a report only; the 10-day target is unchanged. ✅
- **Production decisions (VALIDATION, config.Research)**: feature set `cs_drop_market`; pooled layout (per-family and
  lambdarank did not beat it); model book excludes `momentum_breakout` and EDGAR `offering` (`mgmt_change` is kept:
  its VALIDATION precision is 71%); capacity cap of 5 average trades per active day in the threshold rule. ✅
- **Checking Library Version** warns when scikit-learn is 1.8 or newer (HW4 lesson: random-forest
  probabilities change near thresholds) and logs the versions into **Backtest Report**. ✅
- **Selecting Production Model** yields **Model Bundle** (best VALIDATION AUC; TEST is never used to choose). ✅
- **Cost Model** is cap-aware: one-way 8 bps (large), 15 bps (mid), 25 bps (small); a flat 15 bps option
  remains for comparison. ✅
- **Vector Simulating** requires **Predictions** and **Cost Model** and yields **Backtest Report** (sim1 and sim2). ✅
- **Exact Simulating** requires **Predictions**, **Daily Panel** and **Cost Model** and yields
  **Backtest Report**. It runs a daily loop with cash, reinvestment, a position cap, stops, targets,
  time exits, the regime pause, a long-short variant and the guide's R framework. ✅
- **Benchmarking** requires **market** and **Daily Panel** and yields **Backtest Report**: SPY and IWM
  buy-and-hold, and the HW1 momentum top quintile (point-in-time index members only, like the model's events). ✅
- **Testing Guide Arms** requires **ML Frame** and the exact simulations. It yields **Backtest Report**
  (four-arm table, sensitivity grid, 1×/2×/3× cost stress) and changes **Window-Release Factor** to
  *kept* or *discarded*. Arm D is not applicable to US equities. ✅
- **Diagnosing** requires **Predictions** and yields **Backtest Report** (precision breakdown, false positives, calibration). ✅

## SD4 · Signalling Event Trades in-zoomed (`SD4_signalling.dot`)

- **Scoring Events** requires **ML Frame** (last 90 days) and **Model Bundle** and yields **Signal**.*prob*. ✅
- **Estimating Net Edge** requires **TEST Predictions** and **Cost Model** and affects **Signal**.*expected_net_return*
  (the TEST probability-decile mean minus the round-trip cost of the stock's cap bucket). ✅
- **Tiering** requires **Market Context**, **Cost Model**, **Book Rules** and the **Model Bundle** threshold.
  Events of an excluded family or EDGAR category are SKIP in the model book. It changes **Signal** to one of
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
- **Answering Research Questions** (`GET /v1/research`, MCP `get_research`) requires **Artifact Store**
  and returns the research tables: model comparison, book composition, walk-forward track record,
  holding periods, uniquely-correct predictions, guide arms, survivorship. [SRV] ✅
- **Answering Queries** occurs if **Request** is *authorised*. It requires **REST API**, **MCP server** and
  **Artifact Store**. It invokes **Running Pipeline** and **Previewing Orders**. ✅
- **Scheduling** is handled by **Market Calendar** and invokes **Running Pipeline** at the active **Market**'s run time,
  Monday to Friday (US default 22:30 New York; R7.1). ✅
- **Running Pipeline** changes **Pipeline Job** through *queued → running → success | failed* and affects
  **Artifact Store**. ✅
- **Researcher** handles **Monitoring**, which requires **Dashboard** and **Artifact Store**. ✅
- **Previewing Orders** requires **Artifact Store** and yields **Order Set** in state *previewed*. ✅
- **Submitting Orders** occurs only if **Order Guard** holds (`confirm=true` and `DRIFTREV_ALLOW_ORDERS=true`).
  It requires **Alpaca API** (paper) and changes **Order Set** from *previewed* to *submitted* or *refused*.
  It is not exposed as an agent tool. ✅ (not run against a live Alpaca account)
- **Running Pipeline** invokes **Notifying** after a successful publish. **Notifying** occurs if **Notification
  Settings** are configured; it requires **Artifact Store** and sends a message (data date, regime, signals at
  or above `NOTIFY_MIN_TIER`, with stop / target / exit date) to **Telegram chat / webhook receiver**. [SRV] ✅
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
- **Delivering German CD daily** (`daily_de.yml`, R7) runs `DRIFTREV_MARKET=de` at 17:30 UTC Monday to Friday (18:30 Berlin
  in winter) with a Sunday retrain; its **Data Cache** is `driftrev-de-data-*` (paths under `project/data/de/`). There is no
  German snapshot release. [OPS] ✅ (first real run pending)
- **Composing the Results Site**: each daily workflow saves its site in `driftrev-site-us-*` / `driftrev-site-de-*`, restores
  the other market's latest site, and deploys one Pages site (US at `/`, Germany at `/de/`); both deploy jobs share the
  `pages-deploy` concurrency group. [OPS] ✅ (first real run pending)
- **Housekeeping Data Files** (`python -m driftrev.housekeeping`) requires **Retention Policy** and changes
  **Data Files** from *current* to *deleted*: EDGAR root JSONs older than 2 days, EDGAR history pages
  older than 180 days, raw filing text older than 30 days (flags and tone stay in the warehouse), the daily signal archive older than 60 days, snapshot zips except the newest,
  run-log lines beyond 1000, and cron logs beyond 5 MB. It is invoked after every pipeline run. [OPS] ✅
- **Publishing Container Image** (`image.yml`) occurs on a push to `main` touching `project/**` (or manually). It
  requires **Build Status** *passing*, **GitHub Actions runner** and **GitHub Container Registry**, and changes
  **Container Image** from *unpublished* to *published* (`:latest` and `:sha-<short>`), then runs the pushed image
  and imports the package. [OPS] ✅
- **Operator** handles **Deploying API (optional)** manually. It requires a published **Container Image**, **Cloud
  Targets** (Google Cloud Run, Fly.io or Render manifests in `project/deploy/`), a **Persistent Volume** for `/data`
  (single DuckDB writer) and **Repository Secrets** equivalents (`DRIFTREV_API_TOKEN`, `SEC_USER_AGENT`), and changes
  **Hosted API** from *not deployed* to *serving* (health check `/health`). Templates only; never run by CI. [OPS] ⬜ (not deployed)
- **GitHub Scheduler** handles **Garbage-collecting GitHub** weekly (`cleanup.yml`). It requires
  **Retention Policy**, changes **Data Cache** and **Run Artifact** entries older than their retention to
  *expired* (deleted) but always keeps the newest cache of each prefix (`driftrev-data-`, `driftrev-de-data-`,
  `driftrev-site-us-`, `driftrev-site-de-`), deletes workflow runs older than 30 days, and keeps only the latest snapshot
  asset. [OPS] ✅

## SD6 · Roadmap (`SD6_roadmap.dot`), PLANNED

- ✅ **R1 Loading Historical Membership** yields **membership_history** and affects **universe**
  (reduces survivorship bias; now documented in SD1 and SD2; see the gaps listed under SD1).
- ✅ **R2 Walk-forward Retraining** is implemented, see SD3 (**Walk-forward Retraining**).
- ✅ **R3 Notifying**: implemented, now part of SD5 (`driftrev.notify`, Telegram Bot API + JSON webhook).
- ✅ **R4 Classifying Filing Text** feeds **Transforming**: implemented, see SD2.1 (`driftrev.filing_text`; EX-99.1 guidance language, going-concern phrases, tone).
- ✅ **R5 Cloud Deploying the API** (packaging) affects **Serving**: the image is published to GHCR and manifests
  for Cloud Run, Fly.io and Render exist (SD7, `project/deploy/`). The deploy itself is a manual **Operator** step;
  nothing is deployed by this repository.
- ⬜ **R6 Confirming Intraday Entry** feeds **Signalling** (guide Entry B with intraday bars).

### R7 · German equities on Deutsche Börse Xetra

Scope: DAX 40 + MDAX 50 + SDAX 70 (160 names; large / mid / small like S&P 500 / 400 / 600), traded on Xetra
in EUR. The US system stays the default and must behave exactly as today. All existing invariants hold per
market: no look-ahead (row *t* uses data available by the Xetra close of *t*), TEST is never used to choose,
tunables live in `config.py`, orders stay paper by default behind the **Order Guard**. China A-shares stay out of scope.

- ✅ **R7.1 Selecting Market** (`config.Market`, `config.MARKETS`, env `DRIFTREV_MARKET`, default *us*) affects
  **Market** (*us* | *de*). **Market** exhibits exchange MIC (XNYS | XETR), timezone, scheduled run time (US 22:30 New
  York; DE 18:30 Berlin, after the 17:30 Xetra close), currency, benchmarks, one-way cost per cap bucket (DE values are an
  ASSUMPTION to revisit in R7.4), index set with cap buckets, and its own data directory (*us* keeps `DATA_ROOT`, *de*
  uses `DATA_ROOT/de`, so each market has one DuckDB writer). **Scheduling** uses the active market's timezone and run
  time. **Running Pipeline** and **Scheduling** require an *implemented* **Market** (`MarketNotImplementedError` before any
  file is written). *de* was refused this way until R7.4g marked it implemented. [OPS/ING]
- 🟡 **R7.2 Loading German Index Membership** (R7.2a ✅ membership; R7.2b ✅ symbols and prices; ISIN not stored) requires **Public Sources (de)** (DAX / MDAX / SDAX constituent
  pages and their change history) and yields **membership_history (de)** (ticker, ISIN, index, start, end), which
  **universe** requires. ISIN replaces the SEC CIK as the stable key; Yahoo serves Xetra prices as `<symbol>.DE`.
  The September 2021 index reform (DAX 30 → 40, MDAX 60 → 50) must be reconstructed. *(As implemented in R7.2a/b the identity key is
  `company_key` (redirect-resolved article title); ISIN is not stored.)* If history cannot be
  reconstructed, the gap is reported as survivorship bias exactly as in SD1. [ING]
  *Source check 2026-11-22:* DAX: current 40 + change table (en "DAX" 66 rows, de "DAX" § Zusammensetzungshistorie
  77 rows). MDAX: current 50 + full change table on de "MDAX/Zusammensetzungshistorie" (293 rows, 1994 → 2026; company
  names only, no tickers, so a name → ISIN → Yahoo map is needed). SDAX: current 70 with "im SDAX seit" only, no change
  history page; SDAX moves are partly visible in the MDAX table (names going up or down). 🟡 so SDAX keeps a survivorship
  gap, like the S&P 600 before 2019-12. Yahoo has no prices for delisted names (Wirecard `WDI.DE`, Osram `OSR.DE`
  returned empty). Wikimedia rate-limits bursts (HTTP 429): use the REST page endpoint, a cache and pauses.
  - ✅ **R7.2a Reconstructing German Membership** (`driftrev.de_membership`, `update_de_membership()` with
    `DRIFTREV_MARKET=de`) requires the four Wikipedia pages (DAX, MDAX, SDAX, MDAX/Zusammensetzungshistorie) and the
    Wikipedia redirect API, and yields **membership_history (de)** (company_key, name, index, start, end; end empty =
    member). Change tables are read as a rowspan/colspan grid; SDAX events are derived from the MDAX table's
    `SDAX` markers. Identity: union-find over redirect-resolved article titles and normalised names, plus the
    evidenced aliases in `resources/de_aliases.csv`. Replay rules mirror R1 (adds before removals on a date; a first
    removal means "member before coverage"; current-table dates seed missing open intervals; open intervals of
    non-current names are dropped and reported as orphans). On the 2026-11-22 fixtures: 627 intervals; on 2026-11-20
    exactly 40 / 50 / 70 members; one DAX/MDAX orphan left (Victoria, 2001); 62 SDAX orphans = names that left SDAX
    for no other index (exit date unknown). Their open SDAX intervals stay dropped: closing them at the last Xetra
    trading day was considered but is **not implemented** (R7.2b resolves symbols and prices only).
    *Same-day identity swap (2026-11-23):* when one key has both an add and a removal on one date in one index, it is
    a continuation (share-class or legal-form swap), counted in `same_day_swap`: an open interval stays open; with no
    open interval it opens at the "member before coverage" start if the key has no earlier event, else at that date.
    This fixes Linde: Linde AG and Linde plc both normalise to `linde` (one company), and the DAX row 25.10.2017 swaps
    them; adds-before-removals used to turn that into a dropped zero-length interval, so Linde vanished from the DAX
    (29 members in 2014–2020, 39 in 2021–2022). Now Linde is a DAX member from before coverage to 2023-02-27, and the
    DAX has 30 / 40 members at every year-end 2014–2025. Swaps on the fixtures: Linde 2017 and Degussa 1999 / 2000.
    *Year-end check (notebook 06, 2026-11-23):* SDAX is overstated early: 58 members on 2014-12-31 vs 50, because a
    name whose first SDAX event is a move up to MDAX gets the `start_unknown` start (47 of the 58). 🟡 MDAX is one
    short in 2017–2024 (49 / 59 / 49 instead of 50 / 60 / 50), from two identity collisions: the static `Gea` alias
    also catches the 1999 MDAX removal of the old GEA AG (bought by Metallgesellschaft, today GEA Group), which closes
    GEA Group's MDAX interval in 1999; and the 2017 Metro split (old Metro AG renamed Ceconomy, new Metro AG listed)
    is read as one `Metro AG` key, so the new Metro's 2017 MDAX entry is a double add and Ceconomy gets an interval
    from before coverage. In 2014–2016 the two errors cancel out. Not fixed yet: a fix needs date-scoped identities.
  - ✅ **R7.2b Resolving German Symbols and Prices** (`driftrev.de_universe.update_de_universe()`) requires
    **membership_history (de)** and **Yahoo Finance** (search, cached) and yields **universe (de)** (company_key, symbol,
    exchange, `price_source` xetra | hamburg | other, `resolved_by` search | override | unresolved, latest index and cap
    bucket, first/last membership date) and **prices (de)**. A Yahoo listing is accepted only if every token of its
    long or short name (legal forms removed) prefixes a token of the company's names; Xetra beats Hamburg. This rejects
    real look-alikes in the data (`Osram` → ams-OSRAM `DQW.DE`, `Siemens` → Siemens Energy, `Porsche AG` → `PAH3.DE`).
    Evidenced manual rows live in `resources/de_symbol_overrides.csv`. Live run 2026-11-23 (after the swap fix): 202 companies since
    2014, 181 resolved (164 Xetra by search, 8 Hamburg by search, 9 overrides; Linde plc → `LIN.DE` by search), all 160
    current members on Xetra, 521,042 price rows. 🟡 21 delisted names have no Yahoo prices on any German venue (e.g. Axel Springer, Kuka, Innogy, Stada,
    Morphosys): a remaining survivorship gap. ISIN is not stored (not needed by any consumer yet). Source check
    2026-11-22: Yahoo search finds most names (umlauts need `ae/oe/ue` spelling, e.g. Stroeer → `SAX.DE`; legal form
    helps, `1&1 AG` → `1U1.DE`); delisted names missing on Xetra still have Hamburg prices (`WDI.HM`, `OSR.HM`,
    `RKET.HM` from 2014) with tiny volume, so they need a price-source flag and must not drive the liquidity filter.
- ✅ **R7.3 Loading German Market Series** (`driftrev.de_market`, `python -m driftrev.de_market` with `DRIFTREV_MARKET=de`;
  refuses any other market) requires **Public Sources (de)** and feeds **Acquiring**: DAX, MDAX and
  SDAX index levels as benchmarks for D(t,L), a volatility gauge and Bund yields for the regime monitor. [ING]
  *Source check 2026-11-22:* Yahoo serves `^GDAXI`, `^MDAXI`, `^SDAXI`, `^TECDAX`, `^STOXX50E` and the iShares DAX ETF
  `EXS1.DE` daily from 2014. VDAX-NEW and VSTOXX are **not** served (`^V1X`, `V1X.DE`, `^VDAX`, `^V2TX` empty), so the
  volatility gauge is a realised-volatility proxy. The daily 10-year Bund yield comes from the Bundesbank time-series
  API (`BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A`, CSV, 2014 → today); FRED only has it monthly.
  Implementation: index levels go to the *de* warehouse table **market** (same loader as US prices), Bund yields to
  **macro** as series `bund10y` (re-pulled 60 days back, like FRED). The volatility gauge is
  `indicators.realised_vol` (annualised 20-session close-to-close vol, VIX scale) and `realised_vol_term`
  (20/60 ratio, the VIX/VIX3M analogue), windows in `config.RealisedVol`. Both are point-in-time and tested; they
  are wired into a German **Market Context** in R7.4a; the loaders run in `de_pipeline.ingest_de` (R7.4g).
- ✅ **R7.4a Building German Market Context** (`driftrev.de_context.market_context_de`) requires the *de* **market** and
  **macro** tables and yields **Market Context** under the *market-neutral column contract*: `spy_*` = the market's
  primary benchmark (DAX), `iwm_*` = its small-cap benchmark (SDAX), `vix` / `vix_chg_5d` / `vix_term` = its volatility
  gauge (DAX realised volatility, 20/60 sessions). Every column of the US **Market Context** exists (US-only concepts:
  sector correlation and FRED rates, as NaN), plus `bund10y` (lagged one session), `bund10y_chg_20d`, `mdax_ret_20d`,
  `mid_vs_large_20d`, `rv_20`, `rv_term`. `features.regime_table` runs on it unchanged. [TRF]
- ✅ **R7.4b Building the German Event Dataset** (`driftrev.de_pipeline.transform_de`, `run_de`; `python -m
  driftrev.de_pipeline`) runs the *de* warehouse through the **same** `features.build_all` as the US, with three
  backward-compatible parameters (`ctx` = the R7.4a context, `cap_buckets` = dax/mdax/sdax → large/mid/small,
  `liquid_by_membership` = symbols priced on a regional venue, whose liquidity comes from index membership instead of
  their unrepresentative volume) and yields **Event Dataset**, **Daily Panel** and **Market Context** (with regime) in
  the *de* warehouse. Without filings, events get `edgar_category = "none"`. Families: *shock_crash* and
  *momentum_breakout* only (earnings need R7.5). Live run 2026-11-22: 6,445 events 2014-02-06..2026-11-20 (5,921
  breakouts, 524 crashes; large 3,124 / mid 2,971 / small 350). 🟡 Small caps are under-represented: SDAX member days
  have a median 20-day turnover of about EUR 1.6m, below the shared `EventRule.min_dollar_volume` of 2m, so 58% of them
  fail the liquidity gate; plus the SDAX history gap of R7.2a (38 of ~50 members reconstructed for 2014).
- ✅ **R7.4c Market-specific Liquidity Floors**: **Market** exhibits `min_dollar_volume` (event gate, 20-session average
  turnover; US USD 2m, DE EUR 1m — owner decision 2026-11-22) and `breakout_min_median_turnover` (rolling-window ranking
  and arm A; US 10m, DE EUR 5m keeping the US 5:1 ratio — ASSUMPTION). `build_all` / `ticker_features` /
  `rolling_window.prior_strength` / `arm_flags` take them as parameters (defaults = the US values); `transform_de`
  passes the German ones. Live: 7,993 German events (breakouts 7,394, crashes 581; large 2,809 / mid 4,217 / small 967).
- ✅ **R7.4d Market-aware Research** (`config.research_settings()`, `pipeline.benchmark_tickers()`,
  `de_pipeline.research_de`; `python -m driftrev.de_pipeline --research [--tune]`): **Researching** requires the active
  **Market**. Benchmarks are the market's `primary_benchmark` / `small_benchmark` (US SPY / IWM, DE DAX / SDAX), the
  cap-aware **Cost Model** uses the market's bps (DE 10 / 20 / 35), and **Book Rules** come from
  `RESEARCH_BY_MARKET` (US = the US VALIDATION decisions; DE = no exclusions until German VALIDATION decides). US
  behaviour is unchanged (all US tests pass unchanged).
- 📊 **First German research run (2026-11-22, `research_de(tune=True)`, 110 s; German-only models)**: 7,993 events,
  split train 4,500 / validation 1,354 / test 2,096 (43 purged); base rate on TEST 38.8 %. Production model (best
  VALIDATION AUC) LightGBM: VALIDATION AUC 0.534, TEST AUC 0.549, threshold 0.63 → only 11 TEST trades in the model
  books (`model_top5` CAGR −0.5 %). VALIDATION model comparison winner: pooled + month/week dummies, random forest
  (VAL AUC 0.551). VALIDATION composition study suggests **all families** (no exclusion), which confirms the empty German
  book rules. Walk-forward (yearly refits from 2018, out-of-sample 2024-01-02..2026-11-20): OOS AUC 0.560, precision of
  selected events 49.4 % vs 38.8 % base rate, 67 trades, CAGR 8.5 %, Sharpe 0.73, max drawdown −11.3 %, versus DAX
  buy-and-hold CAGR 14.5 %, Sharpe 0.93 and the HW1 momentum top quintile CAGR 25.0 %. Buying every event
  (`all_events_unconditional`) made CAGR 22.0 %, Sharpe 0.88. Guide arms: A −0.5 % (487 trades), B +0.5 % (17), C
  +0.5 % (9). **Reading:** the German ranking signal is weak (AUC ≈ 0.55) and does not beat the DAX; the VALIDATION
  threshold is too selective on TEST. TEST numbers are reported only, never used to choose.
- ✅ **R7.4f Comparing Pooled and German-only Models** (`driftrev.research_pooled`, read-only on both warehouses):
  four training sources × {LightGBM, RF}, scored on German VALIDATION (choose) and German TEST (report only), feature
  set `cs_drop_market` restricted to features both markets have (+ `is_de` for pooled fits). Run 2026-11-22 (US 79,544
  train events, DE 4,500): German-only LightGBM VAL AUC 0.534 / prec@top10 % 47.4 %; **pooled all US + DE LightGBM VAL
  AUC 0.555 / 50.4 % (VALIDATION winner)**; pooled same families RF 0.540; US-only transfer RF 0.549. TEST AUCs are
  0.544–0.551 for every variant (German-only LightGBM 0.549, pooled winner 0.544). **Reading:** the VALIDATION gain of
  the pooled model (+0.021 AUC) is about 1.3 standard errors at n = 1,354 and does not show up on TEST; no variant gives
  a material German edge. Production for `de` stays German-only (simpler, same TEST quality) until a richer German
  signal (earnings, R7.5) is available; the pooled option remains reproducible with this module.
- ✅ **R7.4g Running the German Day**: `run_pipeline` dispatches on **Market** (*de*: `update` = `de_pipeline.ingest_de` +
  `transform_de`, `warehouse` = `load_from_warehouse_de`, `file` refused), then shares training, publishing, report,
  housekeeping and notification; metrics and run log carry `market`. German **Signal Book** lists only tradable `.DE`
  symbols (Hamburg-priced names stay in the event history). The regime explanation and the report's benchmark and
  description follow the market. **Market** *de* is now *implemented*, so `DRIFTREV_MARKET=de` works for the pipeline, the
  scheduler (18:30 Berlin) and the API. Live run 2026-11-22 (`--mode warehouse`): success; data through 2026-11-20;
  2 German signals (LPKF, Jenoptik: shock crashes, both SKIP); regime *elevated* (DAX below its 200-day MA, realised-vol
  term 1.29, breadth 33 %).
- ✅ **R7.4 German events end to end** (R7.4a–g). Open: German earnings families need R7.5; a daily CD job for *de*
  (SD7) is not set up yet. requires **membership_history (de)** and feeds **Transforming**. It is the
  first German event family because it needs prices only: the rolling-window Variant B state is computed against
  the German benchmark, with a liquidity floor in EUR (a `config.py` ASSUMPTION). Xetra has no cumulative-deviation
  rule either, so Variant A / Arm D remains out of scope. Whether German events are pooled with US events or
  modelled separately is decided on VALIDATION. [TRF/RES]
- 🟡 **R7.5 Dating German Releases** (earnings ✅, ad-hoc notices ⬜) requires **Public Sources (de)** and feeds **Transforming**: earnings release
  dates for the *earnings_** families and MAR Art. 17 ad-hoc notices as the analogue of SEC 8-K items for
  *shock_crash* causes. A release after the 17:30 CET close maps to the next session. [ING/TRF]
  *Source check 2026-11-22:* Yahoo `get_earnings_dates` covers German names back to 2014 with EPS estimate and
  reported EPS (SAP, Allianz, Evonik, Bilfinger: ~49 past releases each), so the *earnings_** families and the
  consensus surprise are feasible. Its timestamps come in US Eastern time and are sometimes only a date, so the
  session mapping must be checked against prices. No free, documented bulk archive of ad-hoc notices was found
  (EQS publishes per-issuer IR feeds only), so ad-hoc causes for *shock_crash* stay ⬜ until a source exists; German
  crashes can still be detected from prices, without a cause taxonomy.
  *Implemented 2026-11-22 (earnings):* **Market** exhibits `close_time` (US 16:00, DE 17:30);
  `consensus.release_session_dates` converts Yahoo's US-Eastern stamps to the market timezone and maps a release at or
  after the local close to the next session (DST-gap weeks handled); `consensus_features` uses the market's local close as
  the knowability cutoff; `build_all(earnings_by_ticker=...)` replaces EDGAR dating for German tickers; `ingest_de` loads
  consensus EPS for Xetra symbols (154 of 170 have data, 5,346 releases 2013-01..2026-11). **Alignment check:** median
  |return| is 2.31 % on the mapped session vs 1.08–1.47 % on the two sessions before and after, and the mapped session is
  the largest of the five in 42 %; releases stamped 18:00–23:59 Berlin (424, ~9 %) look one session late (the session
  before reacts as much), an accepted error because fixing it from prices would be look-ahead.
  **Research with earnings (11,933 events):** 4,197 earnings events (4,170 with a realised 10-day outcome); some former
  crash / breakout days are now earnings days (earnings take priority), so the total rose by 3,940, not 4,197. production RF VAL AUC 0.528, TEST AUC 0.546;
  walk-forward OOS AUC 0.534, CAGR −1.2 %; `earnings_up` and `earnings_down` have the same 10-day hit rate (37.5 % vs
  36.0 %): **no German post-earnings drift** is visible with these features. The earnings families stay in the German
  dataset (VALIDATION composition keeps all families).
- 🟡 **R7.6 Trading on Xetra** (`driftrev.broker_ibkr`; owner decision: Interactive Brokers, `ib_async==2.1.0`, imported
  lazily) requires **Signal Book**, **Deutsche Börse Xetra** (IB exchange `IBIS`, EUR) and a running TWS / IB Gateway, and
  feeds **Serving**: `service.broker_for_market()` routes *us* → Alpaca, *de* → IBKR. Same public API and **Order Guard**
  as Alpaca, plus: no connection unless `IBKR_ENABLED=true`; paper by default and then only a `DU…` account is accepted.
  Orders: hand-built bracket = DAY market BUY parent (submitted after the close, it executes at the next Xetra open) +
  GTC take-profit limit + GTC stop, all tagged `orderRef=driftrev-<SYMBOL>-<yyyymmdd>` for idempotency. Expiry closing
  is long-only, idempotent per day, and cancels the bracket exits before the SELL (a cancel error blocks the SELL).
  Tested offline with a fake IB only. **To verify in a paper account before any real use:** IBIS contract qualification
  and permissions, after-close DAY parent handling, bracket acceptance after disconnect, GTC child activation and sibling
  cancellation, and the cancel-acknowledgement race before the expiry SELL. [OPS]
