# Stock Market Analytics Zoomcamp — Homework 2 (Module 2, 2026 cohort)

> 本 README 仅对应 **Homework 2（Module 2 "One Dataframe", 2026 cohort）**，不覆盖其他作业。

Solutions for Homework 2 (2026 cohort) of the
[Stock Market Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).
Spec: [`cohorts/2026/homework2.md`](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework2.md).
All analysis is reproducible from `solution.py` in this folder.

---

## Answers (summary)

| Question | Answer |
| --- | --- |
| **Q1** — Total withdrawn IPO value for the company class with the highest total | **Acquisition Corp — $499.99M ≈ 500 ($M)** (32 withdrawn rows before 2026-09-11). Runner-up: Inc. $351.00M, Holdings $311.66M. → 选项 **500** |
| **Q2** — Median Sharpe ratio (as of 2026-09-11) for 2025 IPOs (Offer Date < 2025-09-01, Return ≠ 0) | **median Sharpe ≈ 0.050** (finite-only median ≈ 0.048; growth_252d median 0.596, mean 1.060 — high-growth outliers pull the mean up). 146 tickers after filter (live page; spec snapshot said 148), 133 with yfinance data, 131 rows on 2026-09-11, 129 reached the 252-day milestone. → 选项 **0.04**（最接近） |
| **Q3** — Optimal holding period (1–12 months) maximizing **median** future growth | **1 month — max median growth ≈ 0.9401**. Medians decay monotonically after month 1 (2m 0.892 → 11m 0.515), while means stay > 1 on outliers. → 选项 **1** |
| **Q4** — Net income investing $1000 on every RSI < 30 oversold signal (2000-01-01–2025-06-01) | **5,206 trades, avg 30d return +1.26%, win rate 55.13%, net ≈ $65,805.59 ≈ $65.8K**. → 选项 **65** |

Machine-readable versions are written to
[`data/outputs/answers.json`](data/outputs/answers.json) when you run the pipeline.

### Q5 (exploratory, free text)

> Most IPO strategies show negative median/mean returns (Q3 medians < 1 for
> every holding period; Q2 growth_252d median 0.60 vs mean 1.06 — a few
> outliers mask broad underperformance). To increase profitability I would:
> (1) filter IPOs by quality signals before entry — profitable revenue scale,
> reputable underwriters, realistic offer pricing vs. peers, insider/lock-up
> structure; (2) replace blind buy-and-hold with momentum/technical triggers
> (e.g. only hold while price > 20-day MA, or Q4-style RSI mean-reversion
> entries after the initial IPO sell-off); (3) cut losers fast (time-stop /
> trailing stop around month 1–2, where Q3 medians peak then decay) and let
> the few winners run; (4) size positions by volatility (Sharpe-style
> risk adjustment from Q2) instead of equal weight.

---

## Repository layout

```
homework2/
  solution.py         Main pipeline: fetch + analyze Q1–Q4, prints answers
  README.md           This file
  data/               (generated at run time, NOT committed)
    raw/              Cached downloads (*.parquet): iposcoop tables,
                      yfinance OHLCV, precomputed RSI parquet
    outputs/          Analysis results (CSV, JSON)
```

### Deliverables (under `data/outputs/`)

| File | Description |
| --- | --- |
| `answers.json` | Final answers in machine-readable form |
| `q1_withdrawn_by_company_type.csv` | Total withdrawn value per Company Type |
| `q1_withdrawn_details.csv` | 32 withdrawn rows with parsed prices/values |
| `q2_2025_ipos_filtered.csv` | 2025 IPOs after Offer-Date/Return filter |
| `q2_asof_2026-09-11.csv` | growth_252d / volatility / Sharpe on 2026-09-11 |
| `q2_sharpe_describe_2026-09-11.csv`, `q2_top10_growth.csv`, `q2_top10_sharpe.csv` | describe() + top-10 comparison |
| `q3_ipo_entry_growth.csv` | One row per IPO with future_growth_1m…12m |
| `q3_median_mean_by_month.csv`, `q3_future_growth_describe.csv` | Median/mean per holding period |
| `q4_oversold_trades.csv`, `q4_return_describe.csv` | 5,206 RSI<30 trades + return stats |

---

## How to run

Requirements: Python 3.10+, `pandas`, `numpy`, `requests`, `pyarrow`,
`yfinance`, `gdown` (for the Q4 precomputed parquet).

```bash
# run from the repository root that contains homework2/ as homework2/
# (in the course repo: homework/homework2/; in the personal repo: homework2/)
python homework2/solution.py
# or, inside this folder:
python solution.py
```

### Caching & network

- Raw datasets are cached as parquet files in `data/raw/`. Re-runs read the
  cache and do **not** hit the network unless a cached file is missing.
- `data/raw/q4_precomputed.parquet` (~150 MB) is downloaded once from Google
  Drive (file id `1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-`) and excluded from git.
- 13 of the 146 filtered tickers are delisted on Yahoo Finance
  (`CAEP, SDM, AGH, SKBL, AHL, MCTR, MTSR, EMPG, TBH, PTNM, CEPT, MJID, EPWK`)
  and are skipped — 133 tickers remain, matching the spec's "~134".
- Live sources drift: the iposcoop return column changes daily, so the
  Return ≠ 0 filter yields 146 tickers today vs 148 in the spec snapshot;
  the Q2 median (≈0.05) still maps to the **0.04** option unambiguously.

---

## Notes

- Q1 follows the spec order for Company Type
  (Technologies → Acquisition Corp → Inc. → Group → Limited → Holdings → Other);
  `Shares_offered_value = Shares × Avg(Price Low, Price High)`, falling back to
  `Est $ Vol` when prices are missing.
- Q2/Q3 use the `Close` column per spec:
  `growth_252d = Close/Close.shift(252)`,
  `volatility = Close.rolling(30).std()*sqrt(252)`,
  `Sharpe = (growth_252d − 0.05)/volatility`;
  future growth uses 21 trading days = 1 month.
- Q4 replicates the spec snippet exactly: RSI < 30, 2000-01-01–2025-06-01,
  `net = 1000*(growth_future_30d−1).sum()`.
