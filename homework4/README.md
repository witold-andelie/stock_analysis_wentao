# Stock Market Analytics Zoomcamp — Homework 4 (Module 4, 2026 cohort)

Solutions for Homework 4 (2026 cohort) of the
[Stock Market Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).
Spec: [`cohorts/2026/homework4.md`](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework4.md).
The pipeline in `solution.py` follows the
[2026 Module 4 notebook](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/04-trading-strategy-and-simulation/%5B2026%5D_Module_04_Colab_Trading_Simulations.ipynb)
(same dataset, variable sets, dummies, temporal split, hard-coded tree / Random Forest hyperparameters, and `sim1` $100-per-positive-prediction simulation).

Each question is a **fresh run** of the notebook setup, as the spec requires: Q1/Q2 share the shortened dummy list; Q3 starts again from the original dummies and a redefined target.

**Environment matters**: answers below are the official solution values and reproduce only on
**scikit-learn 1.7.x (verified on 1.7.2**, the Colab default when the course workbook was run).
On 1.8.0 the Random Forest probabilities shift slightly at the decision boundaries and Q1/Q2
come out 1.1279 / 0.57 instead — see the root-cause section below; `solution.py` warns if it
detects scikit-learn ≥ 1.8.

---

## Answers (summary)

| Question | Answer |
| --- | --- |
| **Q1** — CAGR of `pred10_rf_best_rule_55` after dropping `Ticker` and `month_wom` dummies | **1.136** (3,514 TEST trades, avg 5.16/day; 1.1359 rounded to 4 decimals — the notebook prints `np.round(CAGR, 3)`). Above the original 301-feature CAGR **1.1195**, as the spec expects. |
| **Q2** — RF probability threshold of the second-best rule with avg. daily investments **< 5** | **0.56** — `pred12_rf_best_rule_56`, CAGR **1.1178** (≈ 1.12), 4.40 trades/day. `pred10` (0.55) is still best on CAGR (1.136) but trades 5.16 times/day — slightly above the 5/day cap, so it is excluded. |
| **Q3** — number of trades for `pred5_clf_10` after redefining the target as `growth_future_30d > 1.022` | **8** — notebook `Average investments per day` = 8.01 → `int(round(8.01)) = 8` (options 6 / 7 / 8 / 9). 7,337 TEST investments over 4 years; CAGR 1.1219, also the best CAGR on the new target. |

Machine-readable versions are in
[`data/outputs/answers.json`](data/outputs/answers.json) (committed results of
the last run; regenerated whenever you run the pipeline).

### Root cause of the first submission's wrong answers (scikit-learn version drift)

The first run used **scikit-learn 1.8.0**. Everything else in the recipe was already correct:
same dataset snapshot (`stocks_df_combined_2026_09_18.parquet.brotli`), same 208/301 features,
same temporal split (136,842 / 32,287 / 33,163), same hard-coded RF
(`n_estimators=100, max_depth=13, random_state=42`), same `>=` thresholds and `sim1`.
Even the manual-prediction gross revenues match the workbook's committed outputs to six
decimals (`pred1` gross 44,647.366885) — so the data and split were pinned accurately from
the start. The only run-to-run variable was the **scikit-learn version**: 1.8.0 builds a
slightly different Random Forest than 1.7.2, which moves a handful of boundary probabilities
(3,756 → 3,514 trades at threshold 0.55) and therefore the whole Q1/Q2 table:

| sklearn | pred10 trades | avg/day | Q75 | net ($) | Q1 CAGR | Q2 pick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.8.0 (first run) | 3,756 | 6.0581 | 7 | 12,984 | 1.1279 | 0.57 (0.56 rule = 5.35/day ✗) |
| **1.7.2 (course env.)** | **3,514** | **5.1601** | **6** | **11,972** | **1.136** | **0.56** (4.40/day ✓) |

High-signal cross-checks against the spec text after switching to 1.7.x: Q2's premise that
`pred10` "trades slightly more than 5 times per day on average" holds at 5.16 (was 6.06),
and its description of the second-best rule ("close to 1.12" CAGR) matches 1.1178 (was 1.0920).

A second, independent "exactness" bug was fixed at the same time: `simulate()` now mirrors the
workbook's **per-row fee/net accounting** — rows whose `growth_future_30d` is NaN (the last ~21
TEST trading days) are skipped by the pandas column sums of gross/net revenue even though they
still count in `sim1_count_investments`, avg/Q75 per day, and the raw fees column. Proof it now
matches the workbook byte-for-byte: the regenerated original-setup `pred6` row (6,677 trades,
gross 22,383.446, fees −1,335.4, net 21,050.646 = 13 NaN-growth rows × $0.2 difference) is
identical to the committed notebook's embedded output. This shifted some CAGRs by ~0.0001–0.0002
only (e.g. Q3: 1.1218 → 1.1219).

### Sanity checks against the lecture notebook

- Dataset `stocks_df_combined_2026_09_18.parquet.brotli`: 240,759 rows × 203 cols;
  202,292 rows after `Date >= 2000-01-01` (same as Module 3 / Module 4).
- Temporal split: train 136,842 (2000-01-03…2018-09-13), validation 32,287
  (2018-09-14…2022-09-15), test 33,163 (2022-09-16…2026-09-18) — identical.
- `NUMERICAL + DUMMIES` is **301** in the original setup (186 numerical + 115 dummies)
  and **208** after removing `Ticker` and `month_wom` (186 + 22) — matching the spec's
  "about 31% fewer".
- Original target positive rate ≈ **58.95%**; after `growth_future_30d > 1.022` it is
  **49.12%**, in line with the spec's "about 50%, down from 60%".
- Hard-coded models, as in the cleaned 2026 workbook:
  `DecisionTreeClassifier(max_depth=10/4/7, random_state=42)` and
  `RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)`.
- `pred0`/`pred1`/`pred2` TEST precision 0.5646 / 0.5787 / 0.5726 — identical to the
  notebook outputs; decision trees reproduce the workbook exactly across sklearn versions,
  only the Random Forest boundary probabilities drift on 1.8.0.

### Q1 note

Dropping the 33 ticker dummies and 60 `month_wom` dummies leaves Month / Weekday /
`ticker_type` only. The Random Forest at depth 13 then produces a smoother probability
distribution, and `pred10_rf_best_rule_55` improves from **1.1195** to **1.136** (~1.1359 at
4 decimals) CAGR on TEST: 3,514 investments, 681 non-empty days, avg 5.16/day, Q75 = 6
($18,000 capital), net revenue $11,971.51. Average investments per day sit slightly above 5,
which is exactly the starting point the spec describes for Q2.

### Q2 note — why 0.56

`tpr_fpr_dataframe` on the Random Forest (not the tree) shows the usual pattern: higher
thresholds cut total trades (`tp+fp`) and eventually lift precision. The threshold ladder on
the reduced-feature setup (TEST, 2.2% round-trip fee, workbook arithmetic):

| threshold | TEST trades | avg/day | CAGR |
| ---: | ---: | ---: | ---: |
| 0.55 (`pred10`) | 3,514 | 5.16 | **1.1359** |
| **0.56 (`pred12`)** | **2,641** | **4.40** | **1.1178** |
| 0.57 | 1,889 | 3.81 | 1.0913 |
| 0.58 | 1,330 | 3.46 | 1.0665 |
| 0.59 | 899 | 3.08 | 1.0614 |
| 0.60 | 583 | 2.78 | 1.0442 |
| 0.61 | 352 | 2.51 | 1.0403 |
| 0.62 | 204 | 2.08 | 1.0259 |
| 0.63 | 123 | 1.68 | 1.0193 |
| 0.64 | 65 | 1.44 | 1.0089 |
| 0.65 (`pred11`) | 31 | 1.35 | 1.0057 |

`pred10` itself averages 5.16 trades/day — above the 5/day cap the question imposes, so among
the newly defined rules the highest-CAGR one below 5 trades/day is **0.56** (CAGR 1.1178, the
spec's "second-best... close to 1.12") — it gives up a little return for a much lighter book.

### Q3 note

Redefining the label to "future 30-day growth > 2.2%" makes the class balance almost even
(49.1% positives). Manual and RF threshold rules lose their edge; the depth-10 tree
`pred5_clf_10` is the best CAGR (**1.1219**) and the spec's "number of trades" is the notebook
line

```
Average investments per day: {int(np.round(sim1_avg_investments_per_day))}
```

which is **8** (`sim1_avg_investments_per_day` = 8.0098). The 4-year TEST investment count is
7,337, which is not among the form options 6 / 7 / 8 / 9. That daily figure is also lower than
the original-target `pred5` run (10.59 trades/day), matching the spec's "fewer trades per day,
since we're now targeting predicted growth above 2.2% instead of above 0%".

### Q4 (exploratory, free text) — ideal trading strategy

I would use the reduced-feature Random Forest (208 features, no ticker or month-week dummies) as the primary signal. Each morning I rank names by predicted probability and keep those with p >= 0.56, the Q2 cutoff that keeps average daily investments below 5 while still delivering CAGR close to 1.12. I invest equally in the top 3 of that list, sizing each ticket as $100 times (p - 0.55) / 0.10 clipped to [0.5, 1.5], so a 0.65 name gets 1.5x a 0.55 name. I only take the trade if the depth-10 tree also predicts 1, because after raising the target to 2.2% future growth that tree had the best CAGR and its splits are macro filters (CPI, DGS10, FEDFUNDS). I cap the book at 5 simultaneous 30-day holds, exit at 30 trading days or on a 10% stop-loss, and skip new entries when S&P 500 30-day growth is below 1 and DGS10 is above 4.

---

## Repository layout

```
homework4/
  solution.py         Main pipeline: Q1+Q2 (reduced dummies) then Q3 (new target)
  README.md           This file
  .gitignore          Excludes data/raw/ and __pycache__/
  data/
    raw/              stocks_df_combined_2026_09_18.parquet.brotli (~144 MB,
                      downloaded at run time, NOT committed)
    outputs/          Analysis results of the last run (CSV, JSON, TXT, committed)
```

### Deliverables (under [`data/outputs/`](data/outputs/))

| File | Description |
| --- | --- |
| [`answers.json`](data/outputs/answers.json) | Final answers in machine-readable form |
| [`run_log.txt`](data/outputs/run_log.txt) | Full console output of the last run |
| [`q1_q2_sim1_results.csv`](data/outputs/q1_q2_sim1_results.csv) | `sim1` table for the reduced-dummy run (Q1/Q2) |
| [`q2_rf_tpr_fpr.csv`](data/outputs/q2_rf_tpr_fpr.csv) | Random Forest `tpr_fpr_dataframe` on TEST |
| [`q2_rf_threshold_sim.csv`](data/outputs/q2_rf_threshold_sim.csv) | CAGR / trades for RF thresholds 0.56…0.64 |
| [`q1_q2_sklearn_sensitivity.csv`](data/outputs/q1_q2_sklearn_sensitivity.csv) | Q1/Q2 under sklearn 1.7.2 vs 1.8.0 (root-cause evidence) |
| [`q3_sim1_results.csv`](data/outputs/q3_sim1_results.csv) | `sim1` table after redefining the 2.2% growth target |

---

## How to run

Requirements: Python 3.10+, `pandas`, `numpy`, `pyarrow`, `scikit-learn == 1.7.x`
(**1.7.2 verified**; 1.8+ shifts RF boundary probabilities and changes Q1/Q2 —
`main()` warns about it), `gdown`.

```bash
# from the repository root
python homework4/solution.py
# or, inside this folder:
python solution.py
```

- The dataset (~144 MB) is downloaded once from Google Drive
  (file id `1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5`, the same file as Module 3)
  into `data/raw/`; re-runs read the local file. If `homework3/data/raw/`
  already has it, the script copies from there.
- Runtime ≈ 2–3 min, dominated by the three Decision Tree fits
  (`max_depth=10/4/7`) on TRAIN+VALIDATION; the two Random Forest fits
  (`n_estimators=100`, `max_depth=13`) take ~5 s each.
- Logistic regression and the neural net from the lecture notebook are skipped:
  they are commented out / precomputed there and are not used for Q1–Q3.

---

## Notes

- Q1 removes `Ticker` and `month_wom` from `CATEGORICAL` **just before**
  `pd.get_dummies()`, leaving `['Month', 'Weekday', 'ticker_type']`.
- Q2 builds `tpr_fpr_dataframe` from `rf_best.predict_proba(X_test)` (not
  `clf_best`) and adds `pred12_rf_best_rule_56` … `pred20_rf_best_rule_64`
  on top of the notebook's `pred10` (0.55) and `pred11` (0.65).
- Q3 overwrites `is_positive_growth_30d_future` with
  `(growth_future_30d > 1.022).astype(int)` on a **fresh** copy of the original
  301-feature setup; every subsequent prediction and the `sim1` simulation
  use that label.
- Simulation (`sim1`) is the workbook's per-row rule: invest $100 on every
  positive TEST prediction, 0.2% round-trip fee; NaN-future-growth rows (last
  ~21 TEST days) are dropped by the gross/net column sums but still count in
  `sim1_count_investments`, avg/Q75 investments per day and the raw fees sum —
  identical to the notebook's accounting; capital = `$100 × 30 × Q75(daily
  investments)`, CAGR over 4 TEST years.
- Features are cleaned `±inf → NaN → 0` once up front (float32/float64 arrays
  give identical trade sets here; **scikit-learn version is the sensitive
  knob**, pin 1.7.x).
- Trees / forests are fitted on TRAIN+VALIDATION and predicted on ALL rows,
  then scored on TEST, same as the notebook.
