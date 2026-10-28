# Stock Market Analytics Zoomcamp — Homework 4 (Module 4, 2026 cohort)

Solutions for Homework 4 (2026 cohort) of the
[Stock Market Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).
Spec: [`cohorts/2026/homework4.md`](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework4.md).
The pipeline in `solution.py` follows the
[2026 Module 4 notebook](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/04-trading-strategy-and-simulation/%5B2026%5D_Module_04_Colab_Trading_Simulations.ipynb)
(same dataset, variable sets, dummies, temporal split, hard-coded tree / Random Forest hyperparameters, and `sim1` $100-per-positive-prediction simulation).

Each question is a **fresh run** of the notebook setup, as the spec requires: Q1/Q2 share the shortened dummy list; Q3 starts again from the original dummies and a redefined target.

---

## Answers (summary)

| Question | Answer |
| --- | --- |
| **Q1** — CAGR of `pred10_rf_best_rule_55` after dropping `Ticker` and `month_wom` dummies | **1.1279** (208 features; avg 6.06 trades/day, 3,756 TEST trades). Slightly above the original 301-feature CAGR **1.1195**. |
| **Q2** — RF probability threshold of the second-best rule with avg. daily investments **< 5** | **0.57** — `pred13_rf_best_rule_57`, CAGR **1.092**, 4.85 trades/day. `pred10` (0.55) is still best on CAGR (1.128) but trades 6.06 times/day. |
| **Q3** — number of trades for `pred5_clf_10` after redefining the target as `growth_future_30d > 1.022` | **8** — notebook `Average investments per day` = 8.01 → `int(round(8.01)) = 8` (options 6 / 7 / 8 / 9). 7,337 TEST investments over 4 years; CAGR 1.122, also the best CAGR on the new target. |

Machine-readable versions are in
[`data/outputs/answers.json`](data/outputs/answers.json) (committed results of
the last run; regenerated whenever you run the pipeline).

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

### Q1 note

Dropping the 33 ticker dummies and 60 `month_wom` dummies leaves Month / Weekday /
`ticker_type` only. The Random Forest at depth 13 then produces a smoother
probability distribution, and `pred10_rf_best_rule_55` improves from **1.1195** to
**1.1279** CAGR on TEST. Average investments per day stay a little above 5
(6.06), which is the starting point for Q2.

### Q2 note — why 0.57

`tpr_fpr_dataframe` on the Random Forest (not the tree) shows the usual pattern:
higher thresholds cut total trades (`tp+fp`) and eventually lift precision.

| threshold | TEST trades | avg / day | CAGR |
| ---: | ---: | ---: | ---: |
| 0.55 (`pred10`) | 3,756 | 6.06 | **1.1279** |
| 0.56 | 2,780 | 5.35 | 1.1013 |
| **0.57** | **2,086** | **4.85** | **1.0920** |
| 0.58 | 1,519 | 4.49 | 1.0671 |
| 0.61 | 522 | 3.60 | 1.0543 |
| 0.65 (`pred11`) | 92 | 3.29 | 1.0094 |

0.56 is still above 5 trades/day. Among rules with **avg. daily investments < 5**,
**0.57** has the highest CAGR (1.092, close to 1.12) and is the spec's
"second-best" model: it gives up a little return for a cheaper time budget.

### Q3 note

Redefining the label to "future 30-day growth > 2.2%" makes the class balance
almost even (49.1% positives). Manual and RF threshold rules lose their edge;
the depth-10 tree `pred5_clf_10` is the best CAGR (**1.1218**) and the spec's
"number of trades" is the notebook line

```
Average investments per day: {int(np.round(sim1_avg_investments_per_day))}
```

which is **8** (`sim1_avg_investments_per_day` = 8.0098). The 4-year TEST
investment count is 7,337, which is not among the form options 6 / 7 / 8 / 9.
That daily figure is also lower than the original-target `pred5` run (10.59
trades/day), matching the spec's "fewer trades per day, since we're now
targeting predicted growth above 2.2% instead of above 0%".

### Q4 (exploratory, free text) — ideal trading strategy

I would use the reduced-feature Random Forest (208 features, no ticker or month-week dummies) as the primary signal. Each morning I rank names by predicted probability and keep those with p >= 0.57, the Q2 cutoff that keeps average daily investments below 5 while still delivering CAGR close to 1.12. I invest equally in the top 3 of that list, sizing each ticket as $100 times (p - 0.55) / 0.10 clipped to [0.5, 1.5], so a 0.65 name gets 1.5x a 0.55 name. I only take the trade if the depth-10 tree also predicts 1, because after raising the target to 2.2% future growth that tree had the best CAGR and its splits are macro filters (CPI, DGS10, FEDFUNDS). I cap the book at 5 simultaneous 30-day holds, exit at 30 trading days or on a 10% stop-loss, and skip new entries when S&P 500 30-day growth is below 1 and DGS10 is above 4.

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
| [`q3_sim1_results.csv`](data/outputs/q3_sim1_results.csv) | `sim1` table after redefining the 2.2% growth target |

---

## How to run

Requirements: Python 3.10+, `pandas`, `numpy`, `pyarrow`, `scikit-learn`,
`gdown`.

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
- Runtime ≈ 2–3 min, dominated by two Random Forest fits
  (`n_estimators=100`, `max_depth=13`) on TRAIN+VALIDATION.
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
- Simulation (`sim1`) is the notebook's vectorised rule: invest $100 on every
  positive TEST prediction, 0.2% round-trip fee, capital =
  `$100 × 30 × Q75(daily investments)`, CAGR over 4 TEST years.
- Trees / forests are fitted on TRAIN+VALIDATION and predicted on ALL rows,
  then scored on TEST, same as the notebook. Features are cleaned
  `±inf → NaN → 0` once up front.
