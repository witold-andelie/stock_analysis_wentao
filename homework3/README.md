# Stock Market Analytics Zoomcamp — Homework 3 (Module 3, 2026 cohort)

> 本 README 仅对应 **Homework 3（Module 3 "Modeling", 2026 cohort）**，不覆盖其他作业。

Solutions for Homework 3 (2026 cohort) of the
[Stock Market Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).
Spec: [`cohorts/2026/homework3.md`](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework3.md).
All analysis is reproducible from `solution.py` in this folder, which follows the
[2026 Module 3 notebook](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/03-modeling/%5B2026%5D_Module_03_Colab_Time_Series_Modeling.ipynb)
step by step (same dataset, variable sets, dummies, temporal split and cleaning).

---

## Answers (summary)

| Question | Answer |
| --- | --- |
| **Q1** — Highest absolute correlation of a `<month>_w<week>` dummy with `is_positive_growth_30d_future` | **0.025** — `month_wom_October_w4` (corr = +0.0246). Runner-up: `November_w3` +0.0233, `February_w1` −0.0209. 60 `month_wom` dummies, 115 dummies in total. |
| **Q2** — Precision on TEST of the best NEW hand rule (`pred3` / `pred4`) | **0.588** — `pred3_manual_dgs10_5` = `(DGS10 <= 4.5) & (DGS5 <= 4)`: 9,351 / 15,910. `pred4_manual_dgs10_fedfunds` reaches only 0.503 (8,001 / 15,921). |
| **Q3** — TEST rows where `pred5_clf_10` is correct while `pred0`–`pred4` are all wrong | **1243** (out of 33,163 TEST rows; `clf10` TEST precision 0.591) → 选项 **1171**（最接近，见下方 Q3 note） |
| **Q4** — Optimal `max_depth` (1–12) by TEST precision | **4** — TEST precision **0.629** (4,201 / 6,677), > 0.58 and better than every earlier prediction. |

Machine-readable versions are in
[`data/outputs/answers.json`](data/outputs/answers.json) (committed results of
the last run; regenerated whenever you run the pipeline).

### Sanity checks against the lecture notebook

- Dataset `stocks_df_combined_2026_09_18.parquet.brotli`: 240,759 rows × 203 cols;
  202,292 rows after `Date >= 2000-01-01` (same as the notebook output).
- Temporal split: train 136,842 (2000-01-03…2018-09-13), validation 32,287
  (2018-09-14…2022-09-15), test 33,163 (2022-09-16…2026-09-18) — identical.
- `pred0`/`pred1`/`pred2` TEST precision 0.5646 / 0.5787 / 0.5726
  (520/921, 11,086/19,158, 8,728/15,244) — identical to the notebook.
- With `random_state=42` the fitted tree's top splits are
  `cpi_core_yoy <= 0.023 → DGS10 <= 1.745` and
  `cpi_core_yoy > 0.023 → FEDFUNDS <= 4.96`, exactly the "original" conditions
  quoted in the Q2 hint.

### Q2 note

The spec says "for the rule that **does** make positive predictions on the TEST set"
(wording carried over from 2025). On the 2026 dataset **both** new rules fire on
TEST (15,910 vs 15,921 positive predictions), so the answer is the better of the
two: `pred3` = **0.588**.

### Q3 note — why 1243 and not exactly 1171

Answer options: 271 / 571 / 1171 / 1571. The spec-exact pipeline gives **1243**,
closest to **1171**. The gap comes from tie-breaking inside the depth-10 tree,
not from the pipeline:

- Deep nodes contain few samples, so several features often give exactly the
  same split gain; sklearn keeps the first one in its `random_state`-shuffled
  feature order, which depends on the column order of `X`.
- Re-running with identical data, rules and `random_state=42`, changing **only
  the column order**, gives Q3 counts of 1237–1396 (spec order 1243,
  `month_wom` dummies first 1332, four random orders 1343 / 1396 / 1383 / 1237).
- scikit-learn 1.6.1 (Colab) and 1.8.0 give the same 1243, so the version is
  not the cause. Using the pre-correction `pred3` rule
  `(DGS10 <= 4) & (DGS5 <= 1)` gives 1801 instead, so it is not that either.
- All four options end in "71", i.e. the distractors are offsets of the
  instructor's own value 1171.
- Q4 is unaffected: the depth-4 tree has TEST precision 0.6292 under every
  column order.

### Q3 — uniquely correct predictions per column (TEST)

Generic vectorized version of the "Advanced" task: a prediction is uniquely
correct when it matches the label and every other prediction does not.

| prediction | uniquely correct |
| --- | ---: |
| pred0_manual_cci | 150 |
| pred1_manual_prev_g1 | 380 |
| pred2_manual_prev_g1_and_snp | 2 |
| pred3_manual_dgs10_5 | 1,020 |
| pred4_manual_dgs10_fedfunds | 1,550 |
| **pred5_clf_10** | **1,243** |

### Q4 — depth tuning

Each depth is fitted on TRAIN+VALIDATION and scored on TEST (the spec's
criterion). As an overfitting check, the same depth is also fitted on TRAIN
only and scored on VALIDATION.

| max_depth | TEST precision | TEST accuracy | TEST positives | VAL precision (TRAIN-only fit) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5701 | 0.5701 | 33,163 | 0.6491 |
| 2 | 0.6225 | 0.5168 | 11,760 | 0.6491 |
| 3 | 0.5720 | 0.4918 | 14,263 | 0.6439 |
| **4** | **0.6292** | 0.4819 | 6,677 | **0.6595** |
| 5 | 0.6000 | 0.4750 | 7,487 | 0.6439 |
| 6 | 0.6062 | 0.4728 | 6,704 | 0.6054 |
| 7 | 0.6211 | 0.4787 | 6,682 | 0.6118 |
| 8 | 0.6218 | 0.4990 | 9,410 | 0.6174 |
| 9 | 0.6112 | 0.4906 | 9,053 | 0.6166 |
| 10 | 0.5912 | 0.4774 | 8,628 | 0.6293 |
| 11 | 0.6028 | 0.4767 | 7,552 | 0.6203 |
| 12 | 0.5819 | 0.4720 | 8,518 | 0.6424 |

Depth 4 is best on TEST **and** on the independent TRAIN→VALIDATION check, so the
choice is not just an artefact of tuning on the test set. Deeper trees do not
improve precision (they memorise macro regimes), and accuracy stays below the
always-positive baseline at depth 1 — the model is useful only as a
high-precision filter for "buy" signals, not as a general classifier.

TEST precision of all predictions:

| prediction | positives | TP | precision |
| --- | ---: | ---: | ---: |
| pred0_manual_cci | 921 | 520 | 0.5646 |
| pred1_manual_prev_g1 | 19,158 | 11,086 | 0.5787 |
| pred2_manual_prev_g1_and_snp | 15,244 | 8,728 | 0.5726 |
| pred3_manual_dgs10_5 | 15,910 | 9,351 | 0.5877 |
| pred4_manual_dgs10_fedfunds | 15,921 | 8,001 | 0.5025 |
| pred5_clf_10 | 8,628 | 5,101 | 0.5912 |
| **pred6_clf_best** (depth 4) | 6,677 | 4,201 | **0.6292** |

### Q5 (exploratory, free text) — what data is missing?

> The tree is driven almost entirely by **US macro levels** (`cpi_core_yoy`,
> `FEDFUNDS`, `DGS10`, `DGS1`) and 30–365d growth of oil/gold/BTC/DJI, while
> calendar dummies barely matter (max |corr| of any month-week dummy is 0.025).
> Levels of rates are regime-specific — `pred4` (DGS10 > 4 & FEDFUNDS ≤ 4.795)
> collapses to 0.50 precision on TEST — so I would add:
>
> 1. **Local macro + FX for the non-US two-thirds of the panel.** India and EU
>    stocks are explained only with US macro today. Add RBI repo rate / India CPI
>    (RBI DBIE, FRED `INDCPIALLMINMEI`), ECB deposit rate / euro-area HICP
>    (ECB Data Portal, FRED `ECBDFR`), and FX `INR=X`, `EURUSD=X` from yfinance,
>    because a USD investor's 30d return in RELIANCE.NS or MC.PA depends on them.
> 2. **Curve shape and credit instead of rate levels:** 10y–2y and 10y–3m spreads
>    (FRED `T10Y2Y`, `T10Y3M`), high-yield OAS (`BAMLH0A0HYM2`). Spreads are more
>    stationary across regimes than raw DGS10/FEDFUNDS levels.
> 3. **Risk sentiment / implied volatility:** `^VIX`, `^VSTOXX` (or `^V2TX`),
>    `^INDIAVIX` — 30d forward returns depend on the volatility regime at entry.
> 4. **Earnings and valuation:** days-to-next-earnings, last EPS surprise and
>    forward P/E (yfinance `get_earnings_dates()` / `info`), since every 30-day
>    window can contain an earnings release.
> 5. **Regional relative strength and cross-sectional ranks:** growth vs the
>    stock's own benchmark (`^NSEI`, `^STOXX50E`, `^GSPC`) and the momentum
>    rank among the 33 tickers on each date, which are scale-free and transfer
>    across time better than absolute growth values.
>
> Modeling-wise, I would also replace raw levels with changes/z-scores and use
> walk-forward validation, because one fixed 2022–2026 test window rewards
> whatever macro regime happened to dominate it.

---

## Repository layout

```
homework3/
  solution.py         Main pipeline: data prep + Q1–Q4, prints answers
  README.md           This file
  .gitignore          Excludes data/raw/
  data/
    raw/              stocks_df_combined_2026_09_18.parquet.brotli (~144 MB,
                      downloaded at run time, NOT committed)
    outputs/          Analysis results of the last run (CSV, TXT, JSON, committed)
```

### Deliverables (under [`data/outputs/`](data/outputs/))

| File | Description |
| --- | --- |
| [`answers.json`](data/outputs/answers.json) | Final answers in machine-readable form |
| [`run_log.txt`](data/outputs/run_log.txt) | Full console output of the last run |
| [`q1_month_wom_corr.csv`](data/outputs/q1_month_wom_corr.csv) | All 60 `month_wom` dummies with `corr`, `abs_corr` (sorted) |
| [`q2_hand_rules_precision.csv`](data/outputs/q2_hand_rules_precision.csv) | TEST positives / TP / FP / precision for pred0–pred4 |
| [`q3_uniquely_correct_counts.csv`](data/outputs/q3_uniquely_correct_counts.csv) | Uniquely-correct TEST rows per prediction |
| [`q3_clf10_feature_importance.csv`](data/outputs/q3_clf10_feature_importance.csv), [`tree_clf10_head.txt`](data/outputs/tree_clf10_head.txt) | Top-20 importances and top 3 levels of `clf10` |
| [`q4_depth_tuning.csv`](data/outputs/q4_depth_tuning.csv) | Precision/accuracy per `max_depth` (TEST + VALIDATION check) |
| [`q4_precision_all_preds.csv`](data/outputs/q4_precision_all_preds.csv), [`tree_best_head.txt`](data/outputs/tree_best_head.txt) | pred0–pred6 TEST precision; best (depth 4) tree |

---

## How to run

Requirements: Python 3.10+, `pandas`, `numpy`, `pyarrow`, `scikit-learn`,
`gdown`.

```bash
# from the repository root
python homework3/solution.py
# or, inside this folder:
python solution.py
```

- The dataset (~144 MB) is downloaded once from Google Drive
  (file id `1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5`, the link used in the notebook)
  into `data/raw/`; re-runs read the local file.
- Runtime ≈ 5–10 min, dominated by the 25 decision-tree fits
  (1 for Q3 + 12 depths × 2 fits for Q4).

---

## Notes

- Week of month = `(day - 1) // 7 + 1`; `month_wom = <MonthName>_w<week>`
  (e.g. `October_w1`), added to `CATEGORICAL = ['Month', 'Weekday', 'Ticker',
  'ticker_type', 'month_wom']` before `pd.get_dummies(..., dtype='int32')`.
  The new dummies stay in the feature set for Q3/Q4 (299 features in total).
- Q1 correlates only the `month_wom` dummies with the target; `DataFrame.corr()`
  is pairwise, so the values are identical to the notebook's full
  `NUMERICAL + DUMMIES + TO_PREDICT` matrix.
- Trees: `DecisionTreeClassifier(max_depth=d, random_state=42)`, fitted on
  TRAIN+VALIDATION after the notebook cleaning (`±inf → NaN → 0`), then
  predicted on all rows (`pred5_clf_10`, `pred6_clf_best`). Features are cast to
  float32 once up front — sklearn trees do this internally anyway, so results
  are unchanged.
- Precision = TP / (TP + FP) on TEST rows where the prediction is 1, the same
  definition as the notebook's `is_correct_*` loop.
