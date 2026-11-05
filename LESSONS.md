# Lessons Learned (cross-homework)

A running log of the debugging experiences and root causes behind answer revisions
in this repository. Entries are appended whenever a submitted answer turns out to
be wrong and the reason is understood.

---

## 2026-11: Homework 4 — Q1 CAGR and Q2 RF threshold (Module 4)

**What was wrong.** First submission reported Q1 CAGR **1.1279** for
`pred10_rf_best_rule_55` (reduced-dummy setup) and Q2 threshold **0.57**. The
official answers are **1.136** and **0.56**. Q3 (8 trades) was correct.

**Answer comparison**

| | sklearn 1.8.0 (first run) | sklearn 1.7.2 (course/Colab env.) |
| --- | ---: | ---: |
| pred10 trades (0.55) | 3,756 | 3,514 |
| pred10 avg trades/day | 6.06 | 5.16 |
| pred10 Q75 investments/day | 7 | 6 |
| pred10 net revenue | $12,984 | $11,972 |
| **Q1 CAGR** | 1.1279 ✗ | **1.136** ✓ |
| 0.56 rule avg/day | 5.35 (<5 fails) ✗ | 4.40 (<5 passes) ✓ |
| **Q2 threshold** | 0.57 ✗ | **0.56** ✓ |

**Root cause #1 — scikit-learn version drift in Random Forests.**
The module recipe (dataset snapshot, 208/301 features, temporal split, hard-coded
`RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)`, `>=`
thresholds, `sim1` arithmetic) was identical between our run and the course
environment — manual-prediction gross revenues even matched the committed workbook
outputs to six decimals (`pred1` gross 44,647.366885). The only run-to-run
variable was the library version itself:

- scikit-learn **1.8.0** builds a slightly different Random Forest than **1.7.2**,
  moving a handful of probabilities across the 0.55/0.56 decision boundaries
  (3,756 → 3,514 trades at 0.55).
- Decision *trees* were unaffected: `clf_10/clf_4/clf_best` precisions and trade
  counts reproduce the workbook exactly on both versions (consistent with
  homework 3 matching all tree outputs on 1.8.0).
- `sim1` amplifies tiny trade-set changes: capital is sized as
  `$100 × 30 × Q75(daily investments)`, and a quantile over small-integer daily
  counts flips in whole steps (Q75 7 → 6 moves $21,000 → $18,000), which swings
  CAGR far more than the underlying net revenue moved.

**Root cause #2 — NaN-growth-row accounting in `sim1`.** Secondary, ~0.0001-effect
on CAGR but easy to miss: the notebook computes gross/fees/net as **per-row
columns** summed with pandas (skip-NA) semantics, so rows whose
`growth_future_30d` is NaN (the last ~21 TEST trading days) drop out of the
gross/net sums while they still count in `sim1_count_investments`, avg/Q75 per
day, and the raw fees column. Our first version instead charged fees for every
positive prediction and added that total to the gross. Proof the final version
mirrors the workbook exactly: regenerated original-setup `pred6` row (6,677
trades, gross 22,383.446, fees −1,335.4, net 21,050.646 = exactly 13 NaN rows ×
$0.2) is byte-identical to the committed notebook's embedded output.

**How the discrepancy was pinned down (method that worked).**

1. Fetched the spec and the exact 2026 workbook from GitHub; read its code cells
   and the **embedded Colab outputs** (python 3.13), extracting the actual
   `sim1`/`temporal_split`/feature-list definitions rather than trusting memory.
2. Proved dataset and split correctness from the embedded outputs: manual-rule
   gross revenues and split sizes matched ours to 6 decimals.
3. Built a differential harness: same data, two scikit-learn versions in isolated
   venvs, one feature-set — float32 vs float64 was hypothesised first and
   **falsified by experiment** (identical trade sets on both 1.8.0 and 1.7.2).
   The 1.7.2 venv then reproduced the official values exactly (1.1359 → prints
   1.136; 0.56 picked with avg/day 4.40).
4. Cross-checked the spec's free-text magnitudes against computed values — the
   spec encodes expected magnitudes: Q2's "pred10 trades *slightly more* than 5
   per day" (1.7.2: 5.16 ✓; 1.8.0: 6.06 ✗) and "close to 1.12" for the
   second-best rule (1.1178 ✓; 1.0920 ✗).

**Lessons, generalised for the remaining homeworks**

1. **Pin and state the environment.** Course answers are generated in a specific
   (Colab-era) environment; results are not guaranteed portable across library
   versions — ensembles with randomness at boundaries especially. Homework 5+
   scripts must print library versions and hw04 keeps a `check_sklearn_version()`
   guard (warns on scikit-learn ≥ 1.8; verified on 1.7.2).
2. **Watch out for financially sensitive quantiles.** Any capital rule derived
   from `Q75(daily counts)` is a step function of a few trades; validate answers
   against both the 4-decimal value and its sensitivity band.
3. **Mirror official accounting exactly, including edge rows.** If the source
   sums column-wise with pandas semantics, replicate those semantics verbatim —
   NaN handling differences hide behind rounded outputs. A single byte-matching
   row (pred6 vs the committed workbook output) is stronger evidence than ten
   plausible arguments.
4. **Use the spec's prose as an assertion set.** Words like "slightly higher",
   "about 31% fewer", "down from 60%" are cheap, falsifiable checks against your
   own output; run them before submitting.
5. **Never accept the first plausible cause without a falsification attempt.**
   dtype (float32/64) looked right and was wrong; the version alone explained
   everything.

**Evidence artifacts**

- [`homework4/README.md`](homework4/README.md) — root-cause section with the comparison table.
- [`homework4/data/outputs/q1_q2_sklearn_sensitivity.csv`](homework4/data/outputs/q1_q2_sklearn_sensitivity.csv) — machine-readable 1.7.2 vs 1.8.0 table.
- [`homework4/data/outputs/run_log.txt`](homework4/data/outputs/run_log.txt) — full log of the reproducing (scikit-learn 1.7.2) run.
- [`homework4/solution.py`](homework4/solution.py) — `check_sklearn_version()` guard + workbook-identical `simulate()`.

---

## Older entries

*(empty — homework 1–3 answers were accepted as submitted; cross-check notes live
in each homework's README.)*
