"""
Stock Market Analytics Zoomcamp - Module 3 Homework (2026 cohort) solution.

Covers (spec: cohorts/2026/homework3.md):
  Q1: Month + Week-of-Month dummies (e.g. 'October_w1'), highest |corr|
      with is_positive_growth_30d_future among the month_wom dummies
  Q2: New hand rules
        pred3_manual_dgs10_5        = (DGS10 <= 4.5) & (DGS5 <= 4)
        pred4_manual_dgs10_fedfunds = (DGS10 > 4) & (FEDFUNDS <= 4.795)
      -> precision on TEST for the new rule(s) with positive predictions
  Q3: DecisionTreeClassifier(max_depth=10, random_state=42) fitted on
      TRAIN+VALIDATION, predicted on ALL rows -> pred5_clf_10;
      count TEST rows where only pred5 is correct (pred0..pred4 all wrong)
  Q4: max_depth 1..12 tuning (random_state=42, fit on TRAIN+VALIDATION),
      best TEST precision -> pred6_clf_best
  Q5: free text (see README.md)

The pipeline mirrors the 2026 Module 3 notebook
(03-modeling/[2026]_Module_03_Colab_Time_Series_Modeling.ipynb):
same dataset, variable sets, 2000-01-01 truncation, dummies, temporal
70/15/15 split, inf->NaN->0 cleaning. The dataset is downloaded once from
Google Drive into data/raw/; analysis outputs go to data/outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

# ----------------------------------------------------------------------------
# paths / helpers
# ----------------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "data"
RAW = DATA / "raw"
OUT = DATA / "outputs"
for p in (RAW, OUT):
    p.mkdir(parents=True, exist_ok=True)

DATASET_FILE = "stocks_df_combined_2026_09_18.parquet.brotli"
GDRIVE_ID = "1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5"

TARGET = "is_positive_growth_30d_future"
RANDOM_STATE = 42
DEPTHS = range(1, 13)

ANSWERS: dict = {}


def _num(x, nd: int = 4):
    """Round floats for answers.json; map NaN/inf -> None (valid JSON)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, nd)


def header(title: str, first: bool = False):
    print(("" if first else "\n") + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# data: download + notebook section 0 (variable sets, dummies, split)
# ----------------------------------------------------------------------------
def load_full() -> pd.DataFrame:
    path = RAW / DATASET_FILE
    if not path.exists():
        try:
            import gdown
        except ImportError as e:
            raise RuntimeError("gdown is required: pip install gdown") from e
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}", str(path), quiet=False
        )
    df = pd.read_parquet(path)
    print(f"  [data] {DATASET_FILE}: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15):
    """Same boundaries as the notebook's temporal_split(), vectorized."""
    span_days = (max_date - min_date).days
    train_end = min_date + pd.Timedelta(days=span_days * train_prop)
    val_end = train_end + pd.Timedelta(days=span_days * val_prop)
    df["split"] = np.select(
        [df["Date"] <= train_end, df["Date"] <= val_end],
        ["train", "validation"],
        default="test",
    )
    return df


def prepare():
    df_full = load_full()

    GROWTH = [g for g in df_full.keys() if g.startswith("growth_") and "future" not in g]
    TECHNICAL_INDICATORS = [
        "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc",
        "bop", "cci", "cmo", "dx", "macd", "macdsignal", "macdhist", "macd_ext",
        "macdsignal_ext", "macdhist_ext", "macd_fix", "macdsignal_fix",
        "macdhist_fix", "mfi", "minus_di", "mom", "plus_di", "dm", "ppo",
        "roc", "rocp", "rocr", "rocr100", "rsi", "slowk", "slowd", "fastk",
        "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr",
        "ad", "adosc", "obv", "atr", "natr", "ht_dcperiod", "ht_dcphase",
        "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine", "ht_sine_leadsine",
        "ht_trendmod", "avgprice", "medprice", "typprice", "wclprice",
    ]
    TECHNICAL_PATTERNS = [g for g in df_full.keys() if "cdl" in g]
    CUSTOM_NUMERICAL = [
        "SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative",
        "volatility", "ln_volume",
    ]
    MACRO = [
        "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
        "FEDFUNDS", "DGS1", "DGS5", "DGS10",
    ]
    NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

    # a small hack to avoid "divide by zero encountered in log" (as in notebook)
    df_full["ln_volume"] = np.log(df_full.Volume.replace(0, np.nan).fillna(1e-9))

    # truncated df_full with 25 years of data
    df = df_full[df_full.Date >= "2000-01-01"].copy()
    print(f"  [data] Date >= 2000-01-01: {len(df)} rows, {df.Ticker.nunique()} tickers")

    # Q1 feature: <MonthName>_w<week-of-month>, week = (day - 1) // 7 + 1
    df["month_wom"] = (
        df["Date"].dt.month_name() + "_w" + ((df["Date"].dt.day - 1) // 7 + 1).astype(str)
    )
    df["Weekday"] = df["Weekday"].astype(str)
    df["Month"] = df["Month"].dt.month.astype(str)

    CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]
    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
    DUMMIES = dummy_variables.keys().to_list()
    MONTH_WOM_DUMMIES = [d for d in DUMMIES if d.startswith("month_wom_")]
    print(f"  [data] dummies: {len(DUMMIES)} total, {len(MONTH_WOM_DUMMIES)} month_wom")

    new_df = pd.concat([df, dummy_variables], axis=1)
    new_df = temporal_split(new_df, new_df.Date.min(), new_df.Date.max())
    split_info = new_df.groupby("split")["Date"].agg(["min", "max", "count"])
    print("  [data] temporal split:")
    print("    " + split_info.to_string().replace("\n", "\n    "))

    return new_df, NUMERICAL, DUMMIES, MONTH_WOM_DUMMIES


def precision_on(df: pd.DataFrame, pred_col: str, split: str = "test"):
    """TP / (TP + FP) on one split; NaN if the column never predicts 1."""
    sub = df[df.split == split]
    pos = sub[pred_col] == 1
    n_pos = int(pos.sum())
    tp = int((pos & (sub[TARGET] == 1)).sum())
    return (tp / n_pos if n_pos else np.nan), n_pos, tp


# ----------------------------------------------------------------------------
# Q1 - month_wom dummies correlation
# ----------------------------------------------------------------------------
def q1(new_df, month_wom_dummies):
    header("Q1 | Month + Week-of-Month dummies: max |corr| with target", first=True)

    # DataFrame.corr() is pairwise, so the (dummy, target) values equal those
    # of the notebook's full NUMERICAL+DUMMIES+TO_PREDICT correlation matrix.
    corr = new_df[month_wom_dummies + [TARGET]].corr()[TARGET].drop(TARGET)
    corr_df = corr.to_frame("corr")
    corr_df["abs_corr"] = corr_df["corr"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False)
    corr_df.to_csv(OUT / "q1_month_wom_corr.csv", index_label="dummy")

    print("  top-10 month_wom dummies by |corr|:")
    print("    " + corr_df.head(10).round(4).to_string().replace("\n", "\n    "))
    top = corr_df.index[0]
    best = float(corr_df["abs_corr"].iloc[0])
    print(f"\n  >>> Q1 answer: {top} |corr| = {best:.3f} "
          f"(signed {corr_df['corr'].iloc[0]:+.4f})")
    ANSWERS["q1"] = {
        "dummy": top,
        "abs_corr": round(best, 3),
        "corr_signed": _num(corr_df["corr"].iloc[0]),
        "n_month_wom_dummies": len(month_wom_dummies),
    }


# ----------------------------------------------------------------------------
# Q2 - hand rules pred0..pred4
# ----------------------------------------------------------------------------
def q2(new_df):
    header("Q2 | Hand rules: precision on TEST")

    new_df["pred0_manual_cci"] = (new_df.cci > 200).astype(int)
    new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)
    new_df["pred2_manual_prev_g1_and_snp"] = (
        (new_df["growth_30d"] > 1) & (new_df["growth_snp500_30d"] > 1)
    ).astype(int)
    new_df["pred3_manual_dgs10_5"] = (
        (new_df["DGS10"] <= 4.5) & (new_df["DGS5"] <= 4)
    ).astype(int)
    new_df["pred4_manual_dgs10_fedfunds"] = (
        (new_df["DGS10"] > 4) & (new_df["FEDFUNDS"] <= 4.795)
    ).astype(int)

    hand = [k for k in new_df.keys() if k.startswith("pred")]
    rows = []
    for pred in hand:
        prec, n_pos, tp = precision_on(new_df, pred)
        rows.append({"prediction": pred, "test_positive_preds": n_pos,
                     "tp": tp, "fp": n_pos - tp, "precision": prec})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "q2_hand_rules_precision.csv", index=False)
    print("    " + res.round(4).to_string(index=False).replace("\n", "\n    "))

    new = res[res.prediction.isin(["pred3_manual_dgs10_5", "pred4_manual_dgs10_fedfunds"])]
    best = new.dropna(subset=["precision"]).sort_values("precision").iloc[-1]
    print(f"\n  >>> Q2 answer: {best.prediction} precision = {best.precision:.3f} "
          f"({best.tp}/{best.test_positive_preds})")
    ANSWERS["q2"] = {
        "best_new_rule": best.prediction,
        "precision": round(float(best.precision), 3),
        "new_rules": {
            r.prediction: {"test_positive_preds": int(r.test_positive_preds),
                           "precision": _num(r.precision)}
            for r in new.itertuples()
        },
    }
    return hand


# ----------------------------------------------------------------------------
# Decision trees (shared by Q3/Q4)
# ----------------------------------------------------------------------------
def build_xy(new_df, features):
    """Notebook cleaning (+-inf -> NaN -> 0) applied to ALL rows at once.

    Cast to float32 up front: sklearn trees convert X to float32 internally
    anyway, so results are identical and we avoid a copy on every fit.
    """
    X_all = (
        new_df[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    )
    return X_all, new_df[TARGET]


def fit_tree(X, y, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    clf.fit(X, y)
    return clf


def uniquely_correct(df, preds):
    """Per-prediction flag: this pred is correct while all the others are wrong."""
    correct = pd.DataFrame({p: df[p] == df[TARGET] for p in preds})
    n_correct = correct.sum(axis=1)
    return pd.DataFrame({p: correct[p] & (n_correct == 1) for p in preds})


# ----------------------------------------------------------------------------
# Q3 - pred5_clf_10 unique correct predictions
# ----------------------------------------------------------------------------
def q3(new_df, X_all, y_all, trainval, hand, features):
    header("Q3 | pred5_clf_10: TEST rows where only the tree is correct")

    clf10 = fit_tree(X_all[trainval], y_all[trainval], 10)
    new_df["pred5_clf_10"] = clf10.predict(X_all)
    prec, n_pos, tp = precision_on(new_df, "pred5_clf_10")
    print(f"  clf10: depth={clf10.get_depth()}, TEST precision={prec:.4f} ({tp}/{n_pos})")

    is_correct = {p: (new_df[p] == new_df[TARGET]).astype(int) for p in hand + ["pred5_clf_10"]}
    new_df["only_pred5_is_correct"] = (
        (is_correct["pred5_clf_10"] == 1)
        & (sum(is_correct[p] for p in hand) == 0)
    ).astype(int)
    test = new_df.split == "test"
    answer = int(new_df.loc[test, "only_pred5_is_correct"].sum())

    # Advanced: the same flag for every prediction, without hardcoding
    uniq = uniquely_correct(new_df[test], hand + ["pred5_clf_10"]).sum()
    uniq.rename("test_uniquely_correct").to_csv(OUT / "q3_uniquely_correct_counts.csv",
                                               index_label="prediction")
    print("  uniquely-correct TEST rows per prediction (generic function):")
    print("    " + uniq.to_string().replace("\n", "\n    "))
    assert uniq["pred5_clf_10"] == answer

    (OUT / "tree_clf10_head.txt").write_text(
        export_text(clf10, feature_names=features, max_depth=3), encoding="utf-8"
    )
    imp = pd.Series(clf10.feature_importances_, index=features).sort_values(ascending=False)
    imp.head(20).rename("importance").to_csv(OUT / "q3_clf10_feature_importance.csv",
                                             index_label="feature")
    print(f"  top features: {', '.join(imp.head(5).index)}")

    print(f"\n  >>> Q3 answer: {answer} TEST rows where only pred5_clf_10 is correct "
          f"(of {int(test.sum())} TEST rows)")
    ANSWERS["q3"] = {
        "only_pred5_correct_test": answer,
        "n_test_rows": int(test.sum()),
        "clf10_test_precision": _num(prec),
    }


# ----------------------------------------------------------------------------
# Q4 - max_depth tuning
# ----------------------------------------------------------------------------
def q4(new_df, X_all, y_all, trainval, hand, features):
    header("Q4 | DecisionTree max_depth tuning (1..12), TEST precision")

    test = (new_df.split == "test").to_numpy()
    train_only = (new_df.split == "train").to_numpy()
    val = (new_df.split == "validation").to_numpy()
    y_test = y_all[test]

    rows, models = [], {}
    for d in DEPTHS:
        clf = fit_tree(X_all[trainval], y_all[trainval], d)
        models[d] = clf
        y_pred = clf.predict(X_all[test])
        # overfitting check: fit on TRAIN only, score on VALIDATION
        clf_tr = fit_tree(X_all[train_only], y_all[train_only], d)
        y_val_pred = clf_tr.predict(X_all[val])
        rows.append({
            "max_depth": d,
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_positive_preds": int(y_pred.sum()),
            "val_precision_train_only": precision_score(
                y_all[val], y_val_pred, zero_division=0),
        })
        print(f"  depth {d:2d}: test precision={rows[-1]['test_precision']:.4f} "
              f"accuracy={rows[-1]['test_accuracy']:.4f} "
              f"positives={rows[-1]['test_positive_preds']:5d} | "
              f"val precision (train-only fit)={rows[-1]['val_precision_train_only']:.4f}")
    tuning = pd.DataFrame(rows)
    tuning.to_csv(OUT / "q4_depth_tuning.csv", index=False)

    best_depth = int(tuning.loc[tuning.test_precision.idxmax(), "max_depth"])
    best_prec = float(tuning.test_precision.max())
    best = models[best_depth]
    new_df["pred6_clf_best"] = best.predict(X_all)

    (OUT / "tree_best_head.txt").write_text(
        export_text(best, feature_names=features, max_depth=3), encoding="utf-8"
    )

    preds = hand + ["pred5_clf_10", "pred6_clf_best"]
    cmp = []
    for p in preds:
        prec, n_pos, tp = precision_on(new_df, p)
        cmp.append({"prediction": p, "test_positive_preds": n_pos, "tp": tp,
                    "precision": prec})
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(OUT / "q4_precision_all_preds.csv", index=False)
    print("\n  TEST precision, all predictions:")
    print("    " + cmp.round(4).to_string(index=False).replace("\n", "\n    "))

    print(f"\n  >>> Q4 answer: best_max_depth = {best_depth} "
          f"(TEST precision {best_prec:.4f})")
    ANSWERS["q4"] = {
        "best_max_depth": best_depth,
        "best_test_precision": _num(best_prec),
        "test_precision_by_depth": {
            int(r.max_depth): _num(r.test_precision) for r in tuning.itertuples()
        },
    }


# ----------------------------------------------------------------------------
def main():
    header("Data | Module 3 dataset + notebook section 0", first=True)
    new_df, NUMERICAL, DUMMIES, MONTH_WOM_DUMMIES = prepare()

    q1(new_df, MONTH_WOM_DUMMIES)
    hand = q2(new_df)

    features = NUMERICAL + DUMMIES
    X_all, y_all = build_xy(new_df, features)
    trainval = new_df.split.isin(["train", "validation"]).to_numpy()
    print(f"\n  [trees] features={len(features)}, TRAIN+VALIDATION rows={int(trainval.sum())}")

    q3(new_df, X_all, y_all, trainval, hand, features)
    q4(new_df, X_all, y_all, trainval, hand, features)

    with open(OUT / "answers.json", "w", encoding="utf-8") as f:
        json.dump(ANSWERS, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("\nDone. Answers written to:", (OUT / "answers.json").resolve())
    print(json.dumps(ANSWERS, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
