"""
Stock Market Analytics Zoomcamp - Module 4 Homework (2026 cohort) solution.

Covers (spec: cohorts/2026/homework4.md), following
[2026]_Module_04_Colab_Trading_Simulations.ipynb:

  Q1: Drop Ticker + month_wom from CATEGORICAL before get_dummies
      (301 -> 208 features) and report CAGR of pred10_rf_best_rule_55.
  Q2: On the Q1 setup, add RF probability rules for thresholds 0.56..0.64
      and pick the second-best rule: best CAGR among those with
      avg. daily investments < 5.
  Q3: Fresh original dummy set; redefine the target as
      is_positive_growth_30d_future = (growth_future_30d > 1.022);
      report the number of TEST trades for pred5_clf_10.
  Q4: free-text strategy (see README.md).

The dataset is the same parquet as Module 3 / the 2026 notebook
(Google Drive id 1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5). Logistic regression
and the neural net from the notebook are skipped: they are commented-out /
precomputed there and are not needed for Q1-Q3. Trees and Random Forest
use the notebook's hard-coded hyperparameters (no retuning).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# paths / constants
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
FEE_RATE = 0.002  # 0.1% buy + 0.1% sell
HOLD_DAYS = 30
TEST_YEARS = 4
RF_MAX_DEPTH = 13
RF_N_ESTIMATORS = 100
CLF_BEST_DEPTH = 7  # notebook "manual second best"

ANSWERS: dict = {}


def _num(x, nd: int = 4):
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
# data
# ----------------------------------------------------------------------------
def load_full() -> pd.DataFrame:
    path = RAW / DATASET_FILE
    if not path.exists():
        sibling = BASE.parent / "homework3" / "data" / "raw" / DATASET_FILE
        if sibling.exists():
            path.write_bytes(sibling.read_bytes())
            print(f"  [data] copied {DATASET_FILE} from homework3/data/raw/")
        else:
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


def feature_lists(df_full: pd.DataFrame):
    """NUMERICAL / CATEGORICAL lists, identical to the 2026 Module 4 notebook."""
    growth = [g for g in df_full.keys() if g.startswith("growth_") and "future" not in g]
    technical_indicators = [
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
    technical_patterns = [g for g in df_full.keys() if "cdl" in g]
    custom_numerical = [
        "SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative",
        "volatility", "ln_volume", "div_payout", "stock_split",
    ]
    macro = [
        "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
        "FEDFUNDS", "DGS1", "DGS5", "DGS10",
    ]
    numerical = growth + technical_indicators + technical_patterns + custom_numerical + macro
    categorical = ["Month", "Weekday", "Ticker", "ticker_type"]
    return numerical, categorical


def prepare(df_full: pd.DataFrame, *, drop_ticker_month_wom: bool, growth_threshold: float | None):
    """Build new_df + feature lists.

    drop_ticker_month_wom=True is the Q1/Q2 setup (208 features).
    growth_threshold is None to keep the parquet target (Q1/Q2); pass 1.022 for Q3.
    """
    numerical, categorical = feature_lists(df_full)

    df = df_full[df_full.Date >= "2000-01-01"].copy()
    df["ln_volume"] = np.log(df["Volume"] + 1e-6)
    df["div_payout"] = (df["Dividends"] > 0).astype(int)
    df["stock_split"] = (df["Stock Splits"] > 0).astype(int)

    if growth_threshold is not None:
        df[TARGET] = (df["growth_future_30d"] > growth_threshold).astype(int)

    if pd.api.types.is_datetime64_any_dtype(df["Month"]):
        df["Month"] = df["Month"].dt.strftime("%B")
    else:
        df["Month"] = pd.to_datetime(df["Month"], format="%m").dt.strftime("%B")
    df["Weekday"] = df["Weekday"].astype(str)

    wom = ((df["Date"].dt.day - 1) // 7 + 1).astype(str)
    df["month_wom"] = df["Month"] + "_w" + wom
    categorical = list(categorical) + ["month_wom"]

    if drop_ticker_month_wom:
        categorical = [c for c in categorical if c not in ("Ticker", "month_wom")]

    dummy_variables = pd.get_dummies(df[categorical], dtype="int32")
    dummies = dummy_variables.keys().to_list()
    n_features = len(numerical) + len(dummies)
    print(
        f"  [data] Date>=2000: {len(df)} rows, {df.Ticker.nunique()} tickers; "
        f"CATEGORICAL={categorical}; NUMERICAL+DUMMIES={n_features} "
        f"({len(numerical)} num + {len(dummies)} dummies)"
    )
    print(f"  [data] target {TARGET}=1 rate: {df[TARGET].mean():.4f}")

    new_df = pd.concat([df, dummy_variables], axis=1)
    new_df = temporal_split(new_df, new_df.Date.min(), new_df.Date.max())
    split_info = new_df.groupby("split")["Date"].agg(["min", "max", "count"])
    print("  [data] temporal split:")
    print("    " + split_info.to_string().replace("\n", "\n    "))
    return new_df, numerical, dummies, n_features


def add_manual_preds(new_df: pd.DataFrame) -> None:
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


def build_xy(new_df: pd.DataFrame, features: list[str]):
    """Notebook cleaning (+-inf -> NaN -> 0) on ALL rows, once."""
    X_all = (
        new_df[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    )
    y_all = new_df[TARGET].fillna(0).astype(int)
    return X_all, y_all


def fit_models(new_df: pd.DataFrame, features: list[str], *, fit_rf: bool = True):
    """Fit the notebook's hard-coded trees + Random Forest on TRAIN+VALIDATION."""
    X_all, y_all = build_xy(new_df, features)
    trainval = new_df.split.isin(["train", "validation"]).to_numpy()
    test = (new_df.split == "test").to_numpy()

    t0 = time.time()
    clf10 = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    clf10.fit(X_all[trainval], y_all[trainval])
    new_df["pred5_clf_10"] = clf10.predict(X_all)

    clf4 = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    clf4.fit(X_all[trainval], y_all[trainval])
    new_df["pred6_clf_4"] = clf4.predict(X_all)

    clf_best = DecisionTreeClassifier(max_depth=CLF_BEST_DEPTH, random_state=RANDOM_STATE)
    clf_best.fit(X_all[trainval], y_all[trainval])
    new_df["pred7_clf_second_best"] = clf_best.predict(X_all)

    tree_proba = clf_best.predict_proba(X_all)[:, 1]
    new_df["pred8_clf_second_best_rule_56"] = (tree_proba >= 0.56).astype(int)
    new_df["pred9_clf_second_best_rule_66"] = (tree_proba >= 0.66).astype(int)
    print(f"  [fit] trees depth 10/4/{CLF_BEST_DEPTH} in {time.time() - t0:.1f}s")

    rf_best = None
    rf_proba = None
    if fit_rf:
        t0 = time.time()
        rf_best = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf_best.fit(X_all[trainval], y_all[trainval])
        rf_proba = rf_best.predict_proba(X_all)[:, 1]
        new_df["pred10_rf_best_rule_55"] = (rf_proba >= 0.55).astype(int)
        new_df["pred11_rf_best_rule_65"] = (rf_proba >= 0.65).astype(int)
        print(
            f"  [fit] RandomForest n_estimators={RF_N_ESTIMATORS} "
            f"max_depth={RF_MAX_DEPTH} in {time.time() - t0:.1f}s"
        )

    return {
        "X_all": X_all,
        "y_all": y_all,
        "trainval": trainval,
        "test": test,
        "clf10": clf10,
        "clf_best": clf_best,
        "rf_best": rf_best,
        "rf_proba": rf_proba,
        "tree_proba": tree_proba,
    }


def add_rf_thresholds(new_df: pd.DataFrame, rf_proba: np.ndarray, thresholds: list[int], start_idx: int = 12):
    """pred{start_idx}_rf_best_rule_{tt} for each threshold in hundredths (56..64)."""
    cols = []
    for i, hundredths in enumerate(thresholds, start=start_idx):
        col = f"pred{i}_rf_best_rule_{hundredths}"
        new_df[col] = (rf_proba >= hundredths / 100.0).astype(int)
        cols.append(col)
    return cols


def tpr_fpr_dataframe(y_true, y_pred, only_even=False) -> pd.DataFrame:
    """Same helper as the notebook (thresholds 0, 0.01, ... or even steps)."""
    thresholds = np.linspace(0, 1, 51 if only_even else 101)
    scores = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for t in thresholds:
        actual_positive = y_true == 1
        actual_negative = y_true == 0
        predict_positive = y_pred >= t
        predict_negative = y_pred < t
        tp = int((predict_positive & actual_positive).sum())
        tn = int((predict_negative & actual_negative).sum())
        fp = int((predict_positive & actual_negative).sum())
        fn = int((predict_negative & actual_positive).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))
    df_scores = pd.DataFrame(
        scores,
        columns=["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "accuracy", "f1_score"],
    )
    df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)
    return df_scores


def simulate(new_df: pd.DataFrame) -> pd.DataFrame:
    """Notebook sim1: invest $100 on every positive TEST prediction, 0.2% round-trip fee."""
    preds = [k for k in new_df.columns if k.startswith("pred")]
    rows = []
    test = new_df.split == "test"
    growth = new_df["growth_future_30d"]
    for pred in preds:
        pos = test & (new_df[pred] == 1)
        n = int(pos.sum())
        gross = float((new_df.loc[pos, pred] * 100 * (growth[pos] - 1)).sum()) if n else 0.0
        fees = float((-n * 100 * FEE_RATE) if n else 0.0)
        net = gross + fees
        daily = new_df.loc[pos].groupby("Date")[pred].count() if n else pd.Series(dtype=float)
        avg_per_day = float(daily.mean()) if len(daily) else 0.0
        q75_per_day = float(daily.quantile(0.75)) if len(daily) else 0.0
        capital = 100 * HOLD_DAYS * q75_per_day
        cagr = ((capital + net) / capital) ** (1 / TEST_YEARS) if capital else np.nan
        rows.append(
            {
                "prediction": pred,
                "sim1_count_investments": n,
                "sim1_gross_rev": gross,
                "sim1_fees": fees,
                "sim1_net_rev": net,
                "sim1_average_net_revenue": (net / n if n else np.nan),
                "sim1_avg_investments_per_day": avg_per_day,
                "sim1_q75_investments_per_day": q75_per_day,
                "sim1_capital": capital,
                "sim1_CAGR": cagr,
            }
        )
    res = pd.DataFrame(rows)
    res["sim1_growth_capital_4y"] = (res.sim1_net_rev + res.sim1_capital) / res.sim1_capital
    return res


def print_sim(res: pd.DataFrame, cols: list[str] | None = None):
    show = res if cols is None else res[res.prediction.isin(cols)]
    fmt = show.copy()
    for c in ("sim1_gross_rev", "sim1_fees", "sim1_net_rev", "sim1_capital"):
        if c in fmt:
            fmt[c] = fmt[c].round(2)
    for c in ("sim1_CAGR", "sim1_avg_investments_per_day", "sim1_average_net_revenue"):
        if c in fmt:
            fmt[c] = fmt[c].round(4)
    print("    " + fmt.to_string(index=False).replace("\n", "\n    "))


# ----------------------------------------------------------------------------
# Q1 + Q2  (reduced dummy set, original target)
# ----------------------------------------------------------------------------
def q1_q2(df_full: pd.DataFrame):
    header("Q1+Q2 | fewer dummies (drop Ticker + month_wom), original target", first=True)

    new_df, numerical, dummies, n_features = prepare(
        df_full, drop_ticker_month_wom=True, growth_threshold=None
    )
    if n_features != 208:
        print(f"  [warn] expected 208 features, got {n_features}")
    add_manual_preds(new_df)
    fitted = fit_models(new_df, numerical + dummies, fit_rf=True)

    extra_cols = add_rf_thresholds(new_df, fitted["rf_proba"], list(range(56, 65)))
    print(f"  [q2] extra RF rules: {extra_cols}")

    y_test = fitted["y_all"][fitted["test"]]
    rf_test_proba = fitted["rf_proba"][fitted["test"]]
    df_scores = tpr_fpr_dataframe(y_test, rf_test_proba, only_even=True)
    df_scores.to_csv(OUT / "q2_rf_tpr_fpr.csv", index=False)
    band = df_scores[(df_scores.threshold >= 0.50) & (df_scores.threshold <= 0.70)]
    print("  [q2] RF df_scores (threshold 0.50..0.70, even steps):")
    print(
        "    "
        + band[["threshold", "tp", "fp", "precision", "recall"]]
        .round(4)
        .to_string(index=False)
        .replace("\n", "\n    ")
    )

    res = simulate(new_df)
    res.to_csv(OUT / "q1_q2_sim1_results.csv", index=False)
    print("\n  simulation (all predictors):")
    print_sim(res)

    row10 = res.loc[res.prediction == "pred10_rf_best_rule_55"].iloc[0]
    cagr10 = float(row10.sim1_CAGR)
    print(
        f"\n  >>> Q1 answer: pred10_rf_best_rule_55 CAGR = {cagr10:.4f} "
        f"(avg trades/day {row10.sim1_avg_investments_per_day:.3f}, "
        f"n_trades={int(row10.sim1_count_investments)}; "
        f"original notebook CAGR was 1.1195; features={n_features})"
    )
    ANSWERS["q1"] = {
        "n_features": int(n_features),
        "n_numerical": len(numerical),
        "n_dummies": len(dummies),
        "pred": "pred10_rf_best_rule_55",
        "cagr": _num(cagr10, 4),
        "avg_investments_per_day": _num(row10.sim1_avg_investments_per_day, 4),
        "n_trades": int(row10.sim1_count_investments),
        "original_cagr": 1.1195,
    }

    extra = res[res.prediction.isin(extra_cols)].copy()
    extra.to_csv(OUT / "q2_rf_threshold_sim.csv", index=False)
    below5 = extra[extra.sim1_avg_investments_per_day < 5].sort_values(
        "sim1_CAGR", ascending=False
    )
    print("\n  RF extra thresholds (0.56..0.64):")
    print_sim(extra)
    if below5.empty:
        raise RuntimeError("Q2: no extra RF threshold has avg daily investments < 5")
    best = below5.iloc[0]
    thresh = int(best.prediction.rsplit("_", 1)[-1])
    print(
        f"\n  >>> Q2 answer: threshold = {thresh / 100:.2f} "
        f"({best.prediction}; CAGR={best.sim1_CAGR:.4f}, "
        f"avg trades/day={best.sim1_avg_investments_per_day:.3f} < 5)"
    )
    ANSWERS["q2"] = {
        "threshold": thresh / 100.0,
        "threshold_hundredths": thresh,
        "pred": best.prediction,
        "cagr": _num(best.sim1_CAGR, 4),
        "avg_investments_per_day": _num(best.sim1_avg_investments_per_day, 4),
        "n_trades": int(best.sim1_count_investments),
        "pred10_avg_investments_per_day": _num(row10.sim1_avg_investments_per_day, 4),
        "candidates_below_5": [
            {
                "pred": r.prediction,
                "cagr": _num(r.sim1_CAGR, 4),
                "avg_investments_per_day": _num(r.sim1_avg_investments_per_day, 4),
            }
            for r in below5.itertuples()
        ],
    }


# ----------------------------------------------------------------------------
# Q3  (original dummies, target = growth_future_30d > 1.022)
# ----------------------------------------------------------------------------
def q3(df_full: pd.DataFrame):
    header("Q3 | original dummies, target growth_future_30d > 1.022")

    new_df, numerical, dummies, n_features = prepare(
        df_full, drop_ticker_month_wom=False, growth_threshold=1.022
    )
    if n_features != 301:
        print(f"  [warn] expected 301 features, got {n_features}")
    add_manual_preds(new_df)
    fit_models(new_df, numerical + dummies, fit_rf=True)

    res = simulate(new_df)
    res.to_csv(OUT / "q3_sim1_results.csv", index=False)
    print("\n  simulation (all predictors):")
    print_sim(res)

    ranked = res.dropna(subset=["sim1_CAGR"]).sort_values("sim1_CAGR", ascending=False)
    print("\n  ranked by CAGR:")
    print_sim(ranked.head(8))

    row5 = res.loc[res.prediction == "pred5_clf_10"].iloc[0]
    n_trades = int(row5.sim1_count_investments)
    best_pred = ranked.iloc[0].prediction
    avg_day = float(row5.sim1_avg_investments_per_day)
    answer_opt = int(np.round(avg_day))
    print(
        f"\n  >>> Q3 answer: pred5_clf_10 Average investments per day = "
        f"{answer_opt}  (raw {avg_day:.4f}; total TEST trades={n_trades}; "
        f"CAGR={row5.sim1_CAGR:.4f}; best CAGR overall is {best_pred} = "
        f"{ranked.iloc[0].sim1_CAGR:.4f})"
    )
    ANSWERS["q3"] = {
        "n_features": int(n_features),
        "target_threshold": 1.022,
        "target_positive_rate": _num(new_df[TARGET].mean(), 4),
        "pred": "pred5_clf_10",
        "avg_investments_per_day": _num(avg_day, 4),
        "n_trades_total_test": n_trades,
        "answer_option": answer_opt,
        "cagr": _num(row5.sim1_CAGR, 4),
        "best_cagr_pred": best_pred,
        "best_cagr": _num(ranked.iloc[0].sim1_CAGR, 4),
    }


def main():
    header("Homework 4 | load dataset", first=True)
    df_full = load_full()
    df_full["ln_volume"] = np.log(df_full.Volume + 1e-6)
    df_full["div_payout"] = (df_full.Dividends > 0).astype(int)
    df_full["stock_split"] = (df_full["Stock Splits"] > 0).astype(int)

    q1_q2(df_full)
    q3(df_full)

    ANSWERS["q4"] = {
        "see": "README.md",
        "summary": (
            "Size each 30-day hold by Random Forest probability (threshold ladder "
            "0.57-0.65), cap concurrent names per day, and only deploy when the "
            "depth-4/10 tree agrees on the macro regime."
        ),
    }
    (OUT / "answers.json").write_text(
        json.dumps(ANSWERS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {OUT / 'answers.json'}")
    print(json.dumps(ANSWERS, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
