"""Research layer: cross-sectional features, walk-forward boundaries, cap-aware costs, capacity threshold."""

import numpy as np
import pandas as pd
import pytest

from driftrev import analysis, config, dataset, models, research, signals, simulate, walkforward


# ---------------------------------------------------------------- cross-sectional features
def _frame(n_days=6, per_day=8, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2023-03-01", periods=n_days), per_day)
    return pd.DataFrame({"date": dates, "ticker": [f"T{i}" for i in range(len(dates))],
                         "ret_1d": rng.normal(size=len(dates)), "rsi_14": rng.uniform(10, 90, len(dates)),
                         "vix": np.repeat(rng.uniform(12, 30, n_days), per_day)})


def test_cross_sectional_is_per_date():
    ml = _frame()
    out = dataset.add_cross_sectional(ml, ["ret_1d", "rsi_14"])
    r = out.groupby("date")["cs_rank_ret_1d"]
    assert (out["cs_rank_ret_1d"] > 0).all() and (out["cs_rank_ret_1d"] <= 1).all()
    assert r.max().eq(1.0).all()                                   # every date has its own top rank
    assert out.groupby("date")["cs_dm_ret_1d"].mean().abs().max() < 1e-6   # demeaned per date
    assert "cs_rank_vix" not in out                                # only requested columns


def test_cross_sectional_has_no_lookahead():
    ml = _frame()
    base = dataset.add_cross_sectional(ml, ["ret_1d"])
    day = ml["date"].unique()[2]
    changed = ml.copy()
    later = changed["date"] > day
    changed.loc[later, "ret_1d"] = changed.loc[later, "ret_1d"].to_numpy()[::-1] * 100   # rewrite the FUTURE
    out = dataset.add_cross_sectional(changed, ["ret_1d"])
    early = ml["date"] <= day
    pd.testing.assert_frame_equal(base[early].reset_index(drop=True), out[early].reset_index(drop=True))


def test_cross_sectional_small_dates_are_neutral():
    ml = _frame(per_day=2)
    out = dataset.add_cross_sectional(ml, ["ret_1d"], min_events=3)
    assert (out["cs_rank_ret_1d"] == 0.5).all()


def test_feature_sets_drop_market_level():
    ml = dataset.add_cross_sectional(_frame(), ["ret_1d"])
    ml["vix"] = 1.0
    ml["sector_corr_20d"] = 1.0
    ml["month_wom_Mar_w1"] = 1
    ml["month_March"] = 1
    ml["sector_Energy"] = 1
    fs = dataset.feature_sets(ml)
    assert "vix" in fs["pooled"] and "vix" not in fs["cs_drop_market"]
    assert "month_wom_Mar_w1" in fs["pooled_wom"] and "month_wom_Mar_w1" not in fs["pooled"]
    assert "month_March" not in fs["cs_drop_market"] and "sector_Energy" in fs["cs_drop_market"]
    assert "cs_rank_ret_1d" in fs["cs_drop_market"] and "cs_rank_ret_1d" not in fs["pooled"]
    assert len(fs["pooled"]) == len(set(fs["pooled"]))


# ---------------------------------------------------------------- walk-forward
def test_walkforward_boundaries_do_not_leak():
    dates = pd.Series(pd.bdate_range("2014-02-03", "2026-11-13"))
    splits = walkforward.year_splits(dates, 2018, 16)
    assert splits[0]["year"] == 2018 and splits[-1]["year"] == 2026
    purge = pd.Timedelta(days=16)
    for sp in splits:
        # a row used to train the year-Y model has its 10-day label closed before Y starts
        assert sp["pred_train_cutoff"] + purge <= sp["pred_start"]
        assert sp["val_train_cutoff"] + purge <= sp["val_start"]
        assert sp["val_end"] < sp["pred_start"]              # threshold year strictly precedes the prediction year
        assert sp["val_start"] < sp["pred_start"] and sp["pred_end"] > sp["pred_start"]
    for a, b in zip(splits, splits[1:]):
        assert a["pred_end"] < b["pred_start"]               # prediction years never overlap


def test_walkforward_run_trains_only_on_closed_labels(monkeypatch):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    ml = pd.DataFrame({"date": np.repeat(dates, 12)})
    ml["x"] = rng.normal(size=len(ml))
    ml[dataset.TARGET] = (ml["x"] + rng.normal(size=len(ml)) > 0.8).astype(float)
    ml["event_family"] = "earnings_up"
    seen = []
    real = research.fit_predict

    def spy(tr, pr, feats, kind, params, layout="pooled"):
        seen.append((tr["date"].max(), pr["date"].min(), pr["date"].max()))
        return real(tr, pr, feats, kind, params, layout)

    monkeypatch.setattr(research, "fit_predict", spy)
    stitched, table = walkforward.run(ml, ["x"], "tree", {"max_depth": 3}, first_year=2018)
    assert list(table["year"]) == [2018, 2019, 2020]
    purge = pd.Timedelta(days=config.SETTINGS.research.wf_purge_days)
    for tr_max, pr_min, _ in seen:
        assert tr_max + purge < pr_min                        # nothing within the label horizon of the boundary
    for y, g in stitched.groupby("wf_year"):                  # each event is predicted only by its own year's model
        assert g["date"].min() >= pd.Timestamp(f"{y}-01-01") and g["date"].max() <= pd.Timestamp(f"{y}-12-31")
    assert np.allclose(stitched["wf_score"] - 0.5, stitched["proba_wf"] - stitched["wf_thr"])


# ---------------------------------------------------------------- cap-aware costs
def test_cap_cost_math():
    assert simulate.one_way_cost("large", "cap") == pytest.approx(8e-4)
    assert simulate.one_way_cost("small", "cap") == pytest.approx(25e-4)
    assert simulate.one_way_cost("mid", "cap", mult=2) == pytest.approx(30e-4)
    assert simulate.one_way_cost("small", "flat") == pytest.approx(15e-4)   # flat option ignores the bucket
    assert simulate.round_trip_cost("large", "cap") == pytest.approx(16e-4)
    assert simulate.one_way_cost(None, "cap") == pytest.approx(15e-4)       # unknown bucket -> mid


def _one_trade(cap, cost_model):
    dates = pd.bdate_range("2024-01-01", periods=15)
    panel = pd.DataFrame({"ticker": "X", "date": dates, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0})
    ev = pd.DataFrame({"ticker": ["X"], "date": [dates[0]], "next_date": [dates[1]], "event_family": ["shock_crash"],
                       "regime_state": ["normal"], "p": [0.9], "cap_bucket": [cap], "close": [10.0]})
    cfg = simulate.SimConfig(name="t", prob_col="p", min_prob=0.5, max_positions=1, hold_days=10, initial_capital=1000,
                             cost_model=cost_model)
    eq, tr = simulate.exact_sim(ev, panel, cfg)
    return eq.iloc[-1], tr.iloc[0]


def test_exact_sim_charges_cap_specific_costs():
    for cap, bps in (("large", 8), ("mid", 15), ("small", 25)):
        final, tr = _one_trade(cap, "cap")
        # flat prices: the only P&L is cost on entry + exit notional (~1000 each side)
        assert tr["ret_net"] == pytest.approx(-2 * bps / 1e4, rel=0.02)
        assert final == pytest.approx(1000 * (1 - 2 * bps / 1e4), rel=0.02)
    _, flat = _one_trade("small", "flat")
    assert flat["ret_net"] == pytest.approx(-30e-4, rel=0.02)


def test_vector_sim_fees_by_cap():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "growth_future_10d": [1.0, 1.0],
                       "mae_10d": [0.0, 0.0], "mfe_10d": [0.0, 0.0], "cap_bucket": ["large", "small"], "pred_x": [1, 1]})
    v = simulate.vector_sim(df, ["pred_x"], cost_model="cap")
    assert v.iloc[0]["fees"] == pytest.approx(100 * (16e-4 + 50e-4))
    v = simulate.vector_sim(df, ["pred_x"], cost_model="flat")
    assert v.iloc[0]["fees"] == pytest.approx(100 * 2 * 30e-4)


def test_signal_net_edge_uses_cap_cost():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 1000)
    d = pd.DataFrame({"p": p, "growth_future_10d": 1.03 + 0 * p})
    lk = signals.expected_return_lookup(d, "p")
    assert lk(0.5, "large") - lk(0.5, "small") == pytest.approx(2 * (25 - 8) / 1e4)
    assert lk(0.5, "large") == pytest.approx(0.03 - 16e-4)


# ---------------------------------------------------------------- capacity-aware threshold
def test_capacity_threshold_respects_cap():
    rng = np.random.default_rng(3)
    n_days, per_day = 100, 20
    dates = np.repeat(np.arange(n_days), per_day)
    p = rng.uniform(0.3, 0.9, n_days * per_day)
    y = (rng.uniform(size=p.size) < p).astype(int)           # precision rises with the threshold
    free = models.choose_threshold(y, p, min_share=0.0, dates=dates)
    capped = models.choose_threshold(y, p, min_share=0.0, dates=dates, max_trades_per_day=6.0)
    assert models.avg_trades_per_day(dates, p >= capped) <= 6.0
    assert capped >= free
    # unreachable capacity: fall back to the most selective cut instead of the 0.5 default
    assert models.choose_threshold(y, p, min_share=0.0, dates=dates, max_trades_per_day=0.1) == 0.8
    # a capacity that the lowest threshold already satisfies changes nothing
    assert models.choose_threshold(y, p, min_share=0.0, dates=dates, max_trades_per_day=1e9) == free
    # a binding capacity forces a higher (more selective) cut than a loose one
    loose = models.choose_threshold(y, p, min_share=0.0, dates=dates, max_trades_per_day=1e9)
    tight = models.choose_threshold(y, p, min_share=0.0, dates=dates, max_trades_per_day=2.0)
    assert tight >= loose


def test_avg_trades_per_day_counts_active_days_only():
    dates = np.array([1, 1, 1, 2, 3])
    sel = np.array([True, True, False, False, True])
    assert models.avg_trades_per_day(dates, sel) == pytest.approx((2 + 1) / 2)


# ---------------------------------------------------------------- HW3 / HW4 helpers
def test_uniquely_correct_and_version_guard():
    d = pd.DataFrame({dataset.TARGET: [1, 0, 1, 0], "pred_a": [1, 0, 0, 0], "pred_b": [1, 1, 0, 0], "pred_c": [0, 1, 1, 0]})
    t = analysis.uniquely_correct_table(d, ["pred_a", "pred_b", "pred_c"]).set_index("prediction")
    # row 2 (y=0): a,d correct? a=0,b=1,c=1 -> only a correct; row 3 (y=1): c correct only? a=0 wrong,b=0 wrong,c=1 right
    assert t.loc["pred_a", "n_uniquely_correct"] == 1 and t.loc["pred_c", "n_uniquely_correct"] == 1
    v = models.check_sklearn_version()
    assert {"sklearn", "sklearn_tested_ok", "lightgbm"} <= set(v)


def test_month_wom_dummies_and_manual_rule():
    ev = pd.DataFrame({"date": pd.to_datetime(["2024-03-04", "2024-03-20"])})
    ev["month"] = ev["date"].dt.month_name()
    wom = ev["date"].dt.month_name().str[:3] + "_w" + ((ev["date"].dt.day - 1) // 7 + 1).astype(str)
    assert list(wom) == ["Mar_w1", "Mar_w3"]
    df = pd.DataFrame({"event_family": ["earnings_up"] * 3, "edgar_category": ["none"] * 3, "close_sma50": 1.1,
                       "close_sma200": 1.1, "abn_ret_1d": 0.05, "rsi_14": 50.0, "arm_a": 0, "arm_b": 0, "arm_c": 0,
                       "dgs10": [4.0, 4.6, 3.0], "dgs5": [3.5, 3.0, 4.5]})
    assert list(models.manual_rules(df)["pred_manual_dgs10_dgs5"]) == [1, 0, 0]
    assert "pred_manual_dgs10_dgs5" not in models.manual_rules(df.drop(columns="dgs5"))
