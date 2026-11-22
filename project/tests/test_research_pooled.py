"""Offline pooled-market comparison, German validation selection and warehouse isolation."""

import dataclasses
import json
import logging
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest
from test_de_pipeline import _build, _world

from driftrev import config, dataset, research, research_pooled


@pytest.fixture(scope="module")
def market_events():
    # Reuse build_all's German schema, with 600 events spanning every research block.
    base, _, _ = _build(_world())
    dates = pd.date_range("2015-01-05", "2025-11-03", periods=98).normalize()
    dates = dates.append(pd.DatetimeIndex(["2021-12-24", "2023-12-24"])).sort_values()
    dates = np.repeat(dates, 6)
    n = len(dates)
    frames = {}
    for seed, market in enumerate(("us", "de")):
        rng = np.random.default_rng(seed)
        events = base.iloc[np.arange(n) % len(base)].reset_index(drop=True).copy()
        events["date"] = dates
        events["next_date"] = events["date"] + pd.Timedelta(days=1)
        events["ticker"] = [f"{market}_{i}" for i in range(n)]
        families = ("shock_crash", "momentum_breakout", "earnings_up") if market == "us" else research_pooled.SAME_FAMILIES
        events["event_family"] = np.resize(families, n)
        events["ret_1d"] = rng.normal(0.1 if market == "us" else -0.1, 0.02, n)
        events["growth_future_10d"] = np.where(np.arange(n) % 2, 1.05, 0.95)
        events.loc[np.arange(n) % 6 == 5, "growth_future_10d"] = np.nan
        events = events.drop(columns="eps_surprise_pct", errors="ignore")
        if market == "us":
            events["eps_surprise_pct"] = rng.normal(size=n)
        frames[market] = events
    return frames


@pytest.fixture(scope="module")
def market_ml(market_events):
    return {market: research_pooled.prepare_market(events, market) for market, events in market_events.items()}


def _write_events(path, events):
    with duckdb.connect(str(path)) as con:
        con.register("input_events", events)
        con.execute("create table events as select * from input_events")


def test_load_events_read_only(tmp_path, monkeypatch):
    warehouse = tmp_path / "warehouse.duckdb"
    events = pd.DataFrame({"ticker": ["AAA", "BBB"], "value": [1.0, 2.0]})
    _write_events(warehouse, events)
    before = warehouse.stat().st_mtime_ns
    real_connect = duckdb.connect
    seen = []

    def connect(path, **kwargs):
        seen.append((path, kwargs))
        return real_connect(path, **kwargs)

    monkeypatch.setattr(duckdb, "connect", connect)
    pd.testing.assert_frame_equal(research_pooled.load_events(warehouse), events)
    assert seen == [(str(warehouse), {"read_only": True})]
    assert warehouse.stat().st_mtime_ns == before
    missing = tmp_path / "missing.duckdb"
    with pytest.raises(duckdb.IOException):
        research_pooled.load_events(missing)
    assert not missing.exists()


def test_prepare_market_tags(market_events):
    for market, events in market_events.items():
        original = events.copy(deep=True)
        ml = research_pooled.prepare_market(events, market)
        expected, _ = dataset.prepare(events)
        assert ml["market"].eq(market).all()
        assert ml["is_de"].eq(1.0 if market == "de" else 0.0).all()
        assert ml["is_de"].dtype == np.dtype("float64")
        pd.testing.assert_frame_equal(ml.drop(columns=["market", "is_de"]), expected)
        pd.testing.assert_frame_equal(events, original)
        assert {"train", "validation", "test", "purged"} == set(ml["split"])
        assert ml.groupby("date")["cs_dm_ret_1d"].mean().abs().max() < 1e-6


def test_common_features_and_stack(market_ml):
    us, de = market_ml["us"], market_ml["de"]
    assert "eps_surprise_pct" in dataset.feature_sets(us)["pooled"]
    assert "event_family_earnings_up" in us and "event_family_earnings_up" not in de
    for feature_set in dataset.FEATURE_SETS:
        common = research_pooled.common_features(us, de, feature_set)
        expected = [c for c in dataset.feature_sets(de)[feature_set] if c in dataset.feature_sets(us)[feature_set]]
        assert common == expected
        assert "eps_surprise_pct" not in common
        assert "event_family_earnings_up" not in common
        assert "market" not in common and "is_de" not in common
    feature = "event_family_shock_crash"
    without = de.drop(columns=feature)
    combined = research_pooled.stack([us, without], ["ret_1d", feature])
    assert len(combined) == len(us) + len(de)
    np.testing.assert_array_equal(combined.loc[:len(us) - 1, feature], us[feature])
    assert combined.loc[len(us):, feature].eq(0.0).all()
    assert feature not in without
    pd.testing.assert_series_equal(combined["ticker"], pd.concat([us["ticker"], de["ticker"]], ignore_index=True))


def test_pooled_comparison_shape_and_rules(market_ml, monkeypatch, caplog):
    us, de = market_ml["us"], market_ml["de"]
    # Make German and active-US settings differ to exercise the German default.
    feature_set = "pooled_wom"
    monkeypatch.setitem(config.RESEARCH_BY_MARKET, "de",
                        dataclasses.replace(config.research_settings(config.MARKETS["de"]), feature_set=feature_set))
    params = {"rf": {"n_estimators": 3}}
    seen = []

    def fit_predict(tr, pr, feats, kind, prm, layout):
        seen.append((tr.copy(), pr.copy(), list(feats), kind, prm, layout))
        favored = kind == "rf" and tr["market"].eq("de").any() and tr["event_family"].eq("earnings_up").any()
        aligned = favored if pr["split"].eq("validation").all() else not favored
        y = pr[dataset.TARGET].to_numpy()
        return 0.1 + 0.8 * (y if aligned else 1 - y)

    monkeypatch.setattr(research, "fit_predict", fit_predict)
    with caplog.at_level(logging.INFO, logger=research_pooled.__name__):
        comparison = research_pooled.pooled_comparison(us, de, params)
    assert len(comparison) == 8 and len(seen) == 16
    assert comparison["variant"].tolist() == [v for v in research_pooled.VARIANTS for _ in range(2)]
    assert comparison["learner"].tolist() == ["lgbm", "rf"] * 4
    assert comparison["feature_set"].eq(feature_set).all()
    common = research_pooled.common_features(us, de, feature_set)
    known_us, known_de = us[us[dataset.TARGET].notna()], de[de[dataset.TARGET].notna()]
    same_us = known_us[known_us["event_family"].isin(research_pooled.SAME_FAMILIES)]
    sources = {"de_only": [known_de], "pooled_all": [known_us, known_de],
               "pooled_same_families": [same_us, known_de], "us_transfer_same_families": [same_us]}
    for i, row in comparison.iterrows():
        pooled = row["variant"] in ("pooled_all", "pooled_same_families")
        feats = common + ["is_de"] if pooled else common
        assert row["n_features"] == len(feats)
        for j, sample in enumerate(("validation", "test")):
            tr, pr, actual_feats, kind, prm, layout = seen[2 * i + j]
            assert actual_feats == feats and kind == row["learner"]
            assert prm == params.get(kind, {}) and layout == "pooled"
            assert tr[dataset.TARGET].notna().all() and pr[dataset.TARGET].notna().all()
            assert pr["market"].eq("de").all() and pr["split"].eq(sample).all()
            assert not tr["split"].isin(["test", "purged"]).any()
            splits = ["train"] if sample == "validation" else ["train", "validation"]
            if sample == "validation":
                assert tr["split"].eq("train").all()
                assert not tr.loc[tr["market"].eq("de"), "ticker"].isin(de.loc[de["split"].ne("train"), "ticker"]).any()
                assert row["n_train_us"] == int(tr["market"].eq("us").sum())
                assert row["n_train_de"] == int(tr["market"].eq("de").sum())
            expected = pd.concat([frame[frame["split"].isin(splits)] for frame in sources[row["variant"]]])
            assert set(tr["ticker"]) == set(expected["ticker"])
            assert len(tr) == len(expected)
            assert set(pr["ticker"]) == set(known_de.loc[known_de["split"].eq(sample), "ticker"])
            prefix = "val" if sample == "validation" else "test"
            assert row[f"{prefix}_n"] == len(pr)
    assert comparison.loc[comparison["variant"].eq("de_only"), "n_train_us"].eq(0).all()
    assert comparison.loc[comparison["variant"].eq("us_transfer_same_families"), "n_train_de"].eq(0).all()
    assert comparison["winner"].dtype == bool and comparison["winner"].sum() == 1
    winner = comparison.loc[comparison["winner"]].iloc[0]
    assert (winner["variant"], winner["learner"]) == ("pooled_all", "rf")
    assert winner["test_auc"] < comparison["test_auc"].max()
    assert winner.name == research.pick_variant(comparison, eligible=("rf", "lgbm")).name
    changed_test = comparison.copy()
    changed_test[[c for c in comparison if c.startswith("test_")]] = 0.0
    assert winner.name == research.pick_variant(changed_test, eligible=("rf", "lgbm")).name
    assert len([r for r in caplog.records if r.name == research_pooled.__name__ and r.levelno == logging.INFO]) == 8


def test_cli_writes_parquet(tmp_path, market_events, monkeypatch):
    us_warehouse, de_warehouse = tmp_path / "us.duckdb", tmp_path / "de.duckdb"
    for market, path in (("us", us_warehouse), ("de", de_warehouse)):
        _write_events(path, market_events[market])
    before = {path: path.stat().st_mtime_ns for path in (us_warehouse, de_warehouse)}
    tuning = tmp_path / "tuning.parquet"
    params = {"rf": {"n_estimators": 3, "max_depth": 3},
              "lgbm": {"n_estimators": 3, "num_leaves": 7, "min_child_samples": 5}}
    pd.DataFrame([{"model": kind, "params": json.dumps(prm), "val_auc": 0.6, "val_prec_top10": 0.6}
                  for kind, prm in params.items()] +
                 [{"model": "rf", "params": json.dumps({"n_estimators": 50}), "val_auc": 0.9, "val_prec_top10": 0.1}]
                 ).to_parquet(tuning, index=False)
    real_comparison = research_pooled.pooled_comparison
    seen_params = []

    def comparison(ml_us, ml_de, actual_params):
        seen_params.append(actual_params)
        return real_comparison(ml_us, ml_de, actual_params)

    monkeypatch.setattr(research_pooled, "pooled_comparison", comparison)
    out = tmp_path / "reports" / "comparison.parquet"
    table = research_pooled.run(us_warehouse, de_warehouse, out, tuning)
    assert seen_params == [params]
    pd.testing.assert_frame_equal(pd.read_parquet(out), table)
    assert len(table) == 8 and table["winner"].sum() == 1
    cli_out = tmp_path / "cli.parquet"
    result = subprocess.run([sys.executable, "-m", "driftrev.research_pooled",
                             "--us-warehouse", str(us_warehouse), "--de-warehouse", str(de_warehouse),
                             "--out", str(cli_out), "--tuning", str(tuning)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    pd.testing.assert_frame_equal(pd.read_parquet(cli_out).drop(columns="seconds"), table.drop(columns="seconds"))
    # Missing/omitted tuning uses model defaults; capture that path without another fitting pass.
    monkeypatch.setattr(research_pooled, "pooled_comparison", lambda ml_us, ml_de, prm: seen_params.append(prm) or table)
    for missing in (None, tmp_path / "absent.parquet"):
        research_pooled.run(us_warehouse, de_warehouse, out, missing)
        assert seen_params[-1] == {}
    for warehouse in (us_warehouse, de_warehouse):
        with pytest.raises(ValueError, match="out must differ"):
            research_pooled.run(us_warehouse, de_warehouse, warehouse)
        assert warehouse.stat().st_mtime_ns == before[warehouse]
