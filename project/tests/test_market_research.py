"""Offline market-aware research settings, trading costs, benchmarks and German orchestration."""

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from driftrev import artifacts, config, dataset, de_pipeline, pipeline, prices, signals, simulate, store


@pytest.fixture
def de_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "de"
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    for name, path in (
        ("DATA_DIR", data_dir),
        ("WAREHOUSE", data_dir / "warehouse.duckdb"),
        ("CACHE_DIR", data_dir / "cache"),
        ("SNAPSHOT_DIR", data_dir / "snapshot"),
        ("ARTIFACTS_DIR", data_dir / "artifacts"),
    ):
        monkeypatch.setattr(config, name, path)
    return data_dir


def test_research_settings_per_market(monkeypatch):
    us = config.research_settings(config.MARKETS["us"])
    de = config.research_settings(config.MARKETS["de"])
    assert us is config.SETTINGS.research
    assert de.book_exclude_families == ()
    assert de.book_exclude_categories == ()
    for field in fields(config.Research):
        if field.name not in ("book_exclude_families", "book_exclude_categories"):
            assert getattr(de, field.name) == getattr(us, field.name)
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    assert config.research_settings() is de


def test_cost_model_per_market(monkeypatch):
    flat = config.Costs().bps_one_way
    old_costs = config.CapCosts()
    for code, costs in (("us", (8, 15, 25)), ("de", (10, 20, 35))):
        monkeypatch.setattr(config, "MARKET", config.MARKETS[code])
        for bucket, bps in zip(("large", "mid", "small"), costs):
            assert simulate.one_way_cost(bucket) == pytest.approx(bps / 1e4)
            assert simulate.one_way_cost(bucket, "cap", mult=2) == pytest.approx(2 * bps / 1e4)
            assert simulate.round_trip_cost(bucket, "cap") == pytest.approx(2 * bps / 1e4)
            assert simulate.one_way_cost(bucket, "flat") == pytest.approx(flat / 1e4)
            if code == "us":
                assert simulate.one_way_cost(bucket) == old_costs.bps(bucket, flat) / 1e4
        for bucket in ("unknown", None):
            assert simulate.one_way_cost(bucket) == pytest.approx(costs[1] / 1e4)
        with monkeypatch.context() as patch:
            patch.setattr(simulate, "S", replace(config.SETTINGS, cap_costs=replace(old_costs, enabled=False)))
            assert simulate.one_way_cost("small") == pytest.approx(flat / 1e4)
            assert simulate.one_way_cost("small", "cap") == pytest.approx(costs[2] / 1e4)


def test_benchmark_tickers(monkeypatch):
    for code, tickers in (("us", ("SPY", "IWM")), ("de", ("^GDAXI", "^SDAXI"))):
        monkeypatch.setattr(config, "MARKET", config.MARKETS[code])
        assert pipeline.benchmark_tickers() == tickers
        block = {"model": "lgbm", "feature_set": "cs_drop_market", "first_year": 2018,
                 "purge_days": 16, "n_oos_events": 10, "oos_auc": 0.7,
                 "oos_precision_selected": 0.6, "oos_base_rate": 0.5, "strategies": [],
                 "benchmarks": [{"name": f"{tickers[0]}_buy_hold", "cagr": 0.1, "sharpe": 0.8,
                                 "max_drawdown": -0.2}]}
        metrics = pipeline._wf_metrics(block)["walkforward"]
        assert metrics["spy_cagr"] == 0.1
        assert metrics["spy_sharpe"] == 0.8
        assert metrics["spy_max_drawdown"] == -0.2


def test_signals_book_rules_follow_market(monkeypatch):
    events = pd.DataFrame({"date": pd.to_datetime(["2024-03-01"] * 2), "ticker": ["BREAK", "CRASH"],
                           "event_family": ["momentum_breakout", "shock_crash"], "edgar_category": "none",
                           "regime_state": "normal", "close": 100.0, "cap_bucket": "large"})
    for code, breakout_tier in (("us", "SKIP"), ("de", "STRONG_BUY")):
        monkeypatch.setattr(config, "MARKET", config.MARKETS[code])
        result = signals.build_signals(events, np.array([0.9, 0.9]), 0.5, "test", lambda p, c: 0.05)
        tiers = result.set_index("ticker")["tier"].to_dict()
        assert tiers == {"BREAK": breakout_tier, "CRASH": "STRONG_BUY"}


def test_standard_strategies_follow_market(monkeypatch):
    for code in ("us", "de"):
        monkeypatch.setattr(config, "MARKET", config.MARKETS[code])
        settings = config.research_settings()
        strategies = {cfg.name: cfg for cfg in simulate.standard_strategies("p", 0.5)}
        for name in ("model_top5", "model_top3_concentrated", "model_top5_risk_managed", "model_pead_only",
                     "model_crash_only", "model_long_short", "guide_model_all_events_R"):
            assert strategies[name].exclude_families == settings.book_exclude_families
            assert strategies[name].exclude_categories == settings.book_exclude_categories
        for name in ("all_events_unconditional", "model_breakout_only", "guide_armA_momentum_breakout",
                     "guide_armB_plus_consolidation", "guide_armC_plus_window_release", "guide_model_breakout_R"):
            assert strategies[name].exclude_families == ()
            assert strategies[name].exclude_categories == ()
        overridden = simulate.standard_strategies("p", 0.5, ("shock_crash",), ("none",))
        assert next(cfg for cfg in overridden if cfg.name == "model_top5").exclude_families == ("shock_crash",)
        assert next(cfg for cfg in overridden if cfg.name == "model_top5").exclude_categories == ("none",)


def test_research_de_refuses_us_and_requires_transform(de_env, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])
    with pytest.raises(RuntimeError, match="DRIFTREV_MARKET=de"):
        de_pipeline.research_de()
    assert not de_env.exists()
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    with pytest.raises(RuntimeError, match="run transform_de first"):
        de_pipeline.research_de()
    events = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "ticker": ["AAA.DE"]})
    store.replace_table("events", events)
    with pytest.raises(RuntimeError, match="run transform_de first"):
        de_pipeline.research_de()
    panel = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=2), "ticker": "AAA.DE"})
    market = pd.DataFrame({"ticker": ["^GDAXI", "^SDAXI"], "date": pd.to_datetime(["2024-01-02"] * 2)})
    store.replace_table("daily_panel", panel)
    store.replace_table(prices.MARKET_TABLE, market)
    ml, feats = events.assign(split="test"), ["x"]

    def prepare(actual):
        pd.testing.assert_frame_equal(actual, events, check_dtype=False)
        return ml, feats

    def train(actual_ml, actual_feats, actual_panel, actual_market, tune, with_walkforward):
        assert actual_ml is ml and actual_feats is feats
        pd.testing.assert_frame_equal(actual_panel, panel, check_dtype=False)
        pd.testing.assert_frame_equal(actual_market, market)
        assert tune is True and with_walkforward is False
        assert config.ARTIFACTS_DIR.is_dir()
        return {"model": {"name": "lgbm", "test_auc": float("nan")}}

    monkeypatch.setattr(dataset, "prepare", prepare)
    monkeypatch.setattr(pipeline, "train_and_backtest", train)
    artifacts.save_metrics({"existing": "kept"})
    metrics = de_pipeline.research_de(tune=True, with_walkforward=False)
    assert metrics["market"] == "de" and metrics["existing"] == "kept"
    assert metrics["n_events"] == 1 and metrics["n_panel_rows"] == 2
    assert metrics["data_through"] == "2024-01-03"
    saved = artifacts.load_metrics()
    assert saved == {**metrics, "model": {"name": "lgbm", "test_auc": None}}
