"""Offline German daily publishing through the shared pipeline and artifact contract."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_de_pipeline import _world

from driftrev import (
    artifacts,
    config,
    de_membership,
    de_pipeline,
    de_universe,
    housekeeping,
    models,
    notify,
    pipeline,
    prices,
    report,
    store,
)


@pytest.fixture
def de_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "de"
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    for name, path in (
        ("DATA_ROOT", data_dir),
        ("DATA_DIR", data_dir),
        ("WAREHOUSE", data_dir / "warehouse.duckdb"),
        ("CACHE_DIR", data_dir / "cache"),
        ("SNAPSHOT_DIR", data_dir / "snapshot"),
        ("ARTIFACTS_DIR", data_dir / "artifacts"),
        ("HISTORY_DIR", data_dir / "artifacts" / "history"),
        ("SITE_DIR", data_dir / "site"),
        ("LOG_DIR", data_dir / "logs"),
    ):
        monkeypatch.setattr(config, name, path)
    return data_dir


@pytest.fixture
def daily_stubs(monkeypatch):
    calls = []

    def train(*args, **kwargs):
        calls.append("train")
        return {"model": {"name": "test", "threshold": 0.5, "test_auc": 0.6}}

    def record(name):
        def run(*args, **kwargs):
            calls.append(name)
            return {"step": name}
        return run

    monkeypatch.setattr(pipeline, "train_and_backtest", train)
    monkeypatch.setattr(models, "load_bundle", lambda: {"name": "test", "threshold": 0.5})
    monkeypatch.setattr(models, "predict_with_bundle", lambda bundle, frame: np.full(len(frame), 0.8))
    monkeypatch.setattr(report, "build", record("report"))
    monkeypatch.setattr(housekeeping, "archive_signals", record("archive"))
    monkeypatch.setattr(housekeeping, "run", record("housekeeping"))
    monkeypatch.setattr(notify, "notify", record("notify"))
    return calls


def _seed_warehouse():
    universe, membership, stocks, market, macro = _world()
    # Keep enough feature history and give both Xetra and Hamburg a latest-session crash.
    last = macro.index[299]
    stocks = stocks.loc[stocks["date"].le(last)].copy()
    latest = stocks["date"].eq(last)
    stocks.loc[latest, "close"] *= 0.85
    stocks.loc[latest, "volume"] *= 5
    stocks.loc[latest, "high"] = stocks.loc[latest, ["open", "close"]].max(axis=1) * 1.01
    stocks.loc[latest, "low"] = stocks.loc[latest, ["open", "close"]].min(axis=1) * 0.99
    market = market.loc[market["date"].le(last)]
    macro = (macro.loc[:last].rename_axis("date").reset_index().rename(columns={"bund10y": "value"})
             .assign(series="bund10y"))
    for table, frame in ((de_universe.TABLE, universe), (de_membership.TABLE, membership),
                         (prices.PRICE_TABLE, stocks), (prices.MARKET_TABLE, market), ("macro", macro)):
        store.replace_table(table, frame)


def _unexpected(*args, **kwargs):
    pytest.fail("pipeline used the wrong market loader")


def test_run_pipeline_de_warehouse_mode(de_env, daily_stubs, monkeypatch):
    _seed_warehouse()
    transformed = de_pipeline.transform_de()
    loaded = de_pipeline.load_from_warehouse_de()
    assert set(loaded) == {"events", "panel", "market", "universe", "ctx"}
    for key in loaded:
        actual = loaded[key].where(loaded[key].notna(), np.nan)
        expected = transformed[key].where(transformed[key].notna(), np.nan)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    monkeypatch.setattr(pipeline, "load_from_warehouse", _unexpected)
    monkeypatch.setattr(de_pipeline, "ingest_de", _unexpected)
    monkeypatch.setattr(de_pipeline, "transform_de", _unexpected)
    artifacts.save_metrics({"existing": "kept"})

    entry = pipeline.run_pipeline(mode="warehouse")

    assert entry["status"] == "success" and entry["market"] == "de"
    assert artifacts.load_run_log() == [entry]
    metrics = artifacts.load_metrics()
    assert metrics["market"] == "de" and metrics["existing"] == "kept"
    assert metrics["universe_size"] == 2 and metrics["universe_size_incl_removed"] == 4
    assert daily_stubs == ["train", "archive", "report", "housekeeping", "notify"]
    assert config.ARTIFACTS_DIR == de_env / "artifacts"
    for name in ("signals", "events"):
        assert artifacts.path(name).is_file()
    signals = artifacts.load_frame("signals")
    events = artifacts.load_frame("events")
    assert not signals.empty and signals["ticker"].str.endswith(".DE").all()
    assert "CCC.HM" in set(events.loc[events["date"].eq(signals["date"].max()), "ticker"])
    assert entry["steps"]["publish"]["n_signals_today"] == len(signals)
    assert entry["steps"]["publish"]["n_actionable_today"] == signals["tier"].isin(["STRONG_BUY", "BUY"]).sum()
    universe = artifacts.load_frame("universe")
    assert universe[["cik", "wikipedia_title"]].isna().all().all()
    assert list(de_env.parent.iterdir()) == [de_env]


def test_run_pipeline_de_update_mode_uses_de_loaders(de_env, daily_stubs, monkeypatch):
    _seed_warehouse()
    data = de_pipeline.transform_de()
    calls = []

    def ingest():
        calls.append("ingest_de")
        return {"german_inputs": True}

    def transform():
        calls.append("transform_de")
        return data

    monkeypatch.setattr(de_pipeline, "ingest_de", ingest)
    monkeypatch.setattr(de_pipeline, "transform_de", transform)
    monkeypatch.setattr(de_pipeline, "load_from_warehouse_de", _unexpected)
    monkeypatch.setattr(pipeline, "ingest", _unexpected)
    monkeypatch.setattr(pipeline, "transform", _unexpected)
    # An existing market-local bundle is reused without training.
    (config.ARTIFACTS_DIR / artifacts.MODEL).touch()
    entry = pipeline.run_pipeline(mode="update")
    assert entry["status"] == "success"
    assert calls == ["ingest_de", "transform_de"]
    assert entry["steps"]["ingest"] == {"german_inputs": True}
    assert daily_stubs == ["archive", "report", "housekeeping", "notify"]


def test_run_pipeline_de_file_mode_refused(de_env, monkeypatch):
    monkeypatch.setattr(pipeline, "load_snapshot", _unexpected)
    with pytest.raises(ValueError, match="mode 'file' is not supported for market 'de' yet"):
        pipeline.run_pipeline(mode="file")
    assert not de_env.exists()


def test_regime_explain_per_market(monkeypatch):
    ctx = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "regime_points": [0], "regime_state": ["normal"]})
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])
    us = pipeline.regime_block(ctx)
    assert us["explain"] == (
        "1 point per trigger: SPY<200d MA, VIX/VIX3M>1, sector-correlation pctile>0.9, "
        "breadth<40% above 200d MA, Baa-10y spread +25bp/20d. 0-1 normal, 2-3 elevated, 4+ stressed.")
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    de = pipeline.regime_block(ctx)
    assert de.keys() == us.keys()
    assert de["explain"] == (
        "1 point per trigger: DAX<200d MA, DAX realised-vol term (20/60)>1, "
        "sector-correlation pctile>0.9 (n/a for DE), breadth<40% above 200d MA, credit spread (n/a for DE). "
        "0-1 normal, 2-3 elevated, 4+ stressed.")


def test_de_is_implemented(monkeypatch):
    market = config.MARKETS["de"]
    assert market.implemented is True
    assert config.require_implemented_market(market) is market
    monkeypatch.setattr(config, "MARKET", market)
    assert config.require_implemented_market() is market
    unimplemented = replace(market, code="xx", implemented=False)
    monkeypatch.setattr(config, "MARKET", unimplemented)
    with pytest.raises(config.MarketNotImplementedError):
        config.require_implemented_market(unimplemented)
    with pytest.raises(config.MarketNotImplementedError):
        config.require_implemented_market()


def test_us_pipeline_path_unchanged(de_env, daily_stubs, monkeypatch):
    _seed_warehouse()
    data = de_pipeline.transform_de()
    for key in ("events", "panel", "universe"):
        data[key]["ticker"] = data[key]["ticker"].replace({"AAA.DE": "AAA", "BBB.DE": "BBB", "CCC.HM": "CCC"})
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])
    calls = []

    def load():
        calls.append("us_warehouse")
        return data

    monkeypatch.setattr(pipeline, "load_from_warehouse", load)
    monkeypatch.setattr(de_pipeline, "load_from_warehouse_de", _unexpected)
    monkeypatch.setattr(de_pipeline, "ingest_de", _unexpected)
    monkeypatch.setattr(de_pipeline, "transform_de", _unexpected)
    entry = pipeline.run_pipeline(mode="warehouse")
    assert entry["status"] == "success" and entry["market"] == "us"
    assert calls == ["us_warehouse"]
    assert artifacts.load_metrics()["market"] == "us"
    assert artifacts.load_run_log()[0]["market"] == "us"
    assert set(artifacts.load_frame("signals")["ticker"]) == {"AAA", "BBB", "CCC"}
    assert daily_stubs == ["train", "archive", "report", "housekeeping", "notify"]


@pytest.mark.parametrize("code,description", [
    ("us", "Event-driven ML on the S&amp;P 1500 (earnings drift · crash reversal · momentum breakout)."),
    ("de", "Event-driven ML on Xetra DAX + MDAX + SDAX (crash reversal · momentum breakout)."),
])
def test_report_labels_follow_market(de_env, monkeypatch, code, description):
    monkeypatch.setattr(config, "MARKET", config.MARKETS[code])
    benchmark = f"{config.MARKET.primary_benchmark}_buy_hold"
    artifacts.save_metrics({"data_through": "2024-01-15", "market": code,
                            "model": {"name": "test", "threshold": 0.5, "test_auc": 0.6}})
    artifacts.save_frame("equity", pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="7D"),
        "strategy": benchmark, "equity": [100.0, 102.0, 105.0],
    }))

    result = report.build()

    assert result["site"] == str(de_env / "site")
    page = (config.SITE_DIR / "index.html").read_text()
    assert description in page and f"{benchmark} (" in page
    other = "SPY_buy_hold" if code == "de" else "^GDAXI_buy_hold"
    assert other not in page
    assert list(de_env.parent.iterdir()) == [de_env]
