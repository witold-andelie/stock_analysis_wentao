"""Offline German event orchestration and backward-compatible shared feature hooks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from driftrev import config, de_context, de_market, de_membership, de_pipeline, de_universe, features, prices, store
from driftrev import membership as mbr


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


def _ohlcv(ticker, dates, seed, base=100.0, volume=0.0, crash=False):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.005, len(dates))
    if crash:
        returns[250] = -0.15
    close = base * np.cumprod(1 + returns)
    opn = np.r_[close[0], close[:-1]]
    volumes = np.full(len(dates), volume, dtype=float)
    if crash:
        volumes[250] *= 5
    return pd.DataFrame({
        "ticker": ticker, "date": dates, "open": opn,
        "high": np.maximum(opn, close) * 1.01, "low": np.minimum(opn, close) * 0.99,
        "close": close, "volume": volumes, "dividends": 0.0, "splits": 0.0,
    })


def _world():
    dates = pd.bdate_range("2020-01-01", periods=420, name="date")
    market = pd.concat([_ohlcv(ticker, dates, seed) for seed, ticker in
                        enumerate(("^GDAXI", "^MDAXI", "^SDAXI"), 1)], ignore_index=True)
    stocks = pd.concat([
        _ohlcv(ticker, dates, seed, base, volume, crash=True)
        for seed, (ticker, base, volume) in enumerate((
            ("AAA.DE", 100.0, 1e6), ("BBB.DE", 50.0, 2e5),
            ("CCC.HM", 20.0, 50.0), ("DDD.DE", 100.0, 1e6),
        ), 4)
    ], ignore_index=True)
    membership = pd.DataFrame([
        ("Alpha", "Alpha AG", "dax", dates[0], pd.NaT),
        ("Beta", "Beta AG", "mdax", dates[200], pd.NaT),
        ("Gamma", "Gamma AG", "sdax", dates[0], dates[300]),
    ], columns=de_membership.MEMBERSHIP_COLS)
    rows = []
    for key, name, symbol, source, index, current, first, last in (
        ("Alpha", "Alpha AG", "AAA.DE", "xetra", "dax", True, dates[0], pd.NaT),
        ("Beta", "Beta AG", "BBB.DE", "xetra", "mdax", True, dates[200], pd.NaT),
        ("Gamma", "Gamma AG", "CCC.HM", "hamburg", "sdax", False, dates[0], dates[300]),
        ("Delta", "Delta AG", "DDD.DE", "xetra", None, False, pd.NaT, pd.NaT),
    ):
        rows.append({
            "company_key": key, "name": name, "symbol": symbol, "price_source": source,
            "exchange": "HAM" if source == "hamburg" else "GER", "resolved_by": "search",
            "yahoo_name": name, "matched_query": key, "index_code": index,
            "cap_bucket": config.MARKETS["de"].cap_bucket(index), "is_current_member": current,
            "first_date": first, "last_date": last,
        })
    universe = pd.DataFrame(rows, columns=de_universe.UNIVERSE_COLS)
    macro = pd.DataFrame({"bund10y": np.linspace(-0.5, 0.1, len(dates))}, index=dates)
    return universe, membership, stocks, market, macro


def _build(world, liquid_by_membership=("CCC.HM",), ctx=None):
    universe, membership, stocks, market, macro = world
    return features.build_all(
        de_pipeline.universe_for_features(universe), stocks, market, macro,
        filings=pd.DataFrame(), sue=pd.DataFrame(), pageviews=pd.DataFrame(),
        membership=de_pipeline.membership_for_features(membership, universe),
        consensus_eps=None, filing_text=None,
        ctx=de_context.market_context_de(market, macro) if ctx is None else ctx,
        cap_buckets=dict(config.MARKETS["de"].cap_bucket_by_index),
        liquid_by_membership=liquid_by_membership)


def test_bucket_intervals_custom_map():
    universe, membership, *_ = _world()
    mapped = de_pipeline.membership_for_features(membership, universe)
    buckets = mbr.bucket_intervals(mapped, {"dax": "large", "mdax": "mid", "sdax": "small"})
    assert {ticker: intervals[0][2] for ticker, intervals in buckets.items()} == {
        "AAA.DE": "large", "BBB.DE": "mid", "CCC.HM": "small"}
    assert mbr.bucket_intervals(mapped) == {}
    assert mbr.bucket_intervals(mapped, {}) == {}


def test_membership_and_universe_mapping():
    universe, membership, *_ = _world()
    unresolved = universe.iloc[[0]].assign(company_key="Unresolved", symbol=None)
    universe = pd.concat([universe, unresolved], ignore_index=True)
    membership = pd.concat([membership, membership.iloc[[0]].assign(company_key="Unresolved")], ignore_index=True)
    original_uni, original_mem = universe.copy(deep=True), membership.copy(deep=True)
    mapped = de_pipeline.membership_for_features(membership, universe)
    assert list(mapped.columns) == ["ticker", "index_name", "start_date", "end_date"]
    expected = membership.iloc[:3][["index_code", "start_date", "end_date"]].rename(
        columns={"index_code": "index_name"}).copy()
    expected.insert(0, "ticker", ["AAA.DE", "BBB.DE", "CCC.HM"])
    pd.testing.assert_frame_equal(mapped, expected)
    uni = de_pipeline.universe_for_features(pd.concat([universe, universe.iloc[[0]]], ignore_index=True))
    assert list(uni.columns) == de_pipeline.FEATURE_UNIVERSE_COLS
    assert list(uni["ticker"]) == ["AAA.DE", "BBB.DE", "CCC.HM", "DDD.DE"]
    assert uni["sector"].eq("unknown").all()
    assert uni[["sub_industry", "cik", "wikipedia_title", "sector_etf"]].isna().all().all()
    for col, source in (("company", "name"), ("cap_bucket", "cap_bucket"),
                        ("is_current_member", "is_current_member"), ("price_source", "price_source")):
        assert uni[col].tolist() == universe.iloc[:4][source].tolist()
    pd.testing.assert_frame_equal(universe, original_uni)
    pd.testing.assert_frame_equal(membership, original_mem)


def test_build_all_de_events():
    world = _world()
    events, panel, _ = _build(world)
    dates = world[4].index
    crashes = events.loc[events["date"].eq(dates[250])].set_index("ticker")
    assert set(crashes.index) == {"AAA.DE", "BBB.DE", "CCC.HM"}
    assert crashes["event_family"].eq("shock_crash").all()
    assert crashes["cap_bucket"].to_dict() == {"AAA.DE": "large", "BBB.DE": "mid", "CCC.HM": "small"}
    assert "DDD.DE" not in set(events["ticker"])
    assert events["in_index"].all()
    assert set(events["event_family"]) == {"shock_crash", "momentum_breakout"}
    assert events["edgar_category"].eq("none").all()
    assert events["edgar_items"].eq("").all()
    assert events["is_earnings"].eq(False).all()
    bbb = panel.loc[panel["ticker"].eq("BBB.DE")].set_index("date")
    assert not bbb.loc[dates[:200], "in_index"].any()
    assert bbb.loc[dates[200]:, "in_index"].all()
    ccc = panel.loc[panel["ticker"].eq("CCC.HM")].set_index("date")
    assert ccc.loc[dates[:300], "is_liquid"].all()
    assert not ccc.loc[dates[300]:, "is_liquid"].any()
    assert ccc.loc[dates[250], "is_crash"]
    assert crashes.loc["CCC.HM", "n_crash_prior_252"] == 0
    baseline, baseline_panel, _ = _build(world, liquid_by_membership=None)
    assert "CCC.HM" not in set(baseline["ticker"])
    unaffected = events.loc[events["ticker"].ne("CCC.HM")].reset_index(drop=True)
    # Counting the regional-name crash also changes the shared market's crash breadth.
    pd.testing.assert_frame_equal(unaffected.drop(columns="crash_breadth"), baseline.drop(columns="crash_breadth"))
    assert crashes["crash_breadth"].eq(1.0).all()
    assert baseline.loc[baseline["date"].eq(dates[250]), "crash_breadth"].eq(2 / 3).all()
    mask = panel["ticker"].ne("CCC.HM")
    pd.testing.assert_frame_equal(panel.loc[mask], baseline_panel.loc[mask])
    # A later crash must know about the first one after liquidity was overridden.
    stocks = world[2].copy()
    later = stocks["ticker"].eq("CCC.HM") & stocks["date"].ge(dates[280])
    stocks.loc[later, "close"] *= 0.85
    ccc_rows = stocks["ticker"].eq("CCC.HM")
    stocks.loc[ccc_rows, "open"] = stocks.loc[ccc_rows, "close"].shift(1).bfill()
    stocks.loc[ccc_rows, "high"] = stocks.loc[ccc_rows, ["open", "close"]].max(axis=1) * 1.01
    stocks.loc[ccc_rows, "low"] = stocks.loc[ccc_rows, ["open", "close"]].min(axis=1) * 0.99
    stocks.loc[ccc_rows & stocks["date"].eq(dates[280]), "volume"] *= 5
    two_crashes, _, _ = _build((world[0], world[1], stocks, world[3], world[4]))
    later_event = two_crashes.loc[two_crashes["ticker"].eq("CCC.HM") & two_crashes["date"].eq(dates[280])]
    assert len(later_event) == 1
    assert later_event.iloc[0]["n_crash_prior_252"] == 1


def test_build_all_uses_given_ctx():
    world = _world()
    ctx = de_context.market_context_de(world[3], world[4])
    original = ctx.copy(deep=True)
    _, _, actual = _build(world, ctx=ctx)
    pd.testing.assert_index_equal(actual.index, ctx.index)
    dax = world[3].loc[world[3]["ticker"].eq("^GDAXI")].set_index("date")["close"]
    pd.testing.assert_series_equal(actual["spy_close"], dax, check_names=False)
    pd.testing.assert_frame_equal(actual[ctx.columns], ctx)
    pd.testing.assert_frame_equal(ctx, original)


def test_us_build_all_unchanged():
    dates = pd.bdate_range("2020-01-01", periods=300, name="date")
    market = pd.concat([_ohlcv(ticker, dates, seed) for seed, ticker in
                        enumerate(("SPY", "IWM", "MDY", "^VIX", "^VIX3M"), 1)], ignore_index=True)
    stocks = pd.concat([_ohlcv(ticker, dates, seed, volume=1e6, crash=True)
                        for seed, ticker in enumerate(("AAA", "BBB"), 10)], ignore_index=True)
    uni = pd.DataFrame({"ticker": ["AAA", "BBB"], "company": ["Alpha", "Beta"], "sector": "unknown",
                        "sub_industry": None, "cap_bucket": "large", "cik": [1, 2],
                        "wikipedia_title": None, "sector_etf": None, "is_current_member": True})
    membership = pd.DataFrame({"ticker": uni["ticker"], "index_name": "sp500",
                               "start_date": dates[0], "end_date": pd.NaT})
    filings = pd.DataFrame({"cik": [1, 2], "form": "8-K", "items": "2.02",
                            "accepted_et": dates[200] + pd.Timedelta(hours=8)})
    for fl in (filings, pd.DataFrame()):
        args = (uni, stocks, market, pd.DataFrame(), fl, pd.DataFrame(), pd.DataFrame())
        default = features.build_all(*args, membership=membership)
        explicit = features.build_all(*args, membership=membership, ctx=None, cap_buckets=None,
                                      liquid_by_membership=None)
        for actual, expected in zip(explicit, default):
            pd.testing.assert_frame_equal(actual, expected)
        assert not default[0].empty


def test_event_summary_plain_types():
    events, _, _ = _build(_world())
    summary = de_pipeline.event_summary(events)
    json.dumps(summary)
    assert summary["events"] == len(events)
    for name in ("by_family", "by_cap_bucket", "by_year"):
        assert sum(summary[name].values()) == len(events)
        assert all(type(v) is int for v in summary[name].values())
    assert all(type(year) is int for year in summary["by_year"])
    assert summary["first_date"] == events["date"].min().date().isoformat()
    assert summary["last_date"] == events["date"].max().date().isoformat()
    empty = de_pipeline.event_summary(pd.DataFrame())
    assert json.loads(json.dumps(empty)) == {
        "events": 0, "by_family": {}, "by_cap_bucket": {}, "by_year": {}, "first_date": "", "last_date": ""}


def test_transform_de_writes_tables(de_env, monkeypatch):
    universe, membership, stocks, market, macro = _world()
    for table, frame in ((de_universe.TABLE, universe), (de_membership.TABLE, membership),
                         (prices.PRICE_TABLE, stocks), (prices.MARKET_TABLE, market),
                         ("macro", macro.rename_axis("date").reset_index().rename(columns={"bund10y": "value"})
                          .assign(series="bund10y"))):
        store.replace_table(table, frame)
    result = de_pipeline.transform_de(save=False)
    assert set(result) == {"events", "panel", "market", "universe", "ctx"}
    assert not result["events"].empty
    assert not any(store.table_exists(name) for name in ("events", "daily_panel", "market_ctx"))
    saved = de_pipeline.transform_de()
    for table, key in (("events", "events"), ("daily_panel", "panel"), ("market_ctx", "ctx")):
        assert store.table_exists(table)
        pd.testing.assert_frame_equal(saved[key], result[key])
        expected = saved[key].where(saved[key].notna(), np.nan)
        pd.testing.assert_frame_equal(store.read_table(table), expected, check_dtype=False)
    assert "date" in saved["ctx"]
    assert config.WAREHOUSE == de_env / "warehouse.duckdb"
    steps = []

    def update(name, **kwargs):
        steps.append((name, kwargs))
        return {"step": name}

    monkeypatch.setattr(de_market, "update_de_market", lambda: update("market"))
    monkeypatch.setattr(de_market, "update_de_rates", lambda: update("rates"))
    monkeypatch.setattr(de_membership, "update_de_membership", lambda refresh=False: update("membership", refresh=refresh))
    monkeypatch.setattr(de_universe, "update_de_universe", lambda: update("universe"))
    report = de_pipeline.run_de(refresh=True)
    assert steps == [("market", {}), ("rates", {}), ("membership", {"refresh": True}), ("universe", {})]
    assert report == {**{name: {"step": name} for name in ("market", "rates", "membership", "universe")},
                      "transform": de_pipeline.event_summary(saved["events"])}


def test_transform_de_refuses_us(de_env, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])

    def unexpected(*args, **kwargs):
        pytest.fail("German pipeline attempted I/O for the US market")

    monkeypatch.setattr(store, "read_table", unexpected)
    monkeypatch.setattr(de_market, "update_de_market", unexpected)
    with pytest.raises(RuntimeError, match="DRIFTREV_MARKET=de"):
        de_pipeline.transform_de()
    with pytest.raises(RuntimeError, match="DRIFTREV_MARKET=de"):
        de_pipeline.run_de()
    assert not config.WAREHOUSE.exists()
    assert not de_env.exists()
