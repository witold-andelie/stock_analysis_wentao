"""Fetch cadence for slowly-changing web sources (Wikipedia membership pages, Yahoo symbol search)."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from driftrev import config, de_membership, de_pipeline, de_universe, membership, pipeline, store


@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for name, path in (("DATA_DIR", data), ("WAREHOUSE", data / "warehouse.duckdb"), ("CACHE_DIR", data / "cache"),
                       ("SNAPSHOT_DIR", data / "snapshot"), ("ARTIFACTS_DIR", data / "artifacts")):
        monkeypatch.setattr(config, name, path)
    return data


@pytest.fixture
def de_warehouse(warehouse, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    return warehouse


def _boom(*args, **kwargs):
    raise AssertionError("network fetch must not happen")


# ---------------------------------------------------------------- fetch log
def test_fetch_log_roundtrip_and_due(warehouse):
    assert store.last_fetch("x") is None
    assert store.fetch_due("x", 7)
    store.record_fetch("x")
    assert not store.fetch_due("x", 7)
    assert store.fetch_due("x", 7, force=True)
    store.record_fetch("x", pd.Timestamp.now() - pd.Timedelta(days=8))
    assert store.fetch_due("x", 7)
    assert store.fetch_age_days("x") >= 8


def test_us_membership_skips_when_fresh(warehouse, monkeypatch):
    store.replace_table(membership.TABLE, pd.DataFrame({"ticker": ["AAA"], "index_name": ["sp500"],
                                                        "start_date": [pd.Timestamp("2000-01-01")], "end_date": [pd.NaT]}))
    store.record_fetch("us_membership")
    monkeypatch.setattr(membership, "fetch_changes", _boom)
    report = membership.update_membership()
    assert report["skipped"] is True and report["table"] == membership.TABLE
    calls = []
    monkeypatch.setattr(membership, "fetch_changes", lambda refresh=False: calls.append(refresh) or (_ for _ in ()).throw(RuntimeError("stop")))
    store.replace_table("universe", pd.DataFrame({"ticker": ["AAA"], "index_name": ["sp500"], "is_current_member": [True]}))
    with pytest.raises(RuntimeError, match="stop"):
        membership.update_membership(refresh=True)
    assert calls == [True]


def test_de_membership_skips_when_fresh(de_warehouse, monkeypatch):
    store.replace_table(de_membership.TABLE, pd.DataFrame(columns=de_membership.MEMBERSHIP_COLS).astype(object).assign(
        company_key=["A"], name=["A"], index_code=["dax"], start_date=[pd.Timestamp("2000-01-01")], end_date=[pd.NaT]))
    store.record_fetch("de_membership")
    monkeypatch.setattr(de_membership, "fetch_pages", _boom)
    monkeypatch.setattr(de_membership, "fetch_redirects", _boom)
    assert de_membership.update_de_membership()["skipped"] is True
    monkeypatch.setattr(de_membership, "fetch_pages", lambda refresh=False: (_ for _ in ()).throw(RuntimeError("fetched")))
    with pytest.raises(RuntimeError, match="fetched"):
        de_membership.update_de_membership(refresh=True)


# ---------------------------------------------------------------- Yahoo search cache
def test_yahoo_search_only_new_or_stale_queries(de_warehouse, monkeypatch):
    old = pd.Timestamp.now() - pd.Timedelta(days=91)
    store.upsert(de_universe.SEARCH_TABLE, pd.DataFrame({
        "query": ["A", "B"], "quotes_json": [json.dumps([{"symbol": "A.DE"}]), "[]"],
        "fetched_at": [pd.Timestamp.now(), old]}), ["query"])
    calls = []

    def fake(query):
        calls.append(query)
        return None if query == "C" else []

    monkeypatch.setattr(de_universe, "search_yahoo", fake)
    out = de_universe.cached_search(["A", "B", "C"])
    assert calls == ["B", "C"]
    assert out == {"A": [{"symbol": "A.DE"}], "B": [], "C": None}
    assert "C" not in set(store.read_table(de_universe.SEARCH_TABLE)["query"])
    calls.clear()
    de_universe.cached_search(["A", "B", "C"])
    assert calls == ["C"]


def test_legacy_json_cache_imported_once(de_warehouse, monkeypatch):
    legacy = config.CACHE_DIR / "yahoo_search_de.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"Hugo Boss": [{"symbol": "BOSS.DE"}], "failed": None}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(de_universe, "search_yahoo", lambda q: calls.append(q) or [])
    out = de_universe.cached_search(["Hugo Boss"])
    assert calls == [] and out == {"Hugo Boss": [{"symbol": "BOSS.DE"}]}
    assert set(store.read_table(de_universe.SEARCH_TABLE)["query"]) == {"Hugo Boss"}


# ---------------------------------------------------------------- manual override
def test_refresh_sources_flag_reaches_de_ingest(de_warehouse, monkeypatch):
    seen = []

    def ingest_de(refresh=False, refresh_membership=False):
        seen.append(refresh)
        raise RuntimeError("stop after ingest")

    monkeypatch.setattr(de_pipeline, "ingest_de", ingest_de)
    monkeypatch.setattr(pipeline.artifacts, "append_run_log", lambda entry: None)
    for flag in (False, True):
        with pytest.raises(RuntimeError, match="stop after ingest"):
            pipeline.run_pipeline(mode="update", refresh_sources=flag)
    assert seen == [False, True]


def test_refresh_membership_de_forces_membership_only(de_warehouse, monkeypatch):
    from driftrev import de_market

    seen = {"membership": [], "search": []}

    def update_membership(refresh=False):
        seen["membership"].append(refresh)
        return {}

    def update_universe(refresh_search=False):
        seen["search"].append(refresh_search)
        raise RuntimeError("stop")

    monkeypatch.setattr(de_market, "update_de_market", lambda: {})
    monkeypatch.setattr(de_market, "update_de_rates", lambda: {})
    monkeypatch.setattr(de_membership, "update_de_membership", update_membership)
    monkeypatch.setattr(de_universe, "update_de_universe", update_universe)
    for kwargs in ({"refresh_membership": True}, {"refresh": True}, {}):
        with pytest.raises(RuntimeError, match="stop"):
            de_pipeline.ingest_de(**kwargs)
    assert seen["membership"] == [True, True, False]
    assert seen["search"] == [False, True, False]


def test_refresh_membership_flag_reaches_ingest(warehouse, monkeypatch):
    seen = []

    def ingest_de(refresh=False, refresh_membership=False):
        seen.append(("de", refresh, refresh_membership))
        raise RuntimeError("stop after ingest")

    def ingest(refresh_sources=False, refresh_membership=False):
        seen.append(("us", refresh_sources, refresh_membership))
        raise RuntimeError("stop after ingest")

    monkeypatch.setattr(de_pipeline, "ingest_de", ingest_de)
    monkeypatch.setattr(pipeline, "ingest", ingest)
    monkeypatch.setattr(pipeline.artifacts, "append_run_log", lambda entry: None)
    for market in ("de", "us"):
        monkeypatch.setattr(config, "MARKET", config.MARKETS[market])
        with pytest.raises(RuntimeError, match="stop after ingest"):
            pipeline.run_pipeline(mode="update", refresh_membership=True)
    assert seen == [("de", False, True), ("us", False, True)]


def test_us_ingest_refresh_membership_forces_membership(warehouse, monkeypatch):
    seen = []

    def update_membership(refresh=False):
        seen.append(refresh)
        raise RuntimeError("stop")

    monkeypatch.setattr(pipeline.universe, "load_or_fetch", lambda refresh=False: pd.DataFrame())
    monkeypatch.setattr(store, "table_exists", lambda name: True)
    monkeypatch.setattr(membership, "update_membership", update_membership)
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.ingest(refresh_membership=True)
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.ingest()
    assert seen == [True, False]


def test_cli_refresh_membership_flag(monkeypatch):
    seen = {}

    def run_pipeline(*args, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(pipeline, "run_pipeline", run_pipeline)
    monkeypatch.setattr(sys, "argv", ["driftrev.pipeline", "--mode", "update", "--refresh-membership"])
    monkeypatch.setattr(config, "require_implemented_market", lambda: None)
    pipeline.main()
    assert seen["refresh_membership"] is True
    assert seen["refresh_sources"] is False


def test_cadence_defaults():
    c = config.SETTINGS.fetch_cadence
    assert c.membership_refresh_days == 7.0 and c.symbol_search_refresh_days == 90.0
