"""Offline German Yahoo symbol resolution, membership metadata and price loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from driftrev import config, de_membership, de_universe, prices, store


def _load_fixture() -> dict:
    path = Path(__file__).parent / "fixtures" / "de_yahoo" / "search.json"
    return json.loads(path.read_text(encoding="utf-8"))


FIXTURE = _load_fixture()
COMPANIES = {company["company_key"]: company for company in FIXTURE["companies"]}
EMPTY_OVERRIDES = pd.DataFrame(columns=["company_key", "symbol", "evidence"])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def unexpected_search(*args, **kwargs):
        pytest.fail("German symbol tests must not call Yahoo")

    monkeypatch.setattr(de_universe.yf, "Search", unexpected_search)


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


def _membership(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=de_membership.MEMBERSHIP_COLS)
    for column in ("start_date", "end_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def test_search_queries_match_fixture():
    assert len(FIXTURE["companies"]) == 201
    for company in FIXTURE["companies"]:
        assert de_universe.search_queries(company["company_key"], iter(company["names"])) == company["queries"]


def test_match_names():
    assert de_universe.match_names("Ströer (Unternehmen)", ["Ströer"]) == ["Ströer", "Ströer"]
    assert de_universe.match_names("Infineon Technologies", ["Infineon"]) == ["Infineon", "Infineon Technologies"]


def test_name_matches_examples():
    for yahoo_name, names, expected in (
        ("HUGO BOSS AG", ["Hugo Boss AG"], True),
        ("ams-OSRAM AG", ["Osram Licht"], False),
        ("OSRAM Licht AG", ["Osram Licht"], True),
        ("Porsche Automobil Hold", ["Porsche Automobil Holding SE"], True),
        ("Porsche Automobil Hold", ["Porsche AG"], False),
        ("STROEER SE + CO. KGAA", ["Ströer SE & Co. KGaA"], True),
        ("1&1 AG", ["1&1 AG"], True),
        ("COMPASS GROUP PLC ORD 11", ["1&1 AG"], False),
        ("AG SE + CO. KGAA", ["Company AG"], False),
        ("", ["Hugo Boss"], False),
        ("Hugo Boss", [], False),
        ("Hugo Boss", ["Hugo", "Boss"], False),
        ("Siemens Aktiengesellschaft", ["Siemens"], True),
        ("Siemens Energy AG", ["Siemens"], False),
        ("Bayerische Motoren Werke Aktiengesellschaft", ["Bayer AG"], False),
    ):
        assert de_universe.name_matches(yahoo_name, iter(names)) is expected


def test_pick_listing_spot_checks():
    for key, symbol, price_source, matched_query in (
        ("Hugo Boss", "BOSS.DE", "xetra", "Hugo Boss"),
        ("Wirecard", "WDI.HM", "hamburg", "Wirecard"),
        ("Ströer (Unternehmen)", "SAX.DE", "xetra", "Stroeer"),
        ("Osram", "OSR.HM", "hamburg", "Osram Licht"),
        ("1&1", "1U1.DE", "xetra", "1&1 AG"),
        ("SAP", "SAP.DE", "xetra", "SAP"),
        ("Airbus", "AIR.DE", "xetra", "Airbus"),
        ("Rocket Internet", "RKET.HM", "hamburg", "Rocket Internet"),
        ("Porsche Automobil Holding", "PAH3.DE", "xetra", "Porsche Automobil Holding"),
    ):
        company = COMPANIES[key]
        listing = de_universe.pick_listing(FIXTURE["queries"], company["queries"], company["names"])
        assert listing is not None, key
        assert listing["symbol"] == symbol, key
        assert listing["price_source"] == price_source
        assert listing["exchange"] == ("GER" if price_source == "xetra" else "HAM")
        assert listing["matched_query"] == matched_query
    porsche = COMPANIES["Porsche"]
    listing = de_universe.pick_listing(FIXTURE["queries"], porsche["queries"], ["Porsche AG"])
    assert listing is None or listing["symbol"] != "PAH3.DE"

    for key, symbol in (
        ("Siemens", "SIE.DE"),
        ("Bayer AG", "BAYN.DE"),
        ("DHL AG", None),  # Yahoo shortname carries a share-class suffix; resolved by override
        ("Infineon Technologies", "IFX.DE"),
        ("Deutsche Bank", "DBK.DE"),
        ("Software AG", None),
    ):
        company = COMPANIES[key]
        listing = de_universe.pick_listing(FIXTURE["queries"], company["queries"], company["names"], company_key=key)
        if symbol is None:
            assert listing is None, key
        else:
            assert listing is not None, key
            assert listing["symbol"] == symbol, key


def test_resolve_symbols_fixture_report():
    companies = pd.DataFrame(FIXTURE["companies"])[["company_key", "names"]]
    resolved, diag = de_universe.resolve_symbols(companies, FIXTURE["queries"], EMPTY_OVERRIDES)
    assert diag["companies"] == 201
    assert diag["override"] == 0
    assert diag["search_xetra"] + diag["search_hamburg"] + len(diag["unresolved"]) == 201
    assert len(resolved) == 201
    assert diag["unresolved"] == sorted(diag["unresolved"])
    assert resolved.loc[resolved["resolved_by"].eq("unresolved"),
                        ["symbol", "exchange", "price_source", "yahoo_name", "matched_query"]].isna().all().all()
    print(diag)

    # Shared symbols are reported for every company rather than silently deduplicated.
    duplicate = companies.loc[companies["company_key"].eq("Hugo Boss")].copy()
    duplicate["company_key"] = "Another Hugo Boss"
    _, shared_diag = de_universe.resolve_symbols(pd.concat([companies, duplicate]), FIXTURE["queries"], EMPTY_OVERRIDES)
    assert shared_diag["shared_symbols"]["BOSS.DE"] == ["Another Hugo Boss", "Hugo Boss"]


def test_override_wins(de_env):
    overrides_path = de_env.parent / "overrides.csv"
    overrides_path.write_text("company_key,symbol,evidence\nOsram,OSR.DE,test\n", encoding="utf-8")
    companies = pd.DataFrame([COMPANIES["Osram"]])[["company_key", "names"]]
    overrides = de_universe.load_overrides(overrides_path)
    resolved, diag = de_universe.resolve_symbols(companies, FIXTURE["queries"], overrides)
    row = resolved.iloc[0]
    assert row["symbol"] == "OSR.DE"
    assert row["resolved_by"] == "override"
    assert row["exchange"] == "GER"
    assert row["price_source"] == "xetra"
    assert row["yahoo_name"] is None and row["matched_query"] is None
    assert diag["override"] == 1 and diag["search_xetra"] == 0
    real = de_universe.load_overrides()
    assert list(real.columns) == ["company_key", "symbol", "evidence"]
    assert real["symbol"].notna().all() and real["evidence"].str.len().gt(10).all()
    assert not real["company_key"].duplicated().any()
    for symbol, exchange, price_source in (("OSR.HM", "HAM", "hamburg"), ("OTHER", None, "other")):
        manual = overrides.assign(symbol=symbol)
        resolved, _ = de_universe.resolve_symbols(companies, {}, manual)
        assert resolved.iloc[0]["exchange"] == exchange
        assert resolved.iloc[0]["price_source"] == price_source


def test_failed_queries_are_skipped():
    quote = {"symbol": "BOSS.DE", "exchange": "GER", "quoteType": "EQUITY",
             "longname": "Hugo Boss AG", "shortname": "Wrong company"}
    results = {"first": None, "second": [quote]}
    listing = de_universe.pick_listing(results, ["missing", "first", "second"], ["Hugo Boss"])
    assert listing["symbol"] == "BOSS.DE" and listing["matched_query"] == "second"
    assert de_universe.pick_listing({"first": None, "second": None}, ["first", "second"], ["Hugo Boss"]) is None
    # Exchange preference beats query/Yahoo order; funds and foreign listings cannot win.
    results = {"first": [dict(quote, symbol="BOSS.HM", exchange="HAM")],
               "second": [dict(quote, symbol="FUND.DE", quoteType="ETF"),
                          dict(quote, symbol="FOREIGN", exchange="NYQ"), quote, dict(quote, symbol="LATER.DE")]}
    assert de_universe.pick_listing(results, ["first", "second"], ["Hugo Boss"])["symbol"] == "BOSS.DE"
    for longname in (None, "", 1):
        short_only = dict(quote, longname=longname, shortname="  HUGO BOSS AG  ")
        picked = de_universe.pick_listing({"q": [short_only]}, ["q"], ["Hugo Boss"])
        assert picked["yahoo_name"] == "HUGO BOSS AG"


def test_build_universe_columns_and_dates():
    membership = _membership([
        ("A", "New A", "mdax", "2020-01-01", None),
        ("A", "Old A", "dax", "2014-01-01", "2020-01-01"),
        ("B", "B AG", "sdax", "2015-01-01", "2021-01-01"),
    ])
    companies = de_universe.companies_since(membership, "2014-01-01")
    resolved, _ = de_universe.resolve_symbols(companies, {}, EMPTY_OVERRIDES)
    universe = de_universe.build_universe(membership, resolved)
    assert list(universe.columns) == de_universe.UNIVERSE_COLS
    a, b = universe.set_index("company_key").loc[["A", "B"]].to_dict("records")
    assert a["name"] == "New A" and a["index_code"] == "mdax" and a["cap_bucket"] == "mid"
    assert a["is_current_member"] and pd.isna(a["last_date"])
    assert a["first_date"] == pd.Timestamp("2014-01-01")
    assert not b["is_current_member"] and b["last_date"] == pd.Timestamp("2021-01-01")
    assert b["name"] == "B AG" and b["cap_bucket"] == "small"
    assert universe["resolved_by"].eq("unresolved").all()
    assert de_universe.build_universe(membership, resolved.iloc[:0]).empty


def test_companies_since():
    membership = _membership([
        ("A", "Obsolete", "dax", "2000-01-01", "2013-12-31"),
        ("A", "Z name", "mdax", "2010-01-01", "2014-01-01"),
        ("A", "A name", "sdax", "2015-01-01", None),
        ("A", "A name", "mdax", "2016-01-01", None),
        ("B", "Excluded", "sdax", "2000-01-01", "2013-12-31"),
        ("C", "Open", "dax", "2000-01-01", None),
    ])
    companies = de_universe.companies_since(membership, "2014-01-01")
    assert list(companies.columns) == ["company_key", "names"]
    assert companies.to_dict("records") == [
        {"company_key": "A", "names": ["A name", "Z name"]}, {"company_key": "C", "names": ["Open"]},
    ]
    assert de_universe.companies_since(membership.iloc[:0], "2014-01-01").empty


def test_update_refuses_us_market(de_env, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])
    for function, args in ((de_universe.update_de_universe, ()), (de_universe.cached_search, (["a"],)),
                           (de_universe.search_yahoo, ("a",))):
        with pytest.raises(RuntimeError, match="DRIFTREV_MARKET=de"):
            function(*args)
    assert not config.WAREHOUSE.exists()
    assert not de_env.exists()


def test_update_de_universe_offline(de_env, monkeypatch):
    with pytest.raises(RuntimeError, match="run update_de_membership first"):
        de_universe.update_de_universe()
    membership = _membership([
        ("Hugo Boss", "Hugo Boss AG", "sdax", "2014-01-01", None),
        ("Wirecard", "Wirecard", "dax", "2014-01-01", "2020-08-24"),
        ("Osram", "Osram Licht", "mdax", "2014-01-01", "2021-03-22"),
    ])
    store.replace_table(de_membership.TABLE, membership)
    queries_seen, prices_seen = [], []
    price_report = {"table": "prices", "rows_written": 0, "failed": [], "resplit": []}

    def fake_search(queries, refresh=False):
        queries_seen.append((list(queries), refresh))
        return FIXTURE["queries"]

    def fake_prices(tickers):
        prices_seen.append(tickers)
        return price_report

    monkeypatch.setattr(de_universe, "cached_search", fake_search)
    monkeypatch.setattr(prices, "update_prices", fake_prices)
    report = de_universe.update_de_universe(refresh_search=True)
    stored = store.read_table(de_universe.TABLE)
    assert len(stored) == report["rows"] == 3
    assert config.WAREHOUSE == de_env / "warehouse.duckdb"
    assert list(stored.columns) == de_universe.UNIVERSE_COLS
    assert stored.set_index("company_key")["symbol"].to_dict() == {
        "Hugo Boss": "BOSS.DE", "Wirecard": "WDI.HM", "Osram": "OSR.HM",
    }
    assert prices_seen == [["BOSS.DE", "OSR.HM", "WDI.HM"]]
    assert queries_seen[0] == (["Hugo Boss", "Hugo Boss AG", "Osram", "Osram Licht", "Wirecard"], True)
    assert report["table"] == "universe" and report["prices"] == price_report
    assert report["companies"] == 3 and report["override"] == 0
    assert report["search_xetra"] == 1 and report["search_hamburg"] == 2 and report["unresolved"] == []
    assert de_universe.update_de_universe(load_prices=False)["prices"] is None
    assert len(prices_seen) == 1
    assert queries_seen[1][1] is False


def test_cached_search_uses_cache(de_env, monkeypatch):
    calls = []

    def fake_search(query):
        calls.append(query)
        return None if query == "failed" else []

    with monkeypatch.context() as patch:
        patch.setattr(de_universe, "search_yahoo", fake_search)
        assert de_universe.cached_search(iter(["a", "b", "a"])) == {"a": [], "b": []}
        assert calls == ["a", "b"]
        # stored in the warehouse table (survives CI runs), not in cache/
        stored = store.read_table(de_universe.SEARCH_TABLE)
        assert {r.query: json.loads(r.quotes_json) for r in stored.itertuples()} == {"a": [], "b": []}
        calls.clear()
        assert de_universe.cached_search(["a", "b"]) == {"a": [], "b": []}
        assert calls == []
        assert de_universe.cached_search(["failed"]) == {"failed": None}
        assert "failed" not in set(store.read_table(de_universe.SEARCH_TABLE)["query"])
        de_universe.cached_search(["failed"])
        assert calls == ["failed", "failed"]
        calls.clear()
        de_universe.cached_search(["a", "b"], refresh=True)
        assert calls == ["a", "b"]

    # Exercise the real wrapper with an offline Search substitute and no waits.
    rate_calls, waits, searches = [], [], []

    def fake_limiter(rps):
        rate_calls.append(rps)
        return SimpleNamespace(wait=lambda: waits.append(True))

    def fake_yahoo(query, **kwargs):
        searches.append((query, kwargs))
        if query == "failed":
            raise RuntimeError("offline failure")
        return SimpleNamespace(quotes=[{"symbol": "A.DE", "exchange": "GER", "quoteType": "EQUITY",
                                       "shortname": "A", "longname": "A AG", "unused": "discard"}])

    monkeypatch.setattr(de_universe, "_search_limiter", None)
    monkeypatch.setattr(store, "RateLimiter", fake_limiter)
    monkeypatch.setattr(de_universe.yf, "Search", fake_yahoo)
    assert rate_calls == []
    assert de_universe.search_yahoo("A") == [{"symbol": "A.DE", "exchange": "GER", "quoteType": "EQUITY",
                                              "shortname": "A", "longname": "A AG"}]
    assert de_universe.search_yahoo("failed") is None
    assert rate_calls == [config.SETTINGS.de_universe.search_rps] and len(waits) == 2
    assert searches == [(query, {"max_results": config.SETTINGS.de_universe.search_max_results, "news_count": 0})
                        for query in ("A", "failed")]
