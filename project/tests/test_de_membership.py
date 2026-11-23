"""Offline German index parsing, identity resolution and membership reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from driftrev import config, de_membership, store

FIXTURES = Path(__file__).parent / "fixtures" / "de_wiki"


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


def _pages() -> dict[str, str]:
    return {key: (FIXTURES / filename).read_text(encoding="utf-8") for key, filename in (
        ("dax", "de_DAX.html"), ("mdax", "de_MDAX.html"), ("sdax", "de_SDAX.html"),
        ("mdax_history", "de_MDAX__Zusammensetzungshistorie.html"),
    )}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict]:
    pages = _pages()
    current = pd.concat([de_membership.parse_current(pages[index], index)
                         for index in de_membership.INDEXES], ignore_index=True)
    dax = de_membership.parse_changes(pages["dax"], "dax")
    mdax = de_membership.parse_changes(pages["mdax_history"], "mdax")
    changes = pd.concat([dax, mdax, de_membership.derive_sdax_changes(mdax)], ignore_index=True)
    cells = list(current[["name", "link"]].itertuples(index=False, name=None))
    for side in ("removed", "added"):
        cells.extend(changes[[f"{side}_name", f"{side}_link"]].itertuples(index=False, name=None))
    redirects = de_membership.load_redirects(json.loads((FIXTURES / "redirects.json").read_text(encoding="utf-8")))
    keys = de_membership.resolve_keys(cells, redirects, de_membership.load_aliases())
    membership, diag = de_membership.build_de_membership(
        changes, current, keys, event_identities=de_membership.load_event_identities())
    return current, changes, keys, membership, diag


def _key(keys: dict, name: str) -> str:
    matches = {key for (text, _), key in keys.items()
               if de_membership.name_key(text) == de_membership.name_key(name)}
    assert len(matches) == 1, (name, matches)
    return matches.pop()


def test_parse_german_date():
    for text, expected in (
        ("24.08.2020", "2020-08-24"), ("22. Juni 1998", "1998-06-22"),
        ("20. Sep. 2021", "2021-09-20"), ("21. März 2022", "2022-03-21"),
        ("19. September 1994", "1994-09-19"), ("1. Juli 1988 [ 3 ]", "1988-07-01"),
    ):
        assert de_membership.parse_german_date(text) == pd.Timestamp(expected)
    for text in ("—", "", None):
        assert pd.isna(de_membership.parse_german_date(text))
    months = ("Januar Jänner Jan", "Februar Feb", "März Mär Mrz", "April Apr", "Mai",
              "Juni Jun", "Juli Jul", "August Aug", "September Sept Sep", "Oktober Okt",
              "November Nov", "Dezember Dez")
    for month, names in enumerate(months, start=1):
        for name in names.split():
            for suffix in ("", "."):
                text = f"[ 44 ] seit 2. {name.upper()}{suffix} 2001; 3. Mai 2002"
                assert de_membership.parse_german_date(text) == pd.Timestamp(2001, month, 2)


def test_parse_current_counts():
    pages = _pages()
    frames = {index: de_membership.parse_current(pages[index], index) for index in de_membership.INDEXES}
    assert {index: len(frame) for index, frame in frames.items()} == {"dax": 40, "mdax": 50, "sdax": 70}
    assert frames["dax"].iloc[0]["name"] == "Adidas"
    assert frames["dax"].iloc[0]["since"] == pd.Timestamp("1998-06-22")
    for index, name, since in (("dax", "Daimler Truck", "2022-03-21"),
                               ("mdax", "Aixtron SE", "2020-08-24"), ("sdax", "1&1 AG", "2025-09-22")):
        row = frames[index].loc[frames[index]["name"] == name]
        assert len(row) == 1
        assert row.iloc[0]["since"] == pd.Timestamp(since)
    for frame in frames.values():
        assert list(frame.columns) == de_membership.CURRENT_COLS
        assert frame["since"].notna().all()


def test_parse_dax_changes():
    changes = de_membership.parse_changes(_pages()["dax"], "dax")
    assert list(changes.columns) == de_membership.CHANGE_COLS
    assert len(changes) == 77
    assert tuple(changes.iloc[0][["date", "removed_name", "added_name"]]) == (
        pd.Timestamp("1990-09-03"), "Feldmühle Nobel", "Metallgesellschaft")
    assert tuple(changes.iloc[-1][["date", "removed_name", "added_name"]]) == (
        pd.Timestamp("2026-06-22"), "Porsche Automobil Holding", "Hochtief")
    reform = changes.loc[changes["date"] == pd.Timestamp("2021-09-20")]
    assert len(reform) == 10
    assert all(value is None for value in reform["removed_name"])
    assert ((changes["date"] == pd.Timestamp("2020-08-24"))
            & changes["removed_name"].eq("Wirecard") & changes["added_name"].eq("Delivery Hero")).sum() == 1
    assert not changes[["removed_name", "added_name"]].stack().str.match(
        r"^\d{1,2}\.\s?\w+\.?\s?\d{4}$|^\d{2}\.\d{2}\.\d{4}$").any()
    # Reordered headers, company rowspans and colspans also preserve one event per origin cell.
    spanned = de_membership.parse_changes("""
        <table><tr><th>Aufgenommene Unternehmen</th><th>Anmerkung</th>
        <th>Datum</th><th>Ausgeschiedene Unternehmen</th></tr>
        <tr><td>Undated</td><td></td><td></td><td>Undated</td></tr>
        <tr><td rowspan="2">First</td><td rowspan="3">Remark</td>
        <td rowspan="2">01.01.2000</td><td>Old First</td></tr>
        <tr><td>Old Second</td></tr>
        <tr><td>Second</td><td>02.01.2000</td><td rowspan="2">Old Third</td></tr>
        <tr><td colspan="2"></td><td>03.01.2000</td></tr>
        <tr><td colspan="2">Third</td><td>04.01.2000</td><td>Old Fourth</td></tr>
        </table>
        """, "dax")
    assert list(spanned[["date", "removed_name", "added_name"]].itertuples(index=False, name=None)) == [
        (pd.Timestamp("2000-01-01"), "Old First", "First"),
        (pd.Timestamp("2000-01-01"), "Old Second", None),
        (pd.Timestamp("2000-01-02"), "Old Third", "Second"),
        (pd.Timestamp("2000-01-04"), "Old Fourth", "Third"),
    ]


def test_parse_mdax_changes_and_markers():
    changes = de_membership.parse_changes(_pages()["mdax_history"], "mdax")
    assert len(changes) == 293
    assert changes["removed_other"].value_counts(dropna=False).to_dict() == {"SDAX": 138, "DAX": 47, None: 108}
    assert changes["added_other"].value_counts(dropna=False).to_dict() == {"SDAX": 128, "DAX": 31, None: 134}
    assert tuple(changes.iloc[-1][["date", "removed_name", "removed_other", "added_name", "added_other"]]) == (
        pd.Timestamp("2026-09-21"), "Hugo Boss", "SDAX", "Ströer", "SDAX")


def test_derive_sdax_changes():
    mdax = de_membership.parse_changes(_pages()["mdax_history"], "mdax")
    sdax = de_membership.derive_sdax_changes(mdax)
    assert list(sdax.columns) == de_membership.CHANGE_COLS
    assert len(sdax) == 266
    assert sdax["added_name"].notna().sum() == 138
    assert sdax["removed_name"].notna().sum() == 128
    assert sdax["index_code"].eq("sdax").all()
    assert sdax[["removed_other", "added_other"]].isna().all().all()


def test_name_key_examples():
    examples = {"Aixtron SE": "aixtron", "Aroundtown S.A.": "aroundtown", "1&1 AG": "1 1",
                "Ströer SE & Co. KGaA": "stroeer", "Ströer": "stroeer", "E.ON SE": "e on",
                "Porsche Automobil Holding SE": "porsche automobil holding", "Preussag (heute TUI )": "preussag",
                "Henkel AG & Co. KGaA": "henkel", "Hugo Boss AG": "hugo boss"}
    for text, expected in examples.items():
        assert de_membership.name_key(text) == expected


def test_canonical_titles():
    redirects = de_membership.load_redirects(json.loads((FIXTURES / "redirects.json").read_text(encoding="utf-8")))
    for title, expected in (("Ströer Media", "Ströer (Unternehmen)"), ("Porsche AG", "Porsche"),
                            ("Daimler Truck Holding", "Daimler Truck"), ("Hugo Boss", "Hugo Boss")):
        assert de_membership.canonical_title(title, redirects) == expected


def test_resolve_keys_identity():
    _, _, keys, _, _ = _load()
    assert keys["Daimler Truck Holding", None] == keys["Daimler Truck", "Daimler Truck"]
    assert keys["Ströer", None] == keys["Ströer Media", "Ströer Media"] == keys[
        "Ströer SE & Co. KGaA", "Ströer (Unternehmen)"]
    cells = [("Lufthansa", "Lufthansa"), ("Deutsche Lufthansa AG", "Lufthansa Group")]
    redirects = de_membership.load_redirects(json.loads((FIXTURES / "redirects.json").read_text(encoding="utf-8")))
    with_alias = de_membership.resolve_keys(cells, redirects, de_membership.load_aliases())
    without_alias = de_membership.resolve_keys(cells, redirects, pd.DataFrame(
        columns=["alias_text", "canonical_title", "evidence"]))
    assert with_alias[cells[0]] == with_alias[cells[1]]
    assert without_alias[cells[0]] != without_alias[cells[1]]


def test_same_day_swap_rule_synthetic():
    rows = [
        ("2000-01-01", None, "Open old", "dax"),
        ("2000-01-02", "Open old", "Open new", "dax"),
        ("2000-01-02", "Open old", "Open new", "dax"),
        ("2000-01-03", "Open new", None, "dax"),
        ("2000-01-02", "First old", "First new", "dax"),
        ("2000-01-03", "First new", None, "dax"),
        ("2000-01-02", "First old", "First new", "sdax"),
        ("2000-01-03", "First new", None, "sdax"),
        ("2000-01-01", None, "Closed old", "dax"),
        ("2000-01-02", "Closed old", None, "dax"),
        ("2000-01-03", "Closed old", "Closed new", "dax"),
        ("2000-01-04", "Closed new", None, "dax"),
        ("2000-01-01", None, "Duplicate", "dax"),
        ("2000-01-01", None, "Duplicate", "dax"),
        ("2000-01-02", None, "Duplicate", "dax"),
        ("2000-01-03", "Duplicate", None, "dax"),
        ("2000-01-03", "Duplicate", None, "dax"),
        ("2000-01-04", "Duplicate", None, "dax"),
    ]
    changes = pd.DataFrame([
        (pd.Timestamp(date), removed, None, None, added, None, None, index)
        for date, removed, added, index in rows
    ], columns=de_membership.CHANGE_COLS)
    current = pd.DataFrame(columns=de_membership.CURRENT_COLS)
    keys = {(name, None): name.split()[0] for _, removed, added, _ in rows
            for name in (removed, added) if name is not None}
    membership, diag = de_membership.build_de_membership(changes, current, keys)

    def intervals(key: str, index: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        selected = membership.loc[membership["company_key"].eq(key) & membership["index_code"].eq(index)]
        return list(selected[["start_date", "end_date"]].itertuples(index=False, name=None))

    assert intervals("Open", "dax") == [(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"))]
    for index in ("dax", "sdax"):
        assert intervals("First", index) == [(de_membership.FOREVER, pd.Timestamp("2000-01-03"))]
    assert intervals("Closed", "dax") == [
        (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02")),
        (pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-04")),
    ]
    assert intervals("Duplicate", "dax") == [(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"))]
    assert diag["same_day_swap"] == [
        ("dax", "Closed", pd.Timestamp("2000-01-03")),
        ("dax", "First", pd.Timestamp("2000-01-02")),
        ("dax", "Open", pd.Timestamp("2000-01-02")),
        ("sdax", "First", pd.Timestamp("2000-01-02")),
    ]
    assert diag["start_unknown"] == [("sdax", "First")]
    assert diag["double_add"] == 2
    assert diag["double_remove"] == 2


def test_event_identity_overrides_synthetic():
    rows = [
        ("2000-01-01", None, "Shared AG", "mdax"),
        ("2000-01-02", "SHARED AG", None, "mdax"),
        ("2000-01-03", "Shared AG", None, "mdax"),
        ("2000-01-02", "SHARED AG", None, "dax"),
    ]
    changes = pd.DataFrame([
        (pd.Timestamp(date), removed, None, None, added, None, None, index)
        for date, removed, added, index in rows
    ], columns=de_membership.CHANGE_COLS)
    current = pd.DataFrame([
        ("mdax", "Shared AG", None, pd.Timestamp("2000-01-04")),
    ], columns=de_membership.CURRENT_COLS)
    keys = {("Shared AG", None): "Shared", ("SHARED AG", None): "Shared"}
    identities = pd.DataFrame([
        ("mdax", pd.Timestamp("2000-01-02"), "removed", "shared", "Old Shared", "Separate old company"),
    ], columns=["index_code", "date", "side", "text", "company_key", "evidence"])
    before, before_diag = de_membership.build_de_membership(changes, current, keys)
    membership, diag = de_membership.build_de_membership(changes, current, keys, event_identities=identities)

    def intervals(frame: pd.DataFrame, key: str, index: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        selected = frame.loc[frame["company_key"].eq(key) & frame["index_code"].eq(index)]
        return list(selected[["start_date", "end_date"]].itertuples(index=False, name=None))

    assert intervals(before, "Old Shared", "mdax") == []
    assert intervals(before, "Shared", "mdax")[0] == (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"))
    assert before_diag["event_identity"] == []
    assert intervals(membership, "Old Shared", "mdax") == [
        (de_membership.FOREVER, pd.Timestamp("2000-01-02")),
    ]
    shared = intervals(membership, "Shared", "mdax")
    assert len(shared) == 2
    assert shared[0] == (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"))
    assert shared[1][0] == pd.Timestamp("2000-01-04") and pd.isna(shared[1][1])
    assert intervals(membership, "Shared", "dax") == [(de_membership.FOREVER, pd.Timestamp("2000-01-02"))]
    assert diag["event_identity"] == [("mdax", pd.Timestamp("2000-01-02"), "removed", "Old Shared")]
    with pytest.raises(ValueError, match="text='shared'.*matched 0 events"):
        de_membership.build_de_membership(
            changes, current, keys, event_identities=identities.assign(date=pd.Timestamp("1999-01-01")))
    ambiguous = pd.concat([changes, changes.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="text='shared'.*matched 2 events"):
        de_membership.build_de_membership(ambiguous, current, keys, event_identities=identities)


def test_gea_and_metro_identities():
    current, changes, keys, membership, diag = _load()

    def intervals(key: str, index: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        rows = membership.loc[membership["company_key"].eq(key) & membership["index_code"].eq(index)]
        return list(rows[["start_date", "end_date"]].itertuples(index=False, name=None))

    assert intervals("GEA Group", "mdax") == [(pd.Timestamp("1996-11-18"), pd.Timestamp("2025-09-22"))]
    gea_dax = intervals("GEA Group", "dax")
    assert len(gea_dax) == 2
    assert gea_dax[0] == (pd.Timestamp("1990-09-03"), pd.Timestamp("1996-11-18"))
    assert gea_dax[1][0] == pd.Timestamp("2025-09-22") and pd.isna(gea_dax[1][1])
    assert intervals("GEA AG (1999)", "mdax") == [(de_membership.FOREVER, pd.Timestamp("1999-09-20"))]
    assert intervals("Ceconomy", "dax") == [(pd.Timestamp("1996-07-22"), pd.Timestamp("2012-09-24"))]
    assert intervals("Ceconomy", "mdax") == [(pd.Timestamp("2012-09-24"), pd.Timestamp("2018-09-24"))]
    assert intervals("Ceconomy", "sdax") == []
    assert ("sdax", "Ceconomy") in diag["orphans"]
    assert intervals("Metro AG", "dax") == []
    assert intervals("Metro AG", "mdax") == [(pd.Timestamp("2017-09-18"), pd.Timestamp("2021-03-22"))]
    _, before_diag = de_membership.build_de_membership(changes, current, keys)
    assert diag["double_add"] == before_diag["double_add"] - 1
    assert diag["double_remove"] == before_diag["double_remove"] - 1
    assert len(diag["event_identity"]) == 4
    assert set(diag["event_identity"]) == {
        ("mdax", pd.Timestamp("1999-09-20"), "removed", "GEA AG (1999)"),
        ("dax", pd.Timestamp("1996-07-22"), "added", "Ceconomy"),
        ("dax", pd.Timestamp("2012-09-24"), "removed", "Ceconomy"),
        ("mdax", pd.Timestamp("2012-09-24"), "added", "Ceconomy"),
    }


def test_mdax_year_end_counts():
    _, _, _, membership, _ = _load()
    for year in range(2014, 2026):
        active = de_membership.membership_on(membership, f"{year}-12-31")
        assert active["index_code"].eq("mdax").sum() == (60 if 2018 <= year <= 2020 else 50)


def test_event_identities_resource_valid():
    identities = de_membership.load_event_identities()
    assert list(identities.columns) == ["index_code", "date", "side", "text", "company_key", "evidence"]
    assert len(identities) == 4
    assert identities["evidence"].notna().all()
    assert identities["evidence"].str.strip().ne("").all()
    assert identities["side"].isin(["added", "removed"]).all()
    assert pd.to_datetime(identities["date"]).notna().all()
    assert all(isinstance(date, pd.Timestamp) for date in identities["date"])
    assert identities["index_code"].isin(de_membership.INDEXES).all()


def test_linde_continuous_in_dax():
    _, _, keys, membership, diag = _load()
    key = _key(keys, "Linde")
    rows = membership.loc[membership["company_key"].eq(key) & membership["index_code"].eq("dax")]
    assert list(rows[["start_date", "end_date"]].itertuples(index=False, name=None)) == [
        (de_membership.FOREVER, pd.Timestamp("2023-02-27")),
    ]
    assert ("dax", key, pd.Timestamp("2017-10-25")) in diag["same_day_swap"]


def test_dax_year_end_counts():
    _, _, _, membership, _ = _load()
    for year in range(2014, 2026):
        active = de_membership.membership_on(membership, f"{year}-12-31")
        assert active["index_code"].eq("dax").sum() == (30 if year < 2021 else 40)


def test_membership_spot_checks():
    _, _, keys, membership, _ = _load()

    def intervals(name: str, index: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        rows = membership.loc[membership["company_key"].eq(_key(keys, name)) & membership["index_code"].eq(index)]
        return list(rows[["start_date", "end_date"]].itertuples(index=False, name=None))

    assert intervals("Wirecard", "dax") == [(pd.Timestamp("2018-09-24"), pd.Timestamp("2020-08-24"))]
    assert intervals("Delivery Hero", "dax") == [(pd.Timestamp("2020-08-24"), pd.Timestamp("2022-06-20"))]
    airbus = intervals("Airbus", "dax")
    assert len(airbus) == 1 and airbus[0][0] == pd.Timestamp("2021-09-20") and pd.isna(airbus[0][1])
    truck = intervals("Daimler Truck", "dax")
    assert len(truck) == 2
    assert truck[0] == (pd.Timestamp("2021-12-10"), pd.Timestamp("2021-12-13"))
    assert truck[1][0] == pd.Timestamp("2022-03-21") and pd.isna(truck[1][1])
    energy = intervals("Siemens Energy", "dax")
    assert (pd.Timestamp("2020-09-28"), pd.Timestamp("2020-09-29")) in energy
    assert (pd.Timestamp("2021-03-22"), pd.Timestamp("2022-03-21")) in energy
    assert any(start < pd.Timestamp("2014-01-01") and end == pd.Timestamp("2020-06-22")
               for start, end in intervals("Lufthansa", "dax"))
    assert any(start == pd.Timestamp("2020-06-22") and pd.isna(end)
               for start, end in intervals("Lufthansa", "mdax"))
    for name, old_index, new_index in (("Hugo Boss", "mdax", "sdax"), ("Ströer", "sdax", "mdax")):
        assert any(end == pd.Timestamp("2026-09-21") for _, end in intervals(name, old_index))
        assert any(start == pd.Timestamp("2026-09-21") and pd.isna(end) for start, end in intervals(name, new_index))


def test_membership_invariants():
    _, _, _, membership, diag = _load()
    print(diag)
    assert list(membership.columns) == de_membership.MEMBERSHIP_COLS
    assert not (membership["start_date"] > membership["end_date"]).any()
    assert not (membership["start_date"] == membership["end_date"]).any()
    for _, rows in membership.groupby(["company_key", "index_code"]):
        rows = rows.sort_values("start_date")
        for previous_end, next_start in zip(rows["end_date"].iloc[:-1], rows["start_date"].iloc[1:]):
            assert pd.notna(previous_end) and previous_end <= next_start
    active = de_membership.membership_on(membership, "2026-11-20")
    assert list(active.columns) == ["company_key", "index_code", "cap_bucket"]
    assert active.groupby("index_code").size().to_dict() == {"dax": 40, "mdax": 50, "sdax": 70}
    assert active.groupby("index_code")["cap_bucket"].unique().map(list).to_dict() == {
        "dax": ["large"], "mdax": ["mid"], "sdax": ["small"],
    }


def test_update_refuses_us_market(de_env, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])

    def unexpected_download(*args, **kwargs):
        pytest.fail("German membership attempted a download for the US market")

    monkeypatch.setattr(de_membership, "fetch_pages", unexpected_download)
    with pytest.raises(RuntimeError, match="DRIFTREV_MARKET=de"):
        de_membership.update_de_membership()
    assert not config.WAREHOUSE.exists()
    assert not de_env.exists()


def test_update_writes_table(de_env, monkeypatch):
    monkeypatch.setattr(de_membership, "fetch_pages", lambda refresh=False: _pages())
    redirects = json.loads((FIXTURES / "redirects.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(de_membership, "fetch_redirects", lambda titles: redirects)
    report = de_membership.update_de_membership()
    stored = store.read_table("membership_history")
    assert config.WAREHOUSE == de_env / "warehouse.duckdb"
    assert report["table"] == "membership_history"
    assert list(stored.columns) == de_membership.MEMBERSHIP_COLS
    assert len(stored) == report["rows"]
    _, _, _, expected, diag = _load()
    assert report == {"table": "membership_history", **{
        key: len(value) if isinstance(value, list) else value
        for key, value in diag.items() if isinstance(value, (int, list))}}
    pd.testing.assert_frame_equal(stored, expected, check_dtype=False)


def test_fetch_redirects_batches(de_env, monkeypatch):
    urls = []
    batches = []
    settings = config.SETTINGS
    titles = [f"Unternehmen {i:03d} & Co." for i in range(120)]
    monkeypatch.setattr(store, "RateLimiter", lambda rps: SimpleNamespace(wait=lambda: None))

    def fake_get(sess, url, limiter):
        limiter.wait()
        urls.append(url)
        batch = parse_qs(urlsplit(url).query)["titles"][0].split("|")
        batches.append(batch)
        title = batch[0]
        return SimpleNamespace(status_code=200, json=lambda: {"query": {
            "redirects": [{"from": title, "to": title + "!"}]}})

    monkeypatch.setattr(store, "get", fake_get)
    result = de_membership.fetch_redirects(iter(list(reversed(titles)) + titles[:10]))
    assert len(urls) == 3
    assert all(url.startswith(config.DE_WIKI_API) and "redirects=1" in url for url in urls)
    assert [len(batch) for batch in batches] == [50, 50, 20]
    assert [title for batch in batches for title in batch] == sorted(titles)
    assert result == {"normalized": {}, "redirects": {batch[0]: batch[0] + "!" for batch in batches}}
    assert config.SETTINGS is settings
