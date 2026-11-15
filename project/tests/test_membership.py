"""R1 historical membership + HW1 Q3 correction state: pure-function tests (no network, no warehouse)."""

import numpy as np
import pandas as pd

from driftrev import features
from driftrev import membership as m

HTML = """<table id="changes"><tr><th>Date</th><th>Added</th><th></th><th>Removed</th><th></th><th>Reason</th><th>Refs</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>March 5, 2020</td><td>NEW</td><td>Newco Inc.</td><td>OLD</td><td>Oldco Corp</td><td>cap</td><td>[1]</td></tr>
<tr><td>January 2, 2018</td><td>BRK.B</td><td>Berkshire</td><td></td><td></td><td>x</td><td>[2]</td></tr></table>"""


def _changes():
    rows = [
        # (date, added, removed, index)
        ("2020-03-05", "NEW", "OLD", "sp400"),      # NEW joins, OLD leaves
        ("2018-01-02", "OLD", None, "sp400"),       # OLD had joined in 2018
        ("2019-06-01", None, "GONE", "sp400"),      # GONE was a member before the table starts
        ("2021-01-04", "BACK", None, "sp400"),
        ("2022-01-03", None, "BACK", "sp400"),
        ("2023-01-03", "BACK", None, "sp400"),      # left and re-joined
        ("2024-01-02", "MIG", None, "sp500"),       # migrates sp400 -> sp500
        ("2024-01-02", None, "MIG", "sp400"),
    ]
    return pd.DataFrame(rows, columns=["date", "added_ticker", "removed_ticker", "index_name"]).assign(
        date=lambda d: pd.to_datetime(d["date"]), added_name=None, removed_name=None, reason=None)


def _current():
    return pd.DataFrame({"ticker": ["NEW", "BACK", "MIG", "STABLE"], "index_name": ["sp400", "sp400", "sp500", "sp400"]})


def test_parse_changes_html():
    df = m.parse_changes(HTML, "sp500")
    assert len(df) == 2
    assert df.loc[0, "added_ticker"] == "NEW" and df.loc[0, "removed_ticker"] == "OLD"
    assert df.loc[1, "added_ticker"] == "BRK-B" and df.loc[1, "removed_ticker"] is None   # yahoo symbol
    assert df.loc[0, "date"] == pd.Timestamp("2020-03-05")


def test_reconstruction_intervals():
    mem, diag = m.build_membership(_changes(), _current())
    iv = {(r.ticker, r.index_name): [] for r in mem.itertuples()}
    for r in mem.itertuples():
        iv[(r.ticker, r.index_name)].append((r.start_date, r.end_date))
    assert iv[("OLD", "sp400")] == [(pd.Timestamp("2018-01-02"), pd.Timestamp("2020-03-05"))]
    assert iv[("NEW", "sp400")] == [(pd.Timestamp("2020-03-05"), pd.NaT)]
    assert iv[("GONE", "sp400")][0][0] == m.FOREVER and iv[("GONE", "sp400")][0][1] == pd.Timestamp("2019-06-01")
    assert len(iv[("BACK", "sp400")]) == 2                              # two separate stints
    assert iv[("STABLE", "sp400")] == [(m.FOREVER, pd.NaT)]             # current member without any row
    assert diag["coverage_start"]["sp400"] == pd.Timestamp("2018-01-02")
    assert diag["orphans"] == [] and diag["inconsistent_current"] == []


def test_orphan_and_inconsistent_current():
    ch = pd.DataFrame([("2019-01-01", "FB", None, "sp500"), ("2020-01-01", None, "ZED", "sp500")],
                      columns=["date", "added_ticker", "removed_ticker", "index_name"]).assign(date=lambda d: pd.to_datetime(d["date"]))
    mem, diag = m.build_membership(ch, pd.DataFrame({"ticker": ["ZED"], "index_name": ["sp500"]}))
    assert diag["orphans"] == ["FB"]                    # added, never removed, not current -> unresolved rename
    assert diag["inconsistent_current"] == ["ZED"]      # current but last row is a removal
    assert "FB" not in set(mem["ticker"])


def test_point_in_time_in_index_flag():
    mem, _ = m.build_membership(_changes(), _current())
    iv = m.merged_intervals(mem)
    dates = pd.to_datetime(["2017-12-29", "2018-01-02", "2020-03-04", "2020-03-05", "2021-06-01"])
    # OLD: member from 2018-01-02 up to (excluding) the effective removal date 2020-03-05
    assert m.in_index_mask(dates, iv["OLD"]).tolist() == [False, True, True, False, False]
    # NEW: member from its addition date
    assert m.in_index_mask(dates, iv["NEW"]).tolist() == [False, False, False, True, True]
    # MIG leaves sp400 and joins sp500 on the same day: union of indices has no gap
    d2 = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])
    assert m.in_index_mask(d2, iv["MIG"]).tolist() == [True, True, True]
    # no information about the ticker (None) -> treated as member (old behaviour)
    assert m.in_index_mask(dates, None).all()


def test_gap_merge_and_reentry():
    mem, _ = m.build_membership(_changes(), _current())
    back = m.merged_intervals(mem, gap_days=10)["BACK"]
    assert len(back) == 2                                # 2022-01-03 .. 2023-01-03 gap is a year: not merged
    dates = pd.to_datetime(["2021-06-01", "2022-06-01", "2023-06-01"])
    assert m.in_index_mask(dates, back).tolist() == [True, False, True]


def test_name_match_and_sic_sector():
    assert m.name_match("Oldco Corp", "OLDCO INC")
    assert not m.name_match("Oldco Corp", "Totally Different Holdings")      # ticker reuse guard
    assert m.sic_to_sector("2834") == "Health Care" and m.sic_to_sector("6022") == "Financials"
    assert m.sic_to_sector("1311") == "Energy" and m.sic_to_sector("bogus") is None


def test_correction_state_math():
    spy = pd.Series([100, 103, 110, 108, 104.4, 100, 99, 105, 110.5, 112, 100.0],
                    index=pd.bdate_range("2024-01-01", periods=11))
    c = features.correction_state(spy, threshold=0.05)
    # ATH 110 -> 104.4 is -5.09%: first in-correction day (index 4); 108 is only -1.8%
    assert c["in_correction"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
    assert c["days_in_correction"].tolist() == [0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 1]
    assert np.isclose(c["correction_depth"].iloc[6], 99 / 110 - 1)
    assert (c["correction_depth"] <= 0).all()
    # point-in-time: appending future data cannot change past values
    c2 = features.correction_state(pd.concat([spy, pd.Series([200.0], index=[spy.index[-1] + pd.Timedelta(days=1)])]), 0.05)
    assert c2.iloc[:len(c)].equals(c)


def test_universe_needs_refresh_rules():
    from driftrev import universe
    now = pd.Timestamp("2026-01-10")
    fresh = {"max_tickers": 0, "built_at": pd.Timestamp("2026-01-08")}
    assert not universe.needs_refresh(fresh, 0, now)
    assert universe.needs_refresh({"max_tickers": 60, "built_at": pd.Timestamp("2026-01-09")}, 0, now)    # truncated, now full
    assert universe.needs_refresh({"max_tickers": 60, "built_at": pd.Timestamp("2026-01-09")}, 100, now)  # asks for more
    assert not universe.needs_refresh({"max_tickers": 60, "built_at": pd.Timestamp("2026-01-09")}, 50, now)
    assert universe.needs_refresh({"max_tickers": 0, "built_at": pd.Timestamp("2025-12-01")}, 0, now)     # older than a week
    assert not universe.needs_refresh(None, 0, now)                                                       # legacy table


def test_survivorship_table_partitions_universes():
    import pandas as pd

    from driftrev.membership import survivorship_table
    ev = pd.DataFrame({
        "event_family": ["shock_crash"] * 4,
        "in_index": [True, True, False, True],
        "is_current_member": [True, False, True, True],
        "growth_future_10d": [1.05, 0.90, 1.10, 1.00],
        "growth_future_10d_vs_spy": [1.04, 0.91, 1.08, 1.0],
    })
    t = survivorship_table(ev).set_index(["universe", "event_family"])
    assert t.loc[("current_members_any_date (old behaviour)", "all"), "events"] == 3
    assert t.loc[("point_in_time_all_names", "all"), "events"] == 3
    assert t.loc[("point_in_time_removed_names", "all"), "events"] == 1
    assert t.loc[("current_names_before_joining (excluded now)", "all"), "events"] == 1
    assert abs(t.loc[("point_in_time_removed_names", "all"), "mean_10d"] + 0.10) < 1e-9
