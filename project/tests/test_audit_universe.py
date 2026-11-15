"""Audit F: point-in-time universe attributes (cap bucket), benchmark universe, survivorship guards."""

import pandas as pd
from test_audit_leakage import _world  # synthetic world

from driftrev import membership as mbr


def _mem():
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"], "index_name": ["sp600", "sp500", "sp400"],
        "start_date": pd.to_datetime(["1900-01-01", "2021-03-01", "1900-01-01"]),
        "end_date": pd.to_datetime(["2021-03-01", None, None])})


def test_cap_bucket_is_point_in_time():
    iv = mbr.bucket_intervals(_mem())
    d = pd.to_datetime(["2019-06-03", "2021-02-26", "2021-03-01", "2024-01-02"])
    assert list(mbr.cap_bucket_at(d, iv["AAA"])) == ["small", "small", "large", "large"]   # not 'large' before 2021-03
    assert list(mbr.cap_bucket_at(d, iv["BBB"])) == ["mid"] * 4
    assert list(mbr.cap_bucket_at(d[:1], [])) == [None]


def test_cap_bucket_overlap_prefers_larger():
    m = pd.DataFrame({"ticker": ["Z", "Z"], "index_name": ["sp400", "sp500"],
                      "start_date": pd.to_datetime(["1900-01-01", "2020-01-02"]),
                      "end_date": pd.to_datetime(["2020-01-02", None])})
    iv = mbr.bucket_intervals(m)["Z"]
    assert list(mbr.cap_bucket_at(pd.to_datetime(["2020-01-01", "2020-01-02"]), iv)) == ["mid", "large"]


def test_build_all_uses_point_in_time_bucket():
    from driftrev import features
    uni, stocks, market, macro, filings = _world(100)
    uni["cap_bucket"] = "large"                       # today's bucket, as in the universe table
    t0 = uni["ticker"].iloc[0]
    mem = pd.DataFrame({"ticker": uni["ticker"], "index_name": "sp500",
                        "start_date": pd.Timestamp("1900-01-01"), "end_date": pd.NaT})
    mem.loc[mem["ticker"] == t0, "index_name"] = "sp600"        # small cap for the whole sample in reality
    sue = pd.DataFrame(columns=["cik", "period_end", "eps", "filed", "diff", "sue", "eps_yoy_pos"])
    ev, _, _ = features.build_all(uni, stocks, market, macro, filings, sue, pd.DataFrame(), membership=mem)
    assert set(ev.loc[ev["ticker"] == t0, "cap_bucket"]) == {"small"}
    assert set(ev.loc[ev["ticker"] != t0, "cap_bucket"]) == {"large"}
