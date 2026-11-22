"""Offline German earnings sessions, local close cutoffs and shared pipeline integration."""

import numpy as np
import pandas as pd
from test_de_pipeline import _ohlcv, _world

from driftrev import (
    config,
    consensus,
    de_context,
    de_market,
    de_membership,
    de_pipeline,
    de_universe,
    features,
    macro,
    prices,
    store,
)


def _cons(tickers, releases):
    ts = pd.to_datetime(releases, format="mixed")
    return pd.DataFrame({"ticker": tickers, "release_et": ts, "release_date": ts.normalize(),
                         "eps_estimate": 1.0, "reported_eps": 1.1, "surprise_pct": 10.0})


def _earnings_world():
    universe, membership, stocks, market, macro_wide = _world()
    dates = macro_wide.index
    # A flat benchmark on the release day makes the stock's +6% move exactly +6% abnormal.
    for frame, ticker, move in ((market, "^GDAXI", 0.0), (stocks, "AAA.DE", 0.06)):
        rows = frame["ticker"].eq(ticker)
        close = frame.loc[rows].set_index("date")["close"]
        factor = close.loc[dates[299]] * (1 + move) / close.loc[dates[300]]
        frame.loc[rows & frame["date"].ge(dates[300]), "close"] *= factor
        frame.loc[rows, "open"] = frame.loc[rows, "close"].shift(1).bfill()
        frame.loc[rows, "high"] = frame.loc[rows, ["open", "close"]].max(axis=1) * 1.01
        frame.loc[rows, "low"] = frame.loc[rows, ["open", "close"]].min(axis=1) * 0.99
    return universe, membership, stocks, market, macro_wide


def test_release_session_berlin_rules():
    sessions = pd.bdate_range("2026-11-01", "2026-11-30")
    cons = _cons(["AAA.DE", "BBB.DE", "CCC.DE", "DDD.DE", "CLOSE.DE", "AAA.DE", "END.DE", "MISSING.DE"],
                 ["2026-11-11 23:00", "2026-11-12 10:00", "2026-11-12 11:45", "2026-11-13 20:00",
                  "2026-11-12 11:30", "2026-11-12 09:00", "2026-11-30 11:30", None]).iloc[::-1]
    original = cons.copy(deep=True)
    out = consensus.release_session_dates(cons, sessions[::-1], config.MARKETS["de"])
    assert list(out.columns) == ["ticker", "date", "release_et", "release_local"]
    actual = out.set_index("ticker")
    expected = {"AAA.DE": "2026-11-12", "BBB.DE": "2026-11-12", "CCC.DE": "2026-11-13",
                "DDD.DE": "2026-11-16", "CLOSE.DE": "2026-11-13"}
    assert actual["date"].to_dict() == {ticker: pd.Timestamp(date) for ticker, date in expected.items()}
    local = {"AAA.DE": "2026-11-12 05:00", "BBB.DE": "2026-11-12 16:00", "CCC.DE": "2026-11-12 17:45",
             "DDD.DE": "2026-11-14 02:00", "CLOSE.DE": "2026-11-12 17:30"}
    assert actual["release_local"].to_dict() == {
        ticker: pd.Timestamp(ts, tz="Europe/Berlin") for ticker, ts in local.items()}
    assert actual.loc["AAA.DE", "release_et"] == pd.Timestamp("2026-11-11 23:00")
    assert actual["release_et"].dt.tz is None
    pd.testing.assert_frame_equal(cons, original)
    holiday_sessions = sessions[sessions != pd.Timestamp("2026-11-12")]
    holiday = consensus.release_session_dates(cons, holiday_sessions, config.MARKETS["de"]).set_index("ticker")
    assert holiday.loc[["AAA.DE", "BBB.DE"], "date"].eq(pd.Timestamp("2026-11-13")).all()
    for data, dates in ((pd.DataFrame(), sessions), (cons, sessions[:0])):
        empty = consensus.release_session_dates(data, dates, config.MARKETS["de"])
        assert empty.empty and list(empty.columns) == list(out.columns)


def test_release_session_dst_gap():
    sessions = pd.bdate_range("2026-03-16", "2026-03-31")
    cons = _cons(["AAA.DE", "BBB.DE", "CCC.DE"],
                 ["2026-03-20 12:00", "2026-03-20 12:45", "2026-03-30 12:00"])
    out = consensus.release_session_dates(cons, sessions, config.MARKETS["de"]).set_index("ticker")
    assert out.loc["AAA.DE", "release_local"] == pd.Timestamp("2026-03-20 17:00", tz="Europe/Berlin")
    assert out.loc["BBB.DE", "release_local"] == pd.Timestamp("2026-03-20 17:45", tz="Europe/Berlin")
    assert out.loc["AAA.DE", "date"] == pd.Timestamp("2026-03-20")
    assert out.loc["BBB.DE", "date"] == pd.Timestamp("2026-03-23")
    assert out.loc["CCC.DE", "release_local"] == pd.Timestamp("2026-03-30 18:00", tz="Europe/Berlin")
    assert out.loc["CCC.DE", "date"] == pd.Timestamp("2026-03-31")


def test_us_rules_unchanged():
    sessions = pd.bdate_range("2026-03-02", "2026-03-31").difference(pd.DatetimeIndex(["2026-03-09"]))
    releases = ["2026-03-06 08:00", "2026-03-06 12:00", "2026-03-06 15:59:59", "2026-03-06 16:00",
                "2026-03-06 23:00", "2026-03-07 10:00", "2026-03-09 09:00", "2026-03-20 16:00",
                "2026-03-31 16:00"]
    cons = _cons(["AAA"] * len(releases), releases).iloc[::-1]
    filings = pd.DataFrame({"cik": 1, "form": "8-K", "items": "2.02", "accepted_et": cons["release_et"]})
    expected = features.earnings_days(filings, 1, sessions.values).reset_index(drop=True)
    out = consensus.release_session_dates(cons, sessions, config.MARKETS["us"])
    actual = out[["date", "release_et"]].rename(columns={"release_et": "earn_accepted_et"})
    pd.testing.assert_frame_equal(actual, expected)
    assert str(out["release_local"].dt.tz) == "America/New_York"
    pd.testing.assert_series_equal(out["release_local"].dt.tz_localize(None), out["release_et"], check_names=False)


def test_consensus_cutoff_per_market(monkeypatch):
    sessions = pd.bdate_range("2026-03-01", "2026-11-30")
    for day, close_et in (("2026-11-12", "11:30"), ("2026-03-20", "12:30")):
        releases = [f"{day} {time}" for time in ("11:00", "12:00", close_et, "16:00", "16:00:01")]
        cons = _cons(["EARLY", "NOON", "DE_CLOSE", "US_CLOSE", "LATE"], releases)
        events = pd.DataFrame({"ticker": cons["ticker"], "date": pd.Timestamp(day)}, index=cons.index)
        events.index = [7, 3, 9, 1, 12]
        us = consensus.consensus_features(events, cons, sessions, market=config.MARKETS["us"])
        de = consensus.consensus_features(events, cons, sessions, market=config.MARKETS["de"])
        assert us["eps_est_available"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
        assert de["eps_est_available"].tolist() == [1.0, float(day == "2026-03-20"), 1.0, 0.0, 0.0]
        for result in (us, de):
            pd.testing.assert_index_equal(result.index, events.index)
            known = result["eps_est_available"].eq(1)
            np.testing.assert_allclose(result.loc[known, "eps_surprise_pct"], 10.0)
            assert result.loc[known, "eps_beat"].eq(1.0).all()
            assert result.loc[known, "eps_release_age_sessions"].eq(0.0).all()
            assert result.loc[~known, ["eps_surprise_pct", "eps_beat", "eps_release_age_sessions"]].isna().all().all()
        monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
        pd.testing.assert_frame_equal(consensus.consensus_features(events, cons, sessions), de)
        pd.testing.assert_frame_equal(consensus.consensus_features(events, cons, sessions, market=None), de)


def test_build_all_with_earnings_mapping(monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    universe, membership, stocks, market, macro_wide = _earnings_world()
    dates = macro_wide.index
    mapping = {"AAA.DE": pd.DataFrame({"date": [dates[300]],
                                       "earn_accepted_et": [dates[300] + pd.Timedelta(hours=8)]})}
    original = mapping["AAA.DE"].copy(deep=True)
    args = (de_pipeline.universe_for_features(universe), stocks, market, macro_wide,
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    kwargs = {"membership": de_pipeline.membership_for_features(membership, universe),
              "ctx": de_context.market_context_de(market, macro_wide),
              "cap_buckets": dict(config.MARKETS["de"].cap_bucket_by_index),
              "liquid_by_membership": ["CCC.HM"], "min_dollar_volume": config.MARKETS["de"].min_dollar_volume,
              "breakout_min_median_turnover": config.MARKETS["de"].breakout_min_median_turnover}
    baseline, _, _ = features.build_all(*args, **kwargs)
    events, _, _ = features.build_all(*args, **kwargs, earnings_by_ticker=mapping)
    release = events.loc[events["ticker"].eq("AAA.DE") & events["date"].eq(dates[300])]
    assert len(release) == 1
    assert release.iloc[0]["event_family"] == "earnings_up"
    assert np.isclose(release.iloc[0]["abn_ret_1d"], 0.06)
    assert release.iloc[0]["earn_accepted_et"] == mapping["AAA.DE"].iloc[0]["earn_accepted_et"]
    assert not baseline["is_earnings"].any()
    assert not baseline.loc[baseline["ticker"].eq("AAA.DE") & baseline["date"].eq(dates[300]),
                            "event_family"].eq("earnings_up").any()
    assert events.loc[events["ticker"].ne("AAA.DE"), "is_earnings"].eq(False).all()
    pd.testing.assert_frame_equal(mapping["AAA.DE"], original)
    negative_stocks = stocks.copy(deep=True)
    rows = negative_stocks["ticker"].eq("AAA.DE")
    negative_stocks.loc[rows & negative_stocks["date"].ge(dates[300]), "close"] *= 0.94 / 1.06
    negative_stocks.loc[rows, "open"] = negative_stocks.loc[rows, "close"].shift(1).bfill()
    negative_stocks.loc[rows, "high"] = negative_stocks.loc[rows, ["open", "close"]].max(axis=1) * 1.01
    negative_stocks.loc[rows, "low"] = negative_stocks.loc[rows, ["open", "close"]].min(axis=1) * 0.99
    negative_args = (args[0], negative_stocks, *args[2:])
    negative_events, _, _ = features.build_all(*negative_args, **kwargs, earnings_by_ticker=mapping)
    down = negative_events.loc[negative_events["ticker"].eq("AAA.DE") & negative_events["date"].eq(dates[300])]
    assert down["event_family"].tolist() == ["earnings_down"]
    assert np.isclose(down.iloc[0]["abn_ret_1d"], -0.06)
    crash_mapping = {"AAA.DE": pd.DataFrame({"date": [dates[250]],
                                             "earn_accepted_et": [dates[250] + pd.Timedelta(hours=8)]})}
    crash_events, _, _ = features.build_all(*args, **kwargs, earnings_by_ticker=crash_mapping)
    assert crash_events.loc[crash_events["ticker"].eq("AAA.DE") & crash_events["date"].eq(dates[250]),
                            "event_family"].tolist() == ["earnings_crash"]

    # Reuse the US-style synthetic inputs and compare all three outputs with explicit None.
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])
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
        explicit = features.build_all(*args, membership=membership, earnings_by_ticker=None)
        for actual, expected in zip(explicit, default):
            pd.testing.assert_frame_equal(actual, expected)
        assert not default[0].empty

    def unexpected(*args, **kwargs):
        raise AssertionError("an explicit earnings mapping must bypass EDGAR dating")

    monkeypatch.setattr(features, "earnings_days", unexpected)
    us_mapping = {"AAA": pd.DataFrame({"date": [dates[200]],
                                       "earn_accepted_et": [dates[200] + pd.Timedelta(hours=8)]})}
    args = (uni, stocks, market, pd.DataFrame(), filings, pd.DataFrame(), pd.DataFrame())
    overridden, _, _ = features.build_all(*args, membership=membership, earnings_by_ticker=us_mapping)
    assert set(overridden.loc[overridden["is_earnings"], "ticker"]) == {"AAA"}
    empty_mapping, _, _ = features.build_all(*args, membership=membership, earnings_by_ticker={})
    assert not empty_mapping["is_earnings"].any()


def test_german_earnings_mapping(monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    universe, membership, stocks, market, macro_wide = _earnings_world()
    dates = macro_wide.index
    # The market calendar, rather than an individual stock's dates, controls release dating.
    market = market.loc[market["date"].ne(dates[301])]
    sessions = dates.delete(301)
    releases = [dates[300] + pd.Timedelta(hours=hour) for hour in (10, 8, 12)]
    cons = _cons(["AAA.DE", "AAA.DE", "BBB.DE"], releases)
    original = cons.copy(deep=True)
    mapping = de_pipeline.german_earnings(cons, sessions)
    assert set(mapping) == {"AAA.DE", "BBB.DE"}
    for ticker, day, release in (("AAA.DE", dates[300], releases[1]), ("BBB.DE", dates[302], releases[2])):
        expected = pd.DataFrame({"date": [day], "earn_accepted_et": [release]})
        pd.testing.assert_frame_equal(mapping[ticker], expected)
        assert mapping[ticker]["earn_accepted_et"].dt.tz is None
    assert de_pipeline.german_earnings(pd.DataFrame(), sessions) == {}
    assert de_pipeline.german_earnings(cons, sessions[:0]) == {}
    pd.testing.assert_frame_equal(cons, original)

    def read_table(name):
        return {de_universe.TABLE: universe, de_membership.TABLE: membership}[name]

    loaded = []

    def load_consensus():
        loaded.append("consensus")
        return cons

    monkeypatch.setattr(store, "read_table", read_table)
    monkeypatch.setattr(prices, "load_prices", lambda: stocks)
    monkeypatch.setattr(prices, "load_market", lambda: market)
    monkeypatch.setattr(macro, "load_macro_wide", lambda: macro_wide)
    monkeypatch.setattr(consensus, "load_consensus", load_consensus)
    transformed = de_pipeline.transform_de(save=False)
    assert loaded == ["consensus"]
    assert dates[301] not in set(transformed["ctx"]["date"])
    earnings = transformed["events"].loc[transformed["events"]["is_earnings"]].set_index("ticker")
    assert earnings["date"].to_dict() == {"AAA.DE": dates[300], "BBB.DE": dates[302]}
    assert earnings.loc["AAA.DE", "event_family"] == "earnings_up"
    assert earnings["eps_est_available"].eq(1.0).all()
    np.testing.assert_allclose(earnings["eps_surprise_pct"], 10.0)

    calls = []

    def update_consensus(tickers):
        calls.append(tickers)
        return {"step": "consensus"}

    monkeypatch.setattr(de_market, "update_de_market", lambda: {"step": "market"})
    monkeypatch.setattr(de_market, "update_de_rates", lambda: {"step": "rates"})
    monkeypatch.setattr(de_membership, "update_de_membership", lambda refresh=False: {"step": "membership"})
    monkeypatch.setattr(de_universe, "update_de_universe", lambda: {"step": "universe"})
    monkeypatch.setattr(consensus, "update_consensus", update_consensus)
    report = de_pipeline.ingest_de()
    assert calls == [["AAA.DE", "BBB.DE", "DDD.DE"]]
    assert report == {name: {"step": name} for name in ("market", "rates", "membership", "universe", "consensus")}
