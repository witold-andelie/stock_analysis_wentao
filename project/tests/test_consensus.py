"""HW1 Q4 consensus surprise: point-in-time matching (no look-ahead)."""

import numpy as np
import pandas as pd

from driftrev import consensus as c

SESS = pd.bdate_range("2024-01-01", "2024-03-29")


def _cons(release_et, est=1.0, rep=1.1, sur=10.0, tkr="AAA"):
    return pd.DataFrame({"ticker": tkr, "release_et": pd.to_datetime([release_et]),
                         "release_date": pd.to_datetime([release_et]).normalize(),
                         "eps_estimate": est, "reported_eps": rep, "surprise_pct": sur})


def _ev(date, tkr="AAA"):
    return pd.DataFrame({"ticker": [tkr], "date": pd.to_datetime([date])}, index=[7])


def test_after_close_release_same_day_not_known():
    # release 2024-02-05 17:00 ET: a crash on 02-05 cannot know it; the 02-06 reaction session can
    cons = _cons("2024-02-05 17:00")
    assert c.consensus_features(_ev("2024-02-05"), cons, SESS)["eps_est_available"].iloc[0] == 0.0
    r = c.consensus_features(_ev("2024-02-06"), cons, SESS)
    assert r["eps_est_available"].iloc[0] == 1.0 and np.isclose(r["eps_surprise_pct"].iloc[0], 10.0)
    assert r["eps_beat"].iloc[0] == 1.0 and r["eps_release_age_sessions"].iloc[0] == 1


def test_never_uses_future_release():
    cons = _cons("2024-02-08 08:00")                    # released two sessions AFTER the event
    r = c.consensus_features(_ev("2024-02-06"), cons, SESS)
    assert r["eps_est_available"].iloc[0] == 0.0 and np.isnan(r["eps_surprise_pct"].iloc[0])


def test_window_three_sessions():
    cons = _cons("2024-02-01 08:00")                    # Thursday
    assert c.consensus_features(_ev("2024-02-06"), cons, SESS)["eps_est_available"].iloc[0] == 1.0   # 3 sessions later
    assert c.consensus_features(_ev("2024-02-07"), cons, SESS)["eps_est_available"].iloc[0] == 0.0   # 4 sessions: stale


def test_latest_release_wins_and_ticker_isolation():
    cons = pd.concat([_cons("2024-01-30 08:00", rep=1.2, sur=20.0), _cons("2024-02-05 08:00", rep=0.9, sur=-10.0),
                      _cons("2024-02-05 08:00", rep=2.0, sur=100.0, tkr="BBB")], ignore_index=True)
    r = c.consensus_features(_ev("2024-02-06"), cons, SESS)
    assert np.isclose(r["eps_surprise_pct"].iloc[0], -10.0) and r["eps_beat"].iloc[0] == 0.0


def test_missing_estimate_and_empty_table():
    cons = _cons("2024-02-05 08:00", est=np.nan, rep=1.0, sur=np.nan)
    r = c.consensus_features(_ev("2024-02-06"), cons, SESS)
    assert r["eps_est_available"].iloc[0] == 0.0 and np.isnan(r["eps_beat"].iloc[0])
    empty = c.consensus_features(_ev("2024-02-06"), pd.DataFrame(), SESS)
    assert empty["eps_est_available"].iloc[0] == 0.0 and list(empty.columns) == c.FEATURE_COLS


def test_normalize_yfinance_frame():
    idx = pd.DatetimeIndex(["2024-02-05 16:00", "2099-01-01 16:00"], tz="America/New_York", name="Earnings Date")
    raw = pd.DataFrame({"EPS Estimate": [1.0, 2.0], "Reported EPS": [1.1, np.nan], "Surprise(%)": [10.0, np.nan]}, index=idx)
    out = c.normalize_earnings_dates(raw, "AAA")
    assert len(out) == 1 and out["release_date"].iloc[0] == pd.Timestamp("2024-02-05")   # future row dropped
