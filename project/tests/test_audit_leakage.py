"""Audit F: end-to-end look-ahead check of the whole feature build (features.build_all).

Everything after a cut date T is replaced by different random data. Every feature of every event
dated <= T must be bit-identical (only targets / next_date may look forward)."""

import numpy as np
import pandas as pd

from driftrev import config, dataset, features

N_DAYS, N_TICK = 520, 14
DATES = pd.bdate_range("2019-01-01", periods=N_DAYS)
T = DATES[400]


def _world(seed_future: int):
    """Same history up to T for every seed_future; a different random future after it."""
    rng_h, rng_f = np.random.default_rng(1), np.random.default_rng(seed_future)
    n_hist = 401
    syms = ["SPY", "IWM", "^VIX", "^VIX3M", "XLK"] + [f"S{i:02d}" for i in range(N_TICK)]
    rows = []
    for k, s in enumerate(syms):
        r = np.concatenate([rng_h.normal(0.0004, 0.02, n_hist), rng_f.normal(0.0004, 0.02, N_DAYS - n_hist)])
        if not s.startswith("^"):          # a few crash days so crash events exist in history and future
            r[[120 + k, 250 + k, 450 + k]] -= 0.14
        base = 20.0 if s.startswith("^") else 60.0 + 3 * k
        c = base * np.exp(np.cumsum(r))
        vol = rng_h.integers(2_000_000, 4_000_000, N_DAYS).astype(float)
        vol[[120 + k, 250 + k, 450 + k]] *= 5
        rows.append(pd.DataFrame({"ticker": s, "date": DATES, "open": c * 0.998, "high": c * 1.012, "low": c * 0.988,
                                  "close": c, "volume": vol, "dividends": 0.0, "splits": 0.0}))
    px = pd.concat(rows, ignore_index=True)
    # make history identical regardless of the future seed (rng_h consumed identically above)
    market = px[px["ticker"].isin(["SPY", "IWM", "^VIX", "^VIX3M", "XLK"])]
    stocks = px[px["ticker"].str.startswith("S") & (px["ticker"] != "SPY")]
    uni = pd.DataFrame({"ticker": sorted(stocks["ticker"].unique())})
    uni["company"], uni["sector"], uni["sub_industry"] = uni["ticker"], "Information Technology", "x"
    uni["wikipedia_title"], uni["cap_bucket"], uni["sector_etf"] = None, "mid", "XLK"
    uni["cik"] = np.arange(1, len(uni) + 1)
    # quarterly earnings 8-Ks after the close
    fl = []
    for cik in uni["cik"]:
        for d in DATES[60::63]:
            fl.append({"cik": cik, "form": "8-K", "items": "2.02,9.01", "accession": f"{cik}-{d.date()}",
                       "accepted_et": d + pd.Timedelta(hours=16, minutes=5)})
    filings = pd.DataFrame(fl)
    macro = pd.DataFrame({"dgs10": np.linspace(2, 3, N_DAYS)}, index=DATES)
    return uni, stocks, market, macro, filings


def _build(seed_future: int):
    uni, stocks, market, macro, filings = _world(seed_future)
    sue = pd.DataFrame(columns=["cik", "period_end", "eps", "filed", "diff", "sue", "eps_yoy_pos"])
    ev, panel, ctx = features.build_all(uni, stocks, market, macro, filings, sue, pd.DataFrame(), restrict_events=False)
    return ev, panel, ctx


def test_build_all_has_no_look_ahead():
    ev_a, _, ctx_a = _build(100)
    ev_b, _, ctx_b = _build(200)
    a, b = ev_a[ev_a["date"] <= T].reset_index(drop=True), ev_b[ev_b["date"] <= T].reset_index(drop=True)
    assert len(a) > 30, "synthetic world should generate events"
    assert a[["ticker", "date"]].equals(b[["ticker", "date"]]), "event set itself changed with the future"
    forward = [c for c in a.columns if "future" in c or c.startswith(("mae_", "mfe_"))]
    cols = [c for c in a.columns if c not in forward]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9)
    # market context too (spy_future_* excluded)
    ca, cb = ctx_a.loc[:T], ctx_b.loc[:T]
    ccols = [c for c in ca.columns if "future" not in c]
    pd.testing.assert_frame_equal(ca[ccols], cb[ccols], check_exact=False, rtol=1e-9)
    # the history-only target of earlier events must be realised: prev_pead_20d only uses closed windows
    assert "prev_pead_20d" in a.columns


def test_prepare_features_exclude_targets():
    ev, _, _ = _build(100)
    for c in ("att_ratio_7_90", "att_spike_1d", "att_level"):    # no pageviews in the synthetic world
        ev[c] = np.nan
    ml, feats = dataset.prepare(ev)
    bad = [f for f in feats if "future" in f or f.startswith(("mae_", "mfe_", "is_positive"))]
    assert not bad
    for banned in ("in_index", "is_current_member"):
        assert banned not in feats and not any(f.startswith(banned) for f in feats)
    assert config.SETTINGS.research.feature_set in dataset.FEATURE_SETS


def test_cross_sectional_features_only_see_their_own_date():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"date": np.repeat(pd.bdate_range("2024-01-01", periods=10), 6),
                      "ret_1d": rng.normal(size=60), "abn_ret_1d": rng.normal(size=60)})
    a = dataset.add_cross_sectional(d.copy(), cols=["ret_1d", "abn_ret_1d"])
    d2 = d.copy()
    later = d2["date"] >= pd.Timestamp("2024-01-08")
    d2.loc[later, ["ret_1d", "abn_ret_1d"]] = rng.normal(size=(later.sum(), 2)) * 50      # rewrite the future
    b = dataset.add_cross_sectional(d2, cols=["ret_1d", "abn_ret_1d"])
    pd.testing.assert_frame_equal(a[~later.to_numpy()].reset_index(drop=True), b[~later.to_numpy()].reset_index(drop=True))
