"""HW1 Q4 · Analyst-consensus EPS surprise (Yahoo `Ticker.get_earnings_dates`).

Yahoo lists, per earnings release, the consensus EPS estimate, the reported EPS and the
surprise in percent. We keep them in the warehouse table `consensus_eps` (loaded incrementally,
partial coverage is fine) and attach them to events point-in-time:

    * only releases stamped at or before the event day's local market close are usable
      (a release after the close of day t is not knowable at t -> no look-ahead);
    * the release must be within `Consensus.window_sessions` (3) trading sessions before the event
      (a surprise older than that is not "the news" any more);
    * if several qualify, the latest one wins.

New event columns: eps_surprise_pct (clipped to +-100), eps_beat (1/0, NaN if unknown),
eps_est_available (1 if both estimate and reported EPS are known), eps_release_age_sessions.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config, store

log = logging.getLogger(__name__)
TABLE = "consensus_eps"
LOG_TABLE = "consensus_fetch_log"
EMPTY_RETRY_DAYS = 30.0          # tickers Yahoo has no table for are retried monthly, not every run
FEATURE_COLS = ["eps_surprise_pct", "eps_beat", "eps_est_available", "eps_release_age_sessions"]


def normalize_earnings_dates(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance frame (index 'Earnings Date', cols 'EPS Estimate','Reported EPS','Surprise(%)') -> table rows."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ticker", "release_et", "release_date", "eps_estimate", "reported_eps", "surprise_pct"])
    df = raw.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df.index = idx
    cols = {c: c.lower().replace(" ", "_").replace("(%)", "_pct").replace("surprise_pct", "surprise_pct") for c in df.columns}
    df = df.rename(columns=cols)
    out = pd.DataFrame({
        "ticker": ticker, "release_et": df.index,
        "eps_estimate": pd.to_numeric(df.get("eps_estimate"), errors="coerce").to_numpy(),
        "reported_eps": pd.to_numeric(df.get("reported_eps"), errors="coerce").to_numpy(),
        "surprise_pct": pd.to_numeric(df.get("surprise_pct", df.get("surprise(_pct)")), errors="coerce").to_numpy()})
    out["release_date"] = out["release_et"].dt.normalize()
    out = out[out["reported_eps"].notna() | out["surprise_pct"].notna()]          # drop future / unreported rows
    out = out[out["release_date"] >= pd.Timestamp("2013-01-01")]
    out = out.drop_duplicates(["ticker", "release_et"], keep="first")
    return out.reset_index(drop=True)


def _fetch_one(ticker: str, limit: int) -> tuple[str, pd.DataFrame | None, str]:
    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception as exc:   # throttling / parse errors / delisted
        msg = str(exc)
        return ticker, None, "ratelimit" if ("Too Many" in msg or "429" in msg or "RateLimit" in type(exc).__name__) else "error"
    return ticker, normalize_earnings_dates(raw, ticker), "ok"


def update_consensus(tickers: list[str], max_seconds: float | None = None) -> dict:
    """Incremental: skip tickers refreshed within `refresh_days` (or empty ones within 30 days)."""
    c = config.SETTINGS.consensus
    now = pd.Timestamp.now()
    log_df = store.read_table(LOG_TABLE)
    todo = list(tickers)
    if not log_df.empty:
        age = (now - pd.to_datetime(log_df["fetched_at"])).dt.total_seconds() / 86400
        limit_days = np.where(log_df["n_rows"] > 0, c.refresh_days, EMPTY_RETRY_DAYS)
        fresh = set(log_df.loc[age < limit_days, "ticker"])
        todo = [t for t in todo if t not in fresh]
    stats = {"requested": len(tickers), "skipped_fresh": len(tickers) - len(todo), "fetched": 0, "empty": 0,
             "errors": 0, "rate_limited": 0, "stopped_early": False}
    t0 = time.time()
    consecutive = 0
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        with ThreadPoolExecutor(max_workers=c.workers) as ex:
            res = list(ex.map(lambda t: _fetch_one(t, c.limit), chunk))
        rows, logs = [], []
        # Yahoo throttling often shows up as silent empty frames: a chunk that is mostly empty is not trusted
        n_empty = sum(1 for _, df, st in res if st == "ok" and (df is None or df.empty))
        if len(chunk) >= 10 and n_empty / len(chunk) > 0.6:
            stats["rate_limited"] += n_empty
            consecutive += n_empty
            res = [(t, d, st) for t, d, st in res if st == "ok" and d is not None and not d.empty]
        for tkr, df, status in res:
            if status != "ok":
                stats["rate_limited" if status == "ratelimit" else "errors"] += 1
                consecutive += 1
                continue
            consecutive = 0
            n = 0 if df is None else len(df)
            stats["fetched" if n else "empty"] += 1
            if n:
                rows.append(df)
            logs.append({"ticker": tkr, "fetched_at": now, "n_rows": n})
        if rows:
            allrows = pd.concat(rows, ignore_index=True)
            with store.connect() as con:      # replace each fetched ticker's rows wholesale
                if store.table_exists(TABLE):
                    con.execute(f"delete from {TABLE} where ticker in (select unnest(?))", [allrows["ticker"].unique().tolist()])
            store.upsert(TABLE, allrows, ["ticker", "release_et"])
        if logs:
            store.upsert(LOG_TABLE, pd.DataFrame(logs), ["ticker"])
        if consecutive >= c.max_failures or stats["rate_limited"] >= c.max_failures:
            stats["stopped_early"] = True
            log.warning("consensus: stopping early after %d failures (partial coverage is fine)", consecutive)
            break
        if max_seconds and time.time() - t0 > max_seconds:
            stats["stopped_early"] = True
            break
        time.sleep(0.5)
    return stats


def load_consensus() -> pd.DataFrame:
    return store.read_table(TABLE)


def release_session_dates(cons: pd.DataFrame, sessions, market: config.Market) -> pd.DataFrame:
    """Map ET-naive releases to the first session able to react in the market's timezone."""
    if cons.empty or len(sessions) == 0:
        return pd.DataFrame(columns=["ticker", "date", "release_et", "release_local"])
    sess = np.asarray(pd.DatetimeIndex(sessions).sort_values().values, dtype="datetime64[ns]")
    rows = cons[["ticker", "release_et"]].dropna(subset=["release_et"]).copy()
    rows["release_et"] = pd.to_datetime(rows["release_et"])
    rows["release_local"] = rows["release_et"].dt.tz_localize("America/New_York").dt.tz_convert(market.timezone)
    local = rows["release_local"].dt.tz_localize(None)
    day = local.dt.normalize()
    hour, minute = map(int, market.close_time.split(":"))
    after_close = local.dt.hour * 60 + local.dt.minute >= hour * 60 + minute
    side_date = np.where(after_close, (day + pd.Timedelta(days=1)).values, day.values)
    idx = np.searchsorted(sess, side_date, side="left")
    ok = idx < len(sess)
    out = rows.loc[ok].copy()
    out.insert(1, "date", sess[idx[ok]])
    return out.sort_values("release_et").drop_duplicates(["ticker", "date"], keep="first").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Point-in-time attachment (pure, tested)
# --------------------------------------------------------------------------------------
def consensus_features(events: pd.DataFrame, cons: pd.DataFrame, sessions: pd.DatetimeIndex | np.ndarray,
                       window: int | None = None, *, market: config.Market | None = None) -> pd.DataFrame:
    """events: ticker, date (index = event index). cons: `consensus_eps` rows. sessions: trading calendar.
    Returns a frame aligned to events.index with FEATURE_COLS (NaN where nothing qualifies)."""
    win = config.SETTINGS.consensus.window_sessions if window is None else window
    out = pd.DataFrame(index=events.index, columns=FEATURE_COLS, dtype="float64")
    out["eps_est_available"] = 0.0
    if cons is None or cons.empty or events.empty:
        return out
    sess = np.asarray(pd.DatetimeIndex(sessions).sort_values().values, dtype="datetime64[ns]")
    left = events[["ticker", "date"]].copy()
    market = config.MARKET if market is None else market
    hour, minute = map(int, market.close_time.split(":"))
    close = pd.to_datetime(left["date"]) + pd.Timedelta(hours=hour, minutes=minute)
    left["cutoff"] = close.dt.tz_localize(market.timezone).dt.tz_convert("America/New_York").dt.tz_localize(None)
    left = left.reset_index().sort_values("cutoff")
    right = cons.dropna(subset=["release_et"]).sort_values("release_et")[
        ["ticker", "release_et", "release_date", "eps_estimate", "reported_eps", "surprise_pct"]]
    m = pd.merge_asof(left, right, left_on="cutoff", right_on="release_et", by="ticker", direction="backward")
    ev_i = np.searchsorted(sess, pd.to_datetime(m["date"]).values.astype("datetime64[ns]"), side="left")
    rl_i = np.searchsorted(sess, pd.to_datetime(m["release_date"]).values.astype("datetime64[ns]"), side="left")
    age = ev_i - rl_i
    ok = m["release_et"].notna().to_numpy() & (age >= 0) & (age <= win)
    est, rep, sur = m["eps_estimate"], m["reported_eps"], m["surprise_pct"]
    calc = (rep - est) / est.abs().replace(0, np.nan) * 100
    surprise = calc.where(calc.notna(), sur).clip(-100, 100)
    m["eps_surprise_pct"] = surprise.where(ok)
    m["eps_beat"] = (surprise > 0).astype(float).where(ok & surprise.notna())
    m["eps_est_available"] = (est.notna() & rep.notna() & ok).astype(float)
    m["eps_release_age_sessions"] = pd.Series(age, index=m.index).where(ok).astype(float)
    res = m.set_index("index")[FEATURE_COLS].reindex(events.index)
    res["eps_est_available"] = res["eps_est_available"].fillna(0.0)
    return res
