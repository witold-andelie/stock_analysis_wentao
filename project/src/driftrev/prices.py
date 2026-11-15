"""Daily OHLCV from Yahoo Finance (yfinance), loaded incrementally into the warehouse.

Design choice for incremental safety: Yahoo's *Adj Close* rewrites the whole history every
time a new dividend is paid, so stored adjusted prices would silently go stale. We store the
split-adjusted OHLC plus the raw dividend column and build total returns ourselves
(`(close_t + div_t) / close_{t-1} - 1`). A split inside a new download triggers a full
re-download of that ticker so its stored history stays on one split basis.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from . import config, store

log = logging.getLogger(__name__)
PRICE_TABLE = "prices"
MARKET_TABLE = "market"
OVERLAP_DAYS = 7          # re-download a week so late vendor corrections are picked up


def _download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download one batch and return a long frame (ticker, date, open, high, low, close,
    volume, dividends, splits)."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, actions=True,
                      group_by="ticker", threads=True, progress=False, timeout=30)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
        raw.columns = pd.MultiIndex.from_product([[tickers[0]], raw.columns])
    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t].dropna(subset=["Close"]).copy()
        if df.empty:
            continue
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={"stock_splits": "splits"})
        df["ticker"] = t
        df.index.name = "date"
        frames.append(df.reset_index())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ("dividends", "splits"):
        if col not in out:
            out[col] = 0.0
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    cols = ["ticker", "date", "open", "high", "low", "close", "volume", "dividends", "splits"]
    out = out[cols]
    out["volume"] = out["volume"].astype("float64")
    return out


def _batched(items: list[str], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def update_prices(tickers: list[str], table: str = PRICE_TABLE, batch: int = 80) -> dict:
    """Incrementally bring `table` up to date for `tickers`. Returns a small report."""
    last = store.max_date_by(table, "ticker")
    groups: dict[str, list[str]] = {}
    for t in tickers:
        start = (last[t] - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d") if t in last else config.START_DATE
        groups.setdefault(start, []).append(t)

    written, failed, resplit = 0, [], []
    for start, names in sorted(groups.items()):
        for chunk in _batched(names, batch):
            df = pd.DataFrame()
            for attempt in range(3):
                try:
                    df = _download(chunk, start)
                    break
                except Exception as exc:  # yfinance raises a zoo of errors on throttling
                    log.warning("yfinance batch failed (%s), retry %d", exc, attempt + 1)
                    time.sleep(5 * (attempt + 1))
            got = set(df["ticker"]) if not df.empty else set()
            failed += [t for t in chunk if t not in got]
            if df.empty:
                continue
            # A split inside an incremental window -> reload that ticker from scratch.
            if start != config.START_DATE:
                split_names = df.loc[df["splits"].fillna(0) > 0, "ticker"].unique().tolist()
                if split_names:
                    resplit += split_names
                    df = df[~df["ticker"].isin(split_names)]
                    full = _download(split_names, config.START_DATE)
                    with store.connect() as con:
                        con.execute(f"delete from {table} where ticker in (select unnest(?))", [split_names])
                    df = pd.concat([df, full], ignore_index=True)
            written += store.upsert(table, df, ["ticker", "date"])
            log.info("prices: %s batch from %s -> %d rows", table, start, len(df))
    return {"table": table, "rows_written": written, "failed": sorted(set(failed)),
            "resplit": sorted(set(resplit))}


def update_market() -> dict:
    """Index / ETF / VIX series used as market context and benchmarks."""
    tickers = list(config.MARKET_TICKERS) + sorted(set(config.SECTOR_ETFS.values()))
    return update_prices(tickers, table=MARKET_TABLE, batch=40)


def load_prices(tickers: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    where, params = [], []
    if tickers:
        where.append("ticker in (select unnest(?))")
        params.append(list(tickers))
    if start:
        where.append("date >= ?")
        params.append(pd.Timestamp(start))
    clause = ("where " + " and ".join(where)) if where else ""
    df = store.read_table(PRICE_TABLE, clause + " order by ticker, date", params)
    return df


def load_market() -> pd.DataFrame:
    return store.read_table(MARKET_TABLE, "order by ticker, date")
