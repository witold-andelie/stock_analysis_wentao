"""German benchmark index levels from Yahoo and daily 10-year Bund yields from Bundesbank.

Every function that loads or writes data refuses to run unless the active market is `de`, so German series can
never be written into the US warehouse. This module only ingests raw series; the
realised-volatility proxy is computed by indicators.py.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re

import pandas as pd

from . import config, prices, store

RATES_TABLE = "macro"
MARKET_TABLE = prices.MARKET_TABLE
REPULL_DAYS = 60


def _require_de() -> None:
    """Refuse to load German series into any other market's warehouse."""
    if config.MARKET.code != "de":
        raise RuntimeError(f"de_market needs DRIFTREV_MARKET=de (active market: {config.MARKET.code})")


def bundesbank_url(series_key: str, start: str) -> str:
    """Build a Bundesbank CSV URL from a <flow>/<key> series identifier."""
    flow, key = series_key.split("/", 1)
    return f"{config.BUNDESBANK_API}/{flow}/{key}?format=csv&lang=en&startPeriod={start}"


def parse_bundesbank_csv(text: str, name: str) -> pd.DataFrame:
    """Parse dated numeric observations, ignoring metadata and missing-value flags."""
    rows = [row[:2] for row in csv.reader(io.StringIO(text.lstrip("\ufeff")))
            if len(row) >= 2 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[0])]
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce").dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df["series"] = name
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)


def fetch_bundesbank(series_key: str, start: str) -> pd.DataFrame:
    """Fetch the raw daily yield observations for a configured Bundesbank series."""
    _require_de()
    resp = store.get(store.session(), bundesbank_url(series_key, start))
    if resp.status_code != 200:
        raise RuntimeError(f"Bundesbank HTTP {resp.status_code} for {series_key}")
    return parse_bundesbank_csv(resp.text, config.DE_RATE_SERIES[series_key])


def update_de_rates() -> dict:
    """Incrementally load rates, re-pulling recent observations for vendor revisions."""
    _require_de()
    last = store.max_date_by(RATES_TABLE, "series")
    written = 0
    for series_key, name in config.DE_RATE_SERIES.items():
        start = ((last[name] - pd.Timedelta(days=REPULL_DAYS)).strftime("%Y-%m-%d")
                 if name in last else config.START_DATE)
        written += store.upsert(RATES_TABLE, fetch_bundesbank(series_key, start), ["series", "date"])
    return {"table": RATES_TABLE, "rows_written": written}


def update_de_market() -> dict:
    """Load the active German market's benchmark index levels through the price loader."""
    _require_de()
    return prices.update_prices([symbol for symbol, _ in config.MARKET.benchmarks],
                                table=MARKET_TABLE, batch=10)


def main() -> None:
    """Load German market series and print the ingestion reports."""
    _require_de()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps({"market": update_de_market(), "rates": update_de_rates()}, indent=2, default=str))


if __name__ == "__main__":
    main()
