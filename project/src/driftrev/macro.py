"""FRED macro series (rates, credit spread) via the public fredgraph CSV endpoint.

FRED publishes a value for day t after the US close of t (often the next morning), so the
feature builder lags every macro series by one row before joining it to events.
"""

from __future__ import annotations

import io

import pandas as pd

from . import config, store

TABLE = "macro"


def fetch_fred(series_id: str, start: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    text = store.get(store.session(), url).text
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series"] = config.FRED_SERIES[series_id]
    return df.dropna(subset=["value"])


def update_macro() -> dict:
    """FRED revises recent values, so each run re-pulls the last 60 days per series."""
    last = store.max_date_by(TABLE, "series")
    written = 0
    for sid, name in config.FRED_SERIES.items():
        start = (last[name] - pd.Timedelta(days=60)).strftime("%Y-%m-%d") if name in last else config.START_DATE
        written += store.upsert(TABLE, fetch_fred(sid, start), ["series", "date"])
    return {"table": TABLE, "rows_written": written}


def load_macro_wide() -> pd.DataFrame:
    df = store.read_table(TABLE)
    if df.empty:
        return df
    return df.pivot_table(index="date", columns="series", values="value").sort_index()
