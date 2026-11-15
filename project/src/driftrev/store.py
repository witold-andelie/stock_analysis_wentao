"""DuckDB warehouse + small HTTP helpers shared by all data sources.

The warehouse (`data/warehouse.duckdb`) is the single place where raw and
transformed data persist between runs, which is what makes loading incremental:
each source asks the warehouse for its last stored date and downloads only the gap.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
import pandas as pd
import requests

from . import config

_LOCK = threading.RLock()  # re-entrant: helpers may nest


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the warehouse. Writers are serialised by a process-wide lock."""
    config.ensure_dirs()
    if read_only and not config.WAREHOUSE.exists():
        raise FileNotFoundError(f"warehouse not found: {config.WAREHOUSE}")
    if read_only:
        con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
        try:
            yield con
        finally:
            con.close()
        return
    with _LOCK:
        con = duckdb.connect(str(config.WAREHOUSE))
        try:
            yield con
        finally:
            con.close()


def table_exists(name: str) -> bool:
    if not config.WAREHOUSE.exists():
        return False
    with connect() as con:
        return bool(con.execute(
            "select count(*) from information_schema.tables where table_name = ?", [name]
        ).fetchone()[0])


def read_table(name: str, where: str = "", params: list | None = None) -> pd.DataFrame:
    """Read a whole table (optionally filtered). `where` is a trusted SQL literal built by code."""
    if not table_exists(name):
        return pd.DataFrame()
    with connect() as con:
        return con.execute(f"select * from {name} {where}", params or []).df()


def replace_table(name: str, df: pd.DataFrame) -> None:
    with connect() as con:
        con.register("df_view", df)
        con.execute(f"create or replace table {name} as select * from df_view")
        con.unregister("df_view")


def upsert(name: str, df: pd.DataFrame, keys: list[str]) -> int:
    """Insert rows, replacing existing rows with the same key. Returns rows written."""
    if df.empty:
        return 0
    with connect() as con:
        con.register("df_view", df)
        exists = con.execute(
            "select count(*) from information_schema.tables where table_name = ?", [name]
        ).fetchone()[0]
        if not exists:
            con.execute(f"create table {name} as select * from df_view")
        else:
            cond = " and ".join(f"t.{k} = d.{k}" for k in keys)
            con.execute(f"delete from {name} t using df_view d where {cond}")
            cols = ", ".join(f'"{c}"' for c in df.columns)
            con.execute(f"insert into {name} ({cols}) select {cols} from df_view")
        con.unregister("df_view")
    return len(df)


def max_date_by(name: str, key: str, date_col: str = "date") -> dict[str, pd.Timestamp]:
    if not table_exists(name):
        return {}
    with connect() as con:
        rows = con.execute(f"select {key}, max({date_col}) from {name} group by 1").fetchall()
    return {k: pd.Timestamp(v) for k, v in rows if v is not None}


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------
class RateLimiter:
    """Thread-safe 'at most N requests per second' limiter."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_time - now
            self.next_time = max(now, self.next_time) + self.interval
        if delay > 0:
            time.sleep(delay)


def session(user_agent: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent or config.USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s


def get(sess: requests.Session, url: str, limiter: RateLimiter | None = None,
        retries: int = 4, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff on 429/5xx. 404 is returned to the caller."""
    last = None
    for attempt in range(retries):
        if limiter:
            limiter.wait()
        try:
            resp = sess.get(url, timeout=timeout)
        except requests.RequestException as exc:  # network blip
            last = exc
            time.sleep(1.5 * 2 ** attempt)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            last = RuntimeError(f"HTTP {resp.status_code} for {url}")
            time.sleep(2.0 * 2 ** attempt)
            continue
        return resp
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")
