"""Wikipedia pageviews: a free daily measure of public attention (idea from TradePulse).

Wikimedia publishes day t's counts after day t ends (UTC), so a value for date t is only
usable from t+1. The feature builder therefore lags pageviews by one trading row.
Coverage starts 2015-07-01 (the REST API's first day).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pandas as pd

from . import store

log = logging.getLogger(__name__)
TABLE = "pageviews"
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
FIRST_DAY = pd.Timestamp("2015-07-01")
_LIMITER = store.RateLimiter(4)   # Wikimedia throttles bursts (HTTP 429) above a few req/s


def fetch_article(title: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = f"{API}/{quote(title, safe='')}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00"
    resp = store.get(store.session(), url, _LIMITER)
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return pd.DataFrame({
        "wikipedia_title": title,
        "date": pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d"),
        "views": [float(i["views"]) for i in items],
    })


def update_pageviews(titles: list[str], workers: int = 2) -> dict:
    last = store.max_date_by(TABLE, "wikipedia_title")
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    jobs = []
    for t in sorted({t for t in titles if isinstance(t, str) and t}):
        start = last[t] - pd.Timedelta(days=3) if t in last else FIRST_DAY
        if start <= end:
            jobs.append((t, start))

    def work(job):
        try:
            return fetch_article(job[0], job[1], end)
        except Exception as exc:
            log.warning("pageviews %s: %s", job[0], exc)
            return pd.DataFrame()

    written, buf = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, df in enumerate(pool.map(work, jobs), 1):
            if not df.empty:
                buf.append(df)
            if i % 200 == 0 or i == len(jobs):
                if buf:
                    written += store.upsert(TABLE, pd.concat(buf, ignore_index=True), ["wikipedia_title", "date"])
                    buf = []
                log.info("pageviews %d/%d articles", i, len(jobs))
    return {"table": TABLE, "articles": len(jobs), "rows_written": written}


def load_pageviews() -> pd.DataFrame:
    return store.read_table(TABLE)
