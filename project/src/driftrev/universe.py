"""Universe: current S&P 1500 constituents = S&P 500 + MidCap 400 + SmallCap 600 (Wikipedia).

For each name we keep ticker, company, GICS sector / sub-industry, the Wikipedia article
title (used for the pageviews attention feature) and the SEC CIK (mapped through the
SEC's own company_tickers.json, which is more reliable than the Wikipedia column).

Known limitation, stated in the README: this is a *current-constituents* universe; membership.py (R1)
adds removed names (is_current_member=False) and the point-in-time membership table.
"""

from __future__ import annotations

from urllib.parse import unquote, urljoin, urlsplit

import pandas as pd
from bs4 import BeautifulSoup

from . import config, store

PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def _wiki_title(cell) -> str | None:
    link = cell.find("a", href=True)
    if not link:
        return None
    url = urlsplit(urljoin("https://en.wikipedia.org/wiki/", link["href"]))
    if url.hostname == "en.wikipedia.org" and url.path.startswith("/wiki/"):
        return unquote(url.path[len("/wiki/"):])
    return None


def parse_constituents(html: str, index_name: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", class_="wikitable")
    header = [th.get_text(" ", strip=True).replace("GICSSector", "GICS Sector") for th in table.find("tr").find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        rec = {h: c for h, c in zip(header, cells)}
        rows.append({
            "ticker": rec["Symbol"].get_text(strip=True),
            "company": rec["Security"].get_text(strip=True),
            "sector": rec["GICS Sector"].get_text(strip=True),
            "sub_industry": rec["GICS Sub-Industry"].get_text(strip=True),
            "wikipedia_title": _wiki_title(rec["Security"]),
            "index_name": index_name,
        })
    return pd.DataFrame(rows)


def to_yahoo(ticker: str) -> str:
    """Wikipedia/SEC use BRK.B, Yahoo uses BRK-B."""
    return ticker.replace(".", "-").upper()


def sec_cik_map() -> pd.DataFrame:
    sess = store.session()
    data = store.get(sess, "https://www.sec.gov/files/company_tickers.json").json()
    df = pd.DataFrame(data.values())
    df["ticker_yahoo"] = df["ticker"].map(to_yahoo)
    return df.rename(columns={"cik_str": "cik"})[["ticker_yahoo", "cik"]].drop_duplicates("ticker_yahoo")


def fetch_universe() -> pd.DataFrame:
    sess = store.session()
    frames = []
    for name in config.UNIVERSE_INDEXES:
        html = store.get(sess, PAGES[name]).text
        frames.append(parse_constituents(html, name))
    uni = pd.concat(frames, ignore_index=True).drop_duplicates("ticker")
    uni["cap_bucket"] = uni["index_name"].map({"sp500": "large", "sp400": "mid", "sp600": "small"})
    uni["ticker"] = uni["ticker"].map(to_yahoo)
    uni = uni.merge(sec_cik_map(), left_on="ticker", right_on="ticker_yahoo", how="left")
    uni = uni.drop(columns="ticker_yahoo")
    uni["cik"] = uni["cik"].astype("Int64")
    uni["sector_etf"] = uni["sector"].map(config.SECTOR_ETFS)
    uni = uni.sort_values("ticker").reset_index(drop=True)
    if config.MAX_TICKERS:
        uni = uni.head(config.MAX_TICKERS)
    return uni


META_TABLE = "universe_meta"
MAX_AGE_DAYS = 7.0


def needs_refresh(meta: dict | None, max_tickers: int, now: pd.Timestamp | None = None) -> bool:
    """True when the stored universe must be rebuilt: it was truncated by a smaller DRIFTREV_MAX_TICKERS than
    now requested (0 = all), or it is older than MAX_AGE_DAYS. No meta (legacy table) -> keep it."""
    if not meta:
        return False
    built_max = int(meta.get("max_tickers") or 0)
    if built_max and (max_tickers == 0 or max_tickers > built_max):
        return True
    built_at = pd.to_datetime(meta.get("built_at"), errors="coerce")
    now = now if now is not None else pd.Timestamp.now()
    return bool(pd.notna(built_at) and (now - built_at) > pd.Timedelta(days=MAX_AGE_DAYS))


def load_or_fetch(refresh: bool = False) -> pd.DataFrame:
    """Universe is refreshed when asked, when the stored one was built with a smaller MAX_TICKERS than now,
    or when it is older than a week (it changes a few times a quarter)."""
    if not refresh and store.table_exists("universe"):
        meta_df = store.read_table(META_TABLE)
        meta = meta_df.iloc[0].to_dict() if not meta_df.empty else None
        if not needs_refresh(meta, config.MAX_TICKERS):
            uni = store.read_table("universe")
            return uni.head(config.MAX_TICKERS) if config.MAX_TICKERS else uni
    uni = fetch_universe()
    uni["is_current_member"] = True
    if store.table_exists("universe"):       # keep removed names recovered by membership.py (R1)
        old = store.read_table("universe")
        if "is_current_member" in old:
            gone = old[~old["is_current_member"].fillna(True).astype(bool) & ~old["ticker"].isin(uni["ticker"])]
            uni = pd.concat([uni, gone], ignore_index=True)
    store.replace_table("universe", uni)
    store.replace_table(META_TABLE, pd.DataFrame({"max_tickers": [int(config.MAX_TICKERS)], "built_at": [pd.Timestamp.now()]}))
    return uni
