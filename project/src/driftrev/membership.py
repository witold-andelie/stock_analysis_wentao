"""R1 · Historical S&P 1500 membership (survivorship-bias reduction).

The universe in `universe.py` is *today's* S&P 1500, so every stock that crashed and was then
dropped from the index (or delisted) is missing: a strategy that buys crashes looks much better
than it is. This module reconstructs who was a member on each date from the Wikipedia
"Historical components of the S&P 500 / 400 / 600" change tables (added / removed + effective
date), walking backwards from today's constituents, and (best effort) recovers prices for the
removed names so they can enter the event dataset while they were members.

Method (per index, per ticker, `build_membership`)
    * the ticker's add / remove rows are sorted by date and replayed as a small state machine;
    * a current member with no rows has been a member since forever (start 1900-01-01);
    * a ticker whose first row is a *removal* was a member before it (start 1900-01-01);
    * member iff start_date <= date < end_date; end_date is the effective removal date;
    * a current member whose last row is a removal re-joined on an undocumented date: we open
      the interval at that removal date and count it as `inconsistent_current`;
    * a non-current ticker added but never removed (a rename such as FB -> META, or a missing
      row) cannot be resolved and is dropped (`orphans`).

Documented gaps (printed by `update_membership` and in the README/OPL):
    * Coverage start: S&P 500 table goes back to 1976, S&P 400 to 2012, S&P 600 only to
      2019-12. Names removed from the S&P 600 before 2019-12 are unknowable here, and current
      S&P 600 members who joined before then are assumed members for the whole sample.
    * Ticker reuse: the state machine is per ticker string, so a recycled ticker is treated as
      one company; removed-name prices are accepted only if Yahoo's series overlaps the
      membership window (>= `min_overlap_days`) and are clipped to the window + `price_tail_days`.
    * Yahoo no longer serves most delisted tickers, so many removed names stay without prices.
    * Index migrations (e.g. 400 -> 500) are merged: `in_index` means member of ANY of the three.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from . import config, prices, store, universe

log = logging.getLogger(__name__)
TABLE = "membership_history"
FOREVER = pd.Timestamp("1900-01-01")
HISTORY_PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
    "sp400": "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_400",
    "sp600": "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_600",
}
CAP_BUCKET = {"sp500": "large", "sp400": "mid", "sp600": "small"}
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")
_STOP = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "holdings", "holding", "group", "the",
         "plc", "llc", "lp", "sa", "nv", "cl", "class", "a", "b", "c", "de", "of", "and", "&", "trust", "new"}


# --------------------------------------------------------------------------------------
# 1. Parsing the change tables
# --------------------------------------------------------------------------------------
def parse_changes(html: str, index_name: str) -> pd.DataFrame:
    """Rows: date, added_ticker, added_name, removed_ticker, removed_name, reason, index_name."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "changes"})
    if table is None:
        return pd.DataFrame(columns=_CHANGE_COLS)
    rows, last_date = [], None
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 5 or tr.find("th", recursive=False) and not tr.find("td"):
            continue
        d = pd.to_datetime(cells[0], errors="coerce")
        if pd.isna(d):                       # date cell merged (rowspan): inherit the previous date
            if last_date is None:
                continue
            d, cells = last_date, [None] + cells
        last_date = d
        if len(cells) < 6:
            cells = cells + [""] * (6 - len(cells))
        a_t, a_n, r_t, r_n, reason = cells[1:6]
        rows.append({
            "date": d,
            "added_ticker": _clean_ticker(a_t), "added_name": a_n or None,
            "removed_ticker": _clean_ticker(r_t), "removed_name": r_n or None,
            "reason": reason or None, "index_name": index_name})
    return pd.DataFrame(rows, columns=_CHANGE_COLS)


_CHANGE_COLS = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason", "index_name"]


def _clean_ticker(raw: str | None) -> str | None:
    if not raw:
        return None
    t = re.sub(r"\[.*?\]", "", raw).strip().upper()
    return universe.to_yahoo(t) if _TICKER.match(t) else None


def fetch_changes(refresh: bool = False, max_age_hours: float = 24) -> pd.DataFrame:
    """Download (cached 24 h on disk, Wikimedia <= `wiki_rps`) and parse the three change tables."""
    limiter = store.RateLimiter(config.SETTINGS.membership.wiki_rps)
    sess = store.session()
    cache = config.CACHE_DIR / "wikipedia"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, url in HISTORY_PAGES.items():
        path = cache / f"history_{name}.html"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
        if fresh and not refresh:
            html = path.read_text(encoding="utf-8")
        else:
            html = store.get(sess, url, limiter).text
            path.write_text(html, encoding="utf-8")
        frames.append(parse_changes(html, name))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# 2. Reconstruction (pure, tested)
# --------------------------------------------------------------------------------------
def build_membership(changes: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """changes: parse_changes output (all indices). current: DataFrame(ticker, index_name) of today's
    constituents. Returns (membership[ticker, index_name, start_date, end_date], diagnostics).
    end_date is NaT for open intervals (still a member)."""
    out, diag = [], {"inconsistent_current": [], "orphans": [], "coverage_start": {}}
    for idx, ch in changes.groupby("index_name"):
        diag["coverage_start"][idx] = ch["date"].min()
        events: dict[str, list[tuple[pd.Timestamp, int]]] = {}
        for r in ch.itertuples():
            if r.added_ticker:
                events.setdefault(r.added_ticker, []).append((r.date, 1))
            if r.removed_ticker:
                events.setdefault(r.removed_ticker, []).append((r.date, -1))
        now = set(current.loc[current["index_name"] == idx, "ticker"])
        for tkr in sorted(now | set(events)):
            evs = sorted(events.get(tkr, []), key=lambda e: (e[0], -e[1]))  # same day: add replayed before removal
            is_now = tkr in now
            open_start = FOREVER if (not evs or evs[0][1] == -1) else None
            last_removal = None
            for d, kind in evs:
                if kind == 1:
                    if open_start is None:
                        open_start = d
                else:
                    if open_start is not None:
                        out.append((tkr, idx, open_start, d))
                        open_start = None
                    last_removal = d
            if is_now:
                if open_start is None:
                    diag["inconsistent_current"].append(tkr)
                    open_start = last_removal
                out.append((tkr, idx, open_start, pd.NaT))
            elif open_start is not None and evs:
                diag["orphans"].append(tkr)             # added, never removed, not current: renamed / missing row
    mem = pd.DataFrame(out, columns=["ticker", "index_name", "start_date", "end_date"])
    mem["start_date"] = pd.to_datetime(mem["start_date"])
    mem["end_date"] = pd.to_datetime(mem["end_date"])
    return mem.sort_values(["ticker", "start_date"]).reset_index(drop=True), diag


def merged_intervals(membership: pd.DataFrame, gap_days: int | None = None) -> dict[str, list[tuple[np.datetime64, np.datetime64]]]:
    """Union of the three indices per ticker as sorted, gap-merged [start, end) intervals (NaT end -> +inf)."""
    gap = pd.Timedelta(days=config.SETTINGS.membership.merge_gap_days if gap_days is None else gap_days)
    far = pd.Timestamp("2262-01-01")
    res: dict[str, list] = {}
    if membership is None or membership.empty:
        return res
    m = membership.assign(end_date=membership["end_date"].fillna(far)).sort_values(["ticker", "start_date"])
    for tkr, g in m.groupby("ticker"):
        cur: list = []
        for s, e in zip(g["start_date"], g["end_date"]):
            if cur and s <= cur[-1][1] + gap:
                cur[-1][1] = max(cur[-1][1], e)
            else:
                cur.append([s, e])
        res[tkr] = [(np.datetime64(s), np.datetime64(e)) for s, e in cur]
    return res


def bucket_intervals(membership: pd.DataFrame) -> dict[str, list[tuple[np.datetime64, np.datetime64, str]]]:
    """Per ticker: [start, end) intervals tagged with the cap bucket of the index it sat in (NaT end -> +inf).
    Overlapping rows (an index migration 400 -> 500 on the same day) are resolved to the LARGER bucket."""
    far = pd.Timestamp("2262-01-01")
    res: dict[str, list] = {}
    if membership is None or membership.empty:
        return res
    m = membership.assign(end_date=membership["end_date"].fillna(far))
    m = m[m["index_name"].isin(CAP_BUCKET)].sort_values(["ticker", "start_date"])
    for tkr, g in m.groupby("ticker"):
        res[tkr] = [(np.datetime64(s), np.datetime64(e), CAP_BUCKET[i])
                    for s, e, i in zip(g["start_date"], g["end_date"], g["index_name"])]
    return res


_BUCKET_RANK = {"large": 3, "mid": 2, "small": 1}


def cap_bucket_at(dates, intervals) -> np.ndarray:
    """Point-in-time cap bucket (object array, None where no interval covers the date): the bucket of the index
    the stock was in ON that date, not the one it sits in today (a stock that later grew into the S&P 500 was
    not 'large' when it was a small cap -> using today's bucket leaks the future and understates its costs)."""
    d = np.asarray(pd.to_datetime(dates).values, dtype="datetime64[ns]")
    out = np.full(len(d), None, dtype=object)
    best = np.zeros(len(d), dtype=int)
    for s, e, b in intervals or []:
        hit = (d >= s) & (d < e) & (_BUCKET_RANK[b] > best)
        out[hit] = b
        best[hit] = _BUCKET_RANK[b]
    return out


def in_index_mask(dates, intervals) -> np.ndarray:
    """Boolean array: date inside any [start, end) interval. `intervals` None -> no information -> all True."""
    d = np.asarray(pd.to_datetime(dates).values, dtype="datetime64[ns]")
    if intervals is None:
        return np.ones(len(d), dtype=bool)
    mask = np.zeros(len(d), dtype=bool)
    for s, e in intervals:
        mask |= (d >= s) & (d < e)
    return mask


# --------------------------------------------------------------------------------------
# 3. Metadata for removed names (CIK, SIC sector)
# --------------------------------------------------------------------------------------
def _tokens(name: str | None) -> set[str]:
    if not name:
        return set()
    toks = re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))
    return {t for t in toks if t not in _STOP and len(t) > 1}


def name_match(a: str | None, b: str | None, min_overlap: float | None = None) -> bool:
    """Token overlap (share of the smaller set) between two company names: ticker-reuse guard."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    thr = config.SETTINGS.membership.name_match_min if min_overlap is None else min_overlap
    return len(ta & tb) / min(len(ta), len(tb)) >= thr


def sic_to_sector(sic) -> str | None:
    """Approximate SIC -> GICS sector (documented approximation; used only for removed names)."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    table = [
        (100, 999, "Consumer Staples"), (1000, 1299, "Materials"), (1300, 1399, "Energy"), (1400, 1499, "Materials"),
        (1500, 1799, "Industrials"), (2000, 2199, "Consumer Staples"), (2200, 2599, "Consumer Discretionary"),
        (2600, 2699, "Materials"), (2700, 2799, "Communication Services"), (2833, 2836, "Health Care"),
        (2844, 2844, "Consumer Staples"), (2800, 2899, "Materials"), (2900, 2999, "Energy"),
        (3000, 3199, "Consumer Discretionary"), (3200, 3399, "Materials"), (3400, 3569, "Industrials"),
        (3570, 3579, "Information Technology"), (3580, 3599, "Industrials"), (3600, 3699, "Information Technology"),
        (3711, 3716, "Consumer Discretionary"), (3700, 3799, "Industrials"), (3841, 3851, "Health Care"),
        (3800, 3899, "Information Technology"), (3900, 3999, "Consumer Discretionary"), (4000, 4799, "Industrials"),
        (4800, 4899, "Communication Services"), (4900, 4999, "Utilities"), (5000, 5199, "Industrials"),
        (5400, 5499, "Consumer Staples"), (5912, 5912, "Consumer Staples"), (5200, 5999, "Consumer Discretionary"),
        (6500, 6599, "Real Estate"), (6798, 6798, "Real Estate"), (6000, 6799, "Financials"),
        (7000, 7299, "Consumer Discretionary"), (7370, 7379, "Information Technology"), (7300, 7369, "Industrials"),
        (7800, 7999, "Communication Services"), (8000, 8099, "Health Care"), (8700, 8799, "Industrials"),
    ]
    for lo, hi, sec in table:       # first match wins: narrow ranges are listed before their wide parents
        if lo <= s <= hi:
            return sec
    return None


def _text(tag: str, xml: str) -> str | None:
    m = re.search(rf"<{tag}>([^<]*)</{tag}>", xml)
    return m.group(1).strip() if m else None


def edgar_lookup(ticker: str, limiter: store.RateLimiter, sess) -> dict | None:
    """EDGAR company search by ticker (atom). Returns cik / names / sic or None."""
    url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + ticker
           + "&type=&dateb=&owner=include&count=1&output=atom")
    for _attempt in range(2):
        try:
            xml = store.get(sess, url, limiter, retries=1, timeout=20).text
        except RuntimeError:
            xml = ""
        if "<conformed-name>" in xml:
            former = re.findall(r"<name>([^<]*)</name>", xml)
            return {"cik": int(_text("cik", xml)), "edgar_name": _text("conformed-name", xml),
                    "former_names": former, "sic": _text("assigned-sic", xml),
                    "sic_desc": _text("assigned-sic-desc", xml)}
        if "File Unavailable" in xml or "Request Rate Threshold" in xml:   # throttled: back off once
            time.sleep(3)
            continue
        return None
    return None


def _meta_cache_path() -> Path:
    return config.CACHE_DIR / "membership" / "removed_meta.json"


def submissions_info(cik: int, limiter: store.RateLimiter, sess) -> dict | None:
    """SIC + names from data.sec.gov submissions (fast, <= 8 rps). None if unavailable."""
    try:
        resp = store.get(sess, f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", limiter, retries=2)
    except RuntimeError:
        return None
    if resp.status_code != 200:
        return None
    j = resp.json()
    return {"edgar_name": j.get("name"), "former_names": [f.get("name") for f in j.get("formerNames", [])],
            "sic": j.get("sic"), "sic_desc": j.get("sicDescription")}


def resolve_removed_meta(removed: pd.DataFrame, sec_titles: dict[str, tuple[int, str]]) -> dict[str, dict]:
    """removed: DataFrame(ticker, name). Returns {ticker: {cik, sic, sector, sic_desc} or {}} (cached on disk).
    1) SEC company_tickers (name-checked, so a recycled ticker is rejected), SIC from the submissions API;
    2) otherwise the EDGAR company search by ticker (slow, throttled: used only as a fallback)."""
    path = _meta_cache_path()
    cache = json.loads(path.read_text()) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    sec_lim = store.RateLimiter(min(config.SEC_MAX_RPS, 8.0))
    search_lim = store.RateLimiter(config.SETTINGS.membership.sec_search_rps)
    sess = store.session()
    for n, r in enumerate(removed.itertuples(), 1):
        if r.ticker in cache:
            continue
        rec: dict = {}
        hit = sec_titles.get(r.ticker)
        if hit and name_match(r.name, hit[1]):
            info = submissions_info(hit[0], sec_lim, sess) or {}
            rec = {"cik": int(hit[0]), "sic": info.get("sic"), "sic_desc": info.get("sic_desc"),
                   "sector": sic_to_sector(info.get("sic"))}
        else:
            info = edgar_lookup(r.ticker, search_lim, sess)
            if info and (name_match(r.name, info["edgar_name"]) or any(name_match(r.name, f) for f in info["former_names"])):
                rec = {"cik": info["cik"], "sic": info["sic"], "sic_desc": info["sic_desc"],
                       "sector": sic_to_sector(info["sic"])}
        cache[r.ticker] = rec
        if n % 25 == 0:
            path.write_text(json.dumps(cache))
    path.write_text(json.dumps(cache))
    return cache


def _sec_titles() -> dict[str, tuple[int, str]]:
    data = store.get(store.session(), "https://www.sec.gov/files/company_tickers.json").json()
    return {universe.to_yahoo(v["ticker"]): (int(v["cik_str"]), v["title"]) for v in data.values()}


# --------------------------------------------------------------------------------------
# 4. Orchestration
# --------------------------------------------------------------------------------------
def clip_removed_prices(tickers: list[str], intervals: dict[str, list]) -> dict:
    """Keep each removed name's prices only inside [first membership start, last end + tail]; drop names whose
    Yahoo series barely overlaps the membership window (recycled ticker). Returns {recovered, reused}."""
    tail = pd.Timedelta(days=config.SETTINGS.membership.price_tail_days)
    min_ov = config.SETTINGS.membership.min_overlap_days
    px = store.read_table(prices.PRICE_TABLE, "where ticker in (select unnest(?))", [list(tickers)]) if tickers else pd.DataFrame()
    recovered, reused = [], []
    with store.connect() as con:
        for tkr in tickers:
            g = px[px["ticker"] == tkr] if not px.empty else px
            iv = intervals.get(tkr)
            if g.empty or not iv:
                continue
            d = g["date"].values
            overlap = sum(int(((d >= s) & (d < e)).sum()) for s, e in iv)
            last_end = pd.Timestamp(max(e for _, e in iv))
            if overlap < min_ov:
                con.execute(f"delete from {prices.PRICE_TABLE} where ticker = ?", [tkr])
                reused.append(tkr)
                continue
            con.execute(f"delete from {prices.PRICE_TABLE} where ticker = ? and date > ?", [tkr, last_end + tail])
            recovered.append(tkr)
    return {"recovered": recovered, "reused": reused}


def update_membership(refresh: bool = False, price_batch: int = 80) -> dict:
    """Incremental R1 load: change tables -> membership_history; removed names -> universe + prices."""
    uni = store.read_table("universe")
    if uni.empty:
        raise RuntimeError("load the universe first (universe.load_or_fetch)")
    if "is_current_member" not in uni:
        uni["is_current_member"] = True
    cur = uni[uni["is_current_member"].fillna(True).astype(bool)]
    changes = fetch_changes(refresh=refresh)
    mem, diag = build_membership(changes, cur[["ticker", "index_name"]])
    store.replace_table(TABLE, mem)

    start = pd.Timestamp(config.START_DATE)
    cur_set = set(cur["ticker"])
    ended = mem[mem["end_date"].notna() & (mem["end_date"] >= start) & ~mem["ticker"].isin(cur_set)]
    cand = sorted(ended["ticker"].unique())
    # last known name + index of each removed ticker
    names = pd.concat([changes[["date", "removed_ticker", "removed_name"]].rename(columns={"removed_ticker": "ticker", "removed_name": "name"}),
                       changes[["date", "added_ticker", "added_name"]].rename(columns={"added_ticker": "ticker", "added_name": "name"})])
    names = names.dropna(subset=["ticker", "name"]).sort_values("date").drop_duplicates("ticker", keep="last").set_index("ticker")["name"]
    last_idx = ended.sort_values("end_date").drop_duplicates("ticker", keep="last").set_index("ticker")["index_name"]

    intervals = merged_intervals(mem)
    price_rep = prices.update_prices(cand, batch=price_batch)
    clip = clip_removed_prices(cand, intervals)
    recovered = clip["recovered"]

    meta = resolve_removed_meta(pd.DataFrame({"ticker": recovered, "name": [names.get(t) for t in recovered]}), _sec_titles()) if recovered else {}
    rows = []
    for t in recovered:
        m = meta.get(t, {})
        sector = m.get("sector")
        rows.append({"ticker": t, "company": names.get(t), "sector": sector, "sub_industry": m.get("sic_desc"),
                     "wikipedia_title": None, "index_name": last_idx.get(t), "cap_bucket": CAP_BUCKET.get(last_idx.get(t)),
                     "cik": m.get("cik"), "sector_etf": config.SECTOR_ETFS.get(sector) if sector else None,
                     "is_current_member": False})
    removed_df = pd.DataFrame(rows, columns=list(uni.columns))
    removed_df["cik"] = pd.to_numeric(removed_df["cik"], errors="coerce").astype("float64")
    base = uni[uni["is_current_member"].fillna(True).astype(bool)].copy()
    base["is_current_member"] = True
    new_uni = pd.concat([base, removed_df], ignore_index=True)
    new_uni["is_current_member"] = new_uni["is_current_member"].astype(bool)
    store.replace_table("universe", new_uni)
    report = {
        "membership_rows": len(mem), "membership_tickers": int(mem["ticker"].nunique()),
        "coverage_start": {k: str(v.date()) for k, v in diag["coverage_start"].items()},
        "inconsistent_current": len(diag["inconsistent_current"]), "orphans": len(diag["orphans"]),
        "removed_candidates": len(cand), "recovered": len(recovered), "missing": len(cand) - len(recovered) - len(clip["reused"]),
        "ticker_reused_discarded": len(clip["reused"]),
        "recovered_with_cik": sum(1 for t in recovered if meta.get(t, {}).get("cik")),
        "recovered_with_sector": sum(1 for t in recovered if meta.get(t, {}).get("sector")),
        "yahoo_failed": len(price_rep["failed"]),
    }
    return report


def load_membership() -> pd.DataFrame:
    return store.read_table(TABLE)


# --------------------------------------------------------------------------------------
# Survivorship study (README §13 / notebook 05): how much does the current-constituents
# universe inflate event outcomes compared with the point-in-time universe?
# --------------------------------------------------------------------------------------
UNIVERSE_DEFS = {
    "current_members_any_date (old behaviour)": lambda e: e["is_current_member"].fillna(True).astype(bool),
    "point_in_time_all_names": lambda e: e["in_index"].fillna(False).astype(bool),
    "point_in_time_current_names": lambda e: e["in_index"].fillna(False).astype(bool) & e["is_current_member"].fillna(True).astype(bool),
    "point_in_time_removed_names": lambda e: e["in_index"].fillna(False).astype(bool) & ~e["is_current_member"].fillna(True).astype(bool),
    "current_names_before_joining (excluded now)": lambda e: ~e["in_index"].fillna(False).astype(bool) & e["is_current_member"].fillna(True).astype(bool),
}


def survivorship_table(events: pd.DataFrame, threshold: float = 1.02) -> pd.DataFrame:
    """events must be built WITHOUT the point-in-time restriction (restrict_events=False) and carry
    in_index, is_current_member, event_family, growth_future_10d(_vs_spy)."""
    known = events[events["growth_future_10d"].notna()]
    rows = []
    for name, rule in UNIVERSE_DEFS.items():
        sub = known[rule(known)]
        for fam, g in [("all", sub)] + list(sub.groupby("event_family")):
            rows.append({"universe": name, "event_family": fam, "events": len(g),
                         "base_rate": float((g["growth_future_10d"] >= threshold).mean()) if len(g) else float("nan"),
                         "mean_10d": float(g["growth_future_10d"].mean() - 1) if len(g) else float("nan"),
                         "mean_10d_vs_spy": float(g["growth_future_10d_vs_spy"].mean() - 1) if len(g) else float("nan")})
    return pd.DataFrame(rows)


def survivorship_study() -> pd.DataFrame:
    """Rebuild events without the point-in-time gate (~10 min on the full warehouse) and save
    artifacts/survivorship.parquet. Does not modify any warehouse table."""
    from . import artifacts, attention, consensus, edgar, features, macro, prices, universe

    uni = universe.load_or_fetch()
    ev, _, _ = features.build_all(
        uni, prices.load_prices(), prices.load_market(), macro.load_macro_wide(), edgar.load_filings(),
        edgar.sue_table(edgar.load_eps_facts()), attention.load_pageviews(),
        membership=load_membership(), consensus_eps=consensus.load_consensus(), restrict_events=False)
    if "is_current_member" not in ev:
        ev = ev.merge(uni[["ticker", "is_current_member"]], on="ticker", how="left")
    table = survivorship_table(ev, config.SETTINGS.target.threshold)
    artifacts.save_frame("survivorship.parquet", table)
    return table
