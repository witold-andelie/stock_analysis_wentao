"""SEC EDGAR: filing history (submissions API) and point-in-time diluted EPS (XBRL).

New data source (not yfinance / FRED). Two products:

* `filings`   one row per filing: form, 8-K item codes, acceptance time in US/Eastern.
              Used to (a) date earnings releases (8-K Item 2.02) precisely and (b) build the
              event taxonomy for crash days (offerings, restatements, delisting notices ...).
* `eps_q`     first-reported quarterly diluted EPS with a seasonal random-walk SUE
              (the TradePulse recipe): only the value the market saw first is kept, so
              later restatements cannot leak into the past.

Timing fact verified while building this: `acceptanceDateTime` in the submissions JSON is
real UTC (Apple's 16:30 ET earnings 8-Ks show 20:30Z in summer and 21:30Z in winter).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config, store

log = logging.getLogger(__name__)
FILINGS = "filings"
EPS_FACTS = "eps_facts"
FETCH_LOG = "edgar_fetch_log"
FORMS_KEPT = {
    "8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A", "NT 10-K", "NT 10-Q",
    "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR", "424B1", "424B2", "424B3", "424B4", "424B5",
    "424B7", "FWP", "4", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A", "SCHEDULE 13D",
    "SCHEDULE 13D/A", "25-NSE", "DEF 14A", "15-12B",
}
_LIMITER = store.RateLimiter(config.SEC_MAX_RPS)


def cik10(cik: int) -> str:
    return f"{int(cik):010d}"


def _get_json(url: str) -> dict | None:
    resp = store.get(store.session(), url, _LIMITER)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _cached_json(url: str, name: str, refresh: bool) -> dict | None:
    """Older submission pages never change -> cache them forever; `refresh` forces a re-pull."""
    path = config.CACHE_DIR / "edgar" / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    payload = _get_json(url)
    if payload is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return payload


def _parse_block(block: dict, cik: int) -> pd.DataFrame:
    if not block or not block.get("form"):
        return pd.DataFrame()
    df = pd.DataFrame({
        "accession": block["accessionNumber"],
        "filing_date": block["filingDate"],
        "accepted": block["acceptanceDateTime"],
        "form": block["form"],
        "items": block.get("items", [""] * len(block["form"])),
    })
    df = df[df["form"].isin(FORMS_KEPT)].copy()
    df["cik"] = int(cik)
    acc = pd.to_datetime(df["accepted"], utc=True, errors="coerce", format="ISO8601")
    df["accepted_et"] = acc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["items"] = df["items"].fillna("").astype(str)
    return df.drop(columns="accepted")


def fetch_filings(cik: int, start: str = config.START_DATE, recent_only: bool = False) -> pd.DataFrame:
    """`recent_only`: the company's history is already in the warehouse, so only the `recent`
    block (last ~1000 filings / >= 1 year) is needed. Keeps daily CI runs to 1 request/company."""
    root = _get_json(f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json")  # always fresh, never cached
    if root is None:
        return pd.DataFrame()
    parts = [_parse_block(root["filings"]["recent"], cik)]
    for f in ([] if recent_only else root["filings"].get("files", [])):
        if f.get("filingTo", "9999") < start:
            continue  # page entirely before our window
        page = _cached_json(f"https://data.sec.gov/submissions/{f['name']}", f["name"], refresh=False)
        parts.append(_parse_block(page, cik))
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
    if df.empty:
        return df
    return df[df["filing_date"] >= pd.Timestamp(start)].drop_duplicates("accession")


def fetch_eps_facts(cik: int) -> pd.DataFrame:
    """Diluted EPS facts (fallback: basic). One small JSON per company."""
    for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
        payload = _get_json(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik10(cik)}/us-gaap/{concept}.json")
        if not payload:
            continue
        units = payload.get("units", {})
        rows = units.get("USD/shares") or next(iter(units.values()), [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "start" not in df:
            continue
        df = df.rename(columns={"val": "value", "accn": "accession"})
        df["cik"] = int(cik)
        df["concept"] = concept
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c])
        return df[["cik", "concept", "start", "end", "value", "filed", "form", "accession"]]
    return pd.DataFrame()


def update_edgar(ciks: list[int], max_age_days: float = 0.5, workers: int = 8) -> dict:
    """Refresh filings + EPS for every CIK not refreshed within `max_age_days`."""
    log_df = store.read_table(FETCH_LOG)
    fresh = set()
    if not log_df.empty:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        fresh = set(log_df.loc[log_df["fetched_at"] >= cutoff, "cik"].astype(int))
    todo = [int(c) for c in ciks if pd.notna(c) and int(c) not in fresh]
    known = set()
    if store.table_exists(FILINGS):
        with store.connect() as con:
            known = {int(r[0]) for r in con.execute(f"select distinct cik from {FILINGS}").fetchall()}

    def work(cik: int):
        try:
            return cik, fetch_filings(cik, recent_only=cik in known), fetch_eps_facts(cik), None
        except Exception as exc:  # keep going; report at the end
            return cik, pd.DataFrame(), pd.DataFrame(), str(exc)

    filings, eps, errors, done = [], [], {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (cik, f, e, err) in enumerate(pool.map(work, todo), 1):
            if err:
                errors[cik] = err
            else:
                done.append(cik)
                filings.append(f)
                eps.append(e)
            if i % 100 == 0:
                log.info("edgar %d/%d companies", i, len(todo))
                _flush(filings, eps, done)
                filings, eps, done = [], [], []
    _flush(filings, eps, done)
    return {"companies_refreshed": len(todo) - len(errors), "errors": errors}


def _flush(filings: list, eps: list, done: list) -> None:
    f = pd.concat([x for x in filings if not x.empty], ignore_index=True) if any(not x.empty for x in filings) else pd.DataFrame()
    e = pd.concat([x for x in eps if not x.empty], ignore_index=True) if any(not x.empty for x in eps) else pd.DataFrame()
    if not f.empty:
        store.upsert(FILINGS, f, ["accession", "cik"])
    if not e.empty:
        if store.table_exists(EPS_FACTS):  # facts are re-pulled whole per company
            with store.connect() as con:
                con.execute(f"delete from {EPS_FACTS} where cik in (select unnest(?))", [sorted(set(e['cik'].tolist()))])
        store.upsert(EPS_FACTS, e, ["cik", "concept", "start", "end", "accession"])
    if done:
        store.upsert(FETCH_LOG, pd.DataFrame({"cik": done, "fetched_at": pd.Timestamp.now()}), ["cik"])


# --------------------------------------------------------------------------------------
# Point-in-time SUE (ported from TradePulse quant/earnings.py)
# --------------------------------------------------------------------------------------
def quarterly_eps(facts: pd.DataFrame) -> pd.DataFrame:
    """First-reported quarterly EPS for one company. Q4 = FY - (Q1+Q2+Q3) when needed."""
    rep = (facts.sort_values("filed", kind="stable")
           .drop_duplicates(["start", "end"], keep="first"))
    dur = (rep["end"] - rep["start"]).dt.days
    q = rep[dur.between(80, 100)]
    fy = rep[dur.between(350, 380)]
    derived = []
    for a in fy.itertuples(index=False):
        inside = q[(q["start"] >= a.start) & (q["end"] < a.end)]
        if len(inside) == 3 and not q["end"].eq(a.end).any() and inside["filed"].le(a.filed).all():
            derived.append({"end": a.end, "value": a.value - inside["value"].sum(), "filed": a.filed})
    out = pd.concat([q[["end", "value", "filed"]], pd.DataFrame(derived, columns=["end", "value", "filed"])],
                    ignore_index=True)
    return out.sort_values("end").drop_duplicates("end").reset_index(drop=True)


def sue_table(eps_facts: pd.DataFrame) -> pd.DataFrame:
    """cik, period_end, filed, eps, sue for every company. sue = (EPS_q - EPS_{q-4}) /
    std(previous 4..8 such diffs)."""
    out = []
    for cik, g in eps_facts.groupby("cik"):
        g = g[g["concept"] == g["concept"].iloc[0]]
        q = quarterly_eps(g)
        if q.empty:
            continue
        q["diff"] = q["value"].diff(4)
        scale = q["diff"].shift(1).rolling(8, min_periods=4).std(ddof=1)
        q["sue"] = q["diff"] / scale.where(scale > 0)
        q["eps_yoy_pos"] = (q["diff"] > 0).astype(float).where(q["diff"].notna())
        q["cik"] = int(cik)
        out.append(q.rename(columns={"end": "period_end", "value": "eps"}))
    if not out:
        return pd.DataFrame(columns=["cik", "period_end", "eps", "filed", "diff", "sue", "eps_yoy_pos"])
    res = pd.concat(out, ignore_index=True)
    res["sue"] = res["sue"].clip(-10, 10).replace([np.inf, -np.inf], np.nan)
    return res


def load_filings() -> pd.DataFrame:
    return store.read_table(FILINGS)


def load_eps_facts() -> pd.DataFrame:
    return store.read_table(EPS_FACTS)
