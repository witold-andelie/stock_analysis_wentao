"""Resolve German member names to Yahoo listings and load their raw prices.

Search names must pass a token-prefix check; Xetra beats Hamburg regardless of
query order. Hamburg preserves a price source for delisted names, with a separate
flag so downstream liquidity filters can distinguish it. Unmatched companies and
shared symbols remain visible in diagnostics; manual overrides require evidence.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config, de_market, de_membership, prices, store

log = logging.getLogger(__name__)
TABLE = "universe"
OVERRIDES_PATH = Path(__file__).parent / "resources" / "de_symbol_overrides.csv"
UNIVERSE_COLS = ["company_key", "name", "symbol", "exchange", "price_source", "resolved_by", "yahoo_name",
                 "matched_query", "index_code", "cap_bucket", "is_current_member", "first_date", "last_date"]
PRICE_SOURCE = {"GER": "xetra", "HAM": "hamburg"}
_STOP = {"se", "ag", "aktiengesellschaft", "co", "kgaa", "sa", "nv", "gmbh", "plc", "inc", "ltd", "the", "und", "and"}
FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss",
                      "é": "e", "ó": "o", "á": "a", "è": "e"})
_RESOLVED_COLS = ["company_key", "symbol", "exchange", "price_source", "resolved_by", "yahoo_name", "matched_query"]
_QUOTE_FIELDS = ("symbol", "exchange", "quoteType", "shortname", "longname")
_search_limiter: store.RateLimiter | None = None


def search_queries(company_key: str, names: Iterable[str]) -> list[str]:
    """Try the article title and recorded names, then their folded spellings."""
    out, seen = [], set()
    for c in [re.sub(r"\s*\([^)]*\)", "", company_key).strip()] + list(names):
        for q in (c, c.translate(FOLD)):
            if q and q.casefold() not in seen:
                seen.add(q.casefold())
                out.append(q)
    return out


def match_names(company_key: str, names: Iterable[str]) -> list[str]:
    """Include the company key without parenthetical qualifiers when matching."""
    return list(names) + [re.sub(r"\s*\([^)]*\)", "", company_key).strip()]


def name_tokens(text: str) -> set[str]:
    """Normalize Yahoo's plus signs and remove legal-form and linking words."""
    return set(de_membership.name_key(text.replace("+", "&")).split()) - _STOP


def name_matches(yahoo_name: str, names: Iterable[str]) -> bool:
    """Require every Yahoo token to prefix a token of one recorded company name."""
    yahoo = name_tokens(yahoo_name)
    return bool(yahoo) and any(
        all(any(token.startswith(part) for token in name_tokens(name)) for part in yahoo)
        for name in names
    )


def pick_listing(results: dict[str, list | None], queries: list[str], names: list[str], *,
                 company_key: str | None = None) -> dict | None:
    """Pick the first strict name match on the most preferred German exchange."""
    names = match_names(company_key, names) if company_key is not None else names
    matches = {}
    for query in queries:
        for quote in results.get(query) or []:
            exchange = quote.get("exchange")
            if quote.get("quoteType") != "EQUITY" or exchange not in config.SETTINGS.de_universe.exchanges:
                continue
            yahoo_name = next((name for name in (quote.get("longname"), quote.get("shortname"))
                               if isinstance(name, str) and name and name_matches(name, names)), None)
            if yahoo_name is None:
                continue
            if exchange not in matches:
                matches[exchange] = {"symbol": quote["symbol"], "exchange": exchange,
                                     "price_source": PRICE_SOURCE[exchange], "yahoo_name": yahoo_name.strip(),
                                     "matched_query": query}
    return next((matches[exchange] for exchange in config.SETTINGS.de_universe.exchanges
                 if exchange in matches), None)


def load_overrides(path: Path = OVERRIDES_PATH) -> pd.DataFrame:
    """Read manually evidenced symbol overrides, including a header-only file."""
    return pd.read_csv(path, usecols=["company_key", "symbol", "evidence"])


def resolve_symbols(companies: pd.DataFrame, results: dict[str, list | None],
                    overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resolve each company without dropping failures or hiding shared symbols."""
    manual = overrides.set_index("company_key")["symbol"].to_dict()
    rows = []
    for company in companies.itertuples(index=False):
        if company.company_key in manual:
            symbol = manual[company.company_key]
            exchange = "GER" if symbol.endswith(".DE") else "HAM" if symbol.endswith(".HM") else None
            listing = {"symbol": symbol, "exchange": exchange, "price_source": PRICE_SOURCE.get(exchange, "other"),
                       "resolved_by": "override", "yahoo_name": None, "matched_query": None}
        else:
            listing = pick_listing(results, search_queries(company.company_key, company.names), company.names,
                                   company_key=company.company_key)
            if listing is not None:
                listing["resolved_by"] = "search"
            else:
                listing = dict.fromkeys(_RESOLVED_COLS[1:])
                listing["resolved_by"] = "unresolved"
        rows.append({"company_key": company.company_key, **listing})
    resolved = pd.DataFrame(rows, columns=_RESOLVED_COLS)
    searched = resolved["resolved_by"].eq("search")
    shared = {symbol: sorted(group["company_key"].tolist())
              for symbol, group in resolved.dropna(subset=["symbol"]).groupby("symbol") if len(group) > 1}
    diag = {"companies": len(resolved), "override": int(resolved["resolved_by"].eq("override").sum()),
            "search_xetra": int((searched & resolved["exchange"].eq("GER")).sum()),
            "search_hamburg": int((searched & resolved["exchange"].eq("HAM")).sum()),
            "unresolved": sorted(resolved.loc[resolved["resolved_by"].eq("unresolved"), "company_key"].tolist()),
            "shared_symbols": shared}
    return resolved, diag


def companies_since(membership: pd.DataFrame, start) -> pd.DataFrame:
    """Collect distinct names only from intervals open or ending since start."""
    ends = pd.to_datetime(membership["end_date"])
    relevant = membership.loc[ends.isna() | ends.ge(pd.Timestamp(start))]
    rows = [(key, sorted(set(group["name"]))) for key, group in relevant.groupby("company_key")]
    return pd.DataFrame(rows, columns=["company_key", "names"])


def build_universe(membership: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    """Attach latest membership metadata and full interval bounds to each company."""
    membership = membership.copy()
    for column in ("start_date", "end_date"):
        membership[column] = pd.to_datetime(membership[column])
    groups = {key: group for key, group in membership.sort_values("start_date", kind="stable").groupby("company_key")}
    rows = []
    for listing in resolved.to_dict("records"):
        intervals = groups[listing["company_key"]]
        latest = intervals.iloc[-1]
        current = bool(intervals["end_date"].isna().any())
        rows.append({**listing, "name": latest["name"], "index_code": latest["index_code"],
                     "cap_bucket": config.MARKETS["de"].cap_bucket(latest["index_code"]),
                     "is_current_member": current, "first_date": intervals["start_date"].min(),
                     "last_date": pd.NaT if current else intervals["end_date"].max()})
    return pd.DataFrame(rows, columns=UNIVERSE_COLS)


def search_yahoo(query: str) -> list[dict] | None:
    """Fetch only the Yahoo quote fields needed for German symbol resolution."""
    de_market._require_de()
    global _search_limiter
    try:
        if _search_limiter is None:
            _search_limiter = store.RateLimiter(config.SETTINGS.de_universe.search_rps)
        _search_limiter.wait()
        quotes = yf.Search(query, max_results=config.SETTINGS.de_universe.search_max_results, news_count=0).quotes
        return [{field: quote.get(field) for field in _QUOTE_FIELDS} for quote in quotes]
    except Exception as exc:  # yfinance can fail for throttling, transport or parsing errors
        log.warning("Yahoo search failed for %s: %s", query, exc)
        return None


def cached_search(queries: Iterable[str], refresh: bool = False) -> dict[str, list | None]:
    """Cache successful queries (including empty results), retrying failed fetches."""
    de_market._require_de()
    path = config.CACHE_DIR / "yahoo_search_de.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cache = {query: quotes for query, quotes in cache.items() if quotes is not None}
    results = {}
    for query in dict.fromkeys(queries):
        quotes = search_yahoo(query) if refresh or query not in cache else cache[query]
        results[query] = quotes
        if quotes is not None:
            cache[query] = quotes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def update_de_universe(refresh_search: bool = False, load_prices: bool = True) -> dict:
    """Write the German universe and incrementally load every resolved symbol."""
    de_market._require_de()
    membership = store.read_table(de_membership.TABLE)
    if membership.empty:
        raise RuntimeError("run update_de_membership first")
    companies = companies_since(membership, config.START_DATE)
    queries = [query for company in companies.itertuples(index=False)
               for query in search_queries(company.company_key, company.names)]
    results = cached_search(queries, refresh=refresh_search)
    resolved, diag = resolve_symbols(companies, results, load_overrides())
    universe = build_universe(membership, resolved)
    store.replace_table(TABLE, universe)
    price_report = prices.update_prices(sorted(set(resolved["symbol"].dropna()))) if load_prices else None
    counts = {key: len(value) if isinstance(value, list) else value
              for key, value in diag.items() if isinstance(value, (int, list))}
    return {"table": TABLE, "rows": len(universe), **counts, "unresolved": diag["unresolved"], "prices": price_report}
