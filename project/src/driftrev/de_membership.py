"""R7.2 · German point-in-time index membership from German Wikipedia REST HTML.

Sources are the DAX, MDAX and SDAX current tables, the DAX change table and
MDAX/Zusammensetzungshistorie. MDAX's SDAX markers supply partial SDAX history.
Company texts and article targets change over time: union-find joins normalized
names to redirect-resolved article titles, with evidence-backed manual aliases
from resources/de_aliases.csv. Components prefer the smallest article title as
their key; names without a linked identity retain a name key and are reported,
never guessed. Symbol/ISIN resolution and price loading belong to R7.2b.
Evidence-backed event-level identities separate companies that share historical cell names.

Replay additions before removals on each date, opening and closing [start, end)
intervals. A date with additions and removals for one identity is a continuation,
preserving an open interval, opening one before coverage if unrecorded, or
reopening at that date after a closed interval.
An initial removal implies membership from 1900-01-01 (SDAX starts
are reported as unknown). Duplicate events are counted; zero-length intervals
are dropped, while short spin-off intervals are retained. Current-table dates
seed missing open intervals without overlapping recorded history; conflicting
dates and current members with contradictory events are reported. Open intervals
for non-current companies are dropped and reported as orphans. Display names
come from the current table, or otherwise the most recent event.

Coverage starts at each index's first change date. SDAX companies that entered
and left without touching MDAX are invisible; its history has survivorship bias.
Delisted names have no prices. Unresolved names and missing rows remain visible
in diagnostics rather than being assigned speculative identities or intervals.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
from bs4 import BeautifulSoup, Tag

from . import config, store
from .de_market import _require_de

log = logging.getLogger(__name__)
TABLE = "membership_history"
INDEXES = ("dax", "mdax", "sdax")
FOREVER = pd.Timestamp("1900-01-01")
ALIASES_PATH = Path(__file__).parent / "resources" / "de_aliases.csv"
EVENT_IDENTITIES_PATH = Path(__file__).parent / "resources" / "de_event_identities.csv"
CHANGE_COLS = ["date", "removed_name", "removed_link", "removed_other",
               "added_name", "added_link", "added_other", "index_code"]
CURRENT_COLS = ["index_code", "name", "link", "since"]
MEMBERSHIP_COLS = ["company_key", "name", "index_code", "start_date", "end_date"]

_FOOTNOTE = re.compile(r"\[\s*\d+\s*\]")
_MONTHS = {
    name.casefold(): month
    for month, names in enumerate((
        "Januar Jänner Jan", "Februar Feb", "März Mär Mrz", "April Apr", "Mai",
        "Juni Jun", "Juli Jul", "August Aug", "September Sept Sep", "Oktober Okt",
        "November Nov", "Dezember Dez",
    ), start=1)
    for name in names.split()
}
_DATE = re.compile(r"\b(\d{1,2})\.\s*(?:(\d{1,2})\.(\d{4})|([^\W\d_]+)\.?\s+(\d{4}))\b")
_LEGAL_FORM = re.compile(
    r"(?<!\w)(?:AG\s*&\s*Co\.\s*KGaA|SE\s*&\s*Co\.\s*KGaA|GmbH\s*&\s*Co\.\s*KGaA"
    r"|KGaA|AG|SE|S\.A\.|N\.V\.|GmbH|plc|Inc\.|Ltd\.)(?!\w)", re.IGNORECASE,
)


def parse_german_date(text: str | None) -> pd.Timestamp:
    """Read the first numeric or German named-month date, ignoring footnotes."""
    for match in _DATE.finditer(_FOOTNOTE.sub("", text or "")):
        day, numeric_month, numeric_year, month_name, named_year = match.groups()
        month = int(numeric_month) if numeric_month else _MONTHS.get(month_name.casefold())
        if month is not None:
            try:
                return pd.Timestamp(year=int(numeric_year or named_year), month=month, day=int(day))
            except ValueError:
                return pd.NaT
    return pd.NaT


def parse_cell(td: Tag) -> tuple[str | None, str | None, str | None]:
    """Extract a company name, article target and optional index migration marker."""
    cell = deepcopy(td)
    marker = cell.find(["sub", "sup"])
    other = marker.get_text(" ", strip=True).upper() or None if marker else None
    for tag in cell.find_all(["sub", "sup"]):
        tag.decompose()
    link = None
    for anchor in cell.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith("./") and not href.startswith("./Datei:"):
            link = unquote(href[2:]).split("#", 1)[0].replace("_", " ")
            break
    text = " ".join(_FOOTNOTE.sub("", cell.get_text(" ", strip=True)).split())
    return text or None, link, other


def parse_current(html: str, index_code: str) -> pd.DataFrame:
    """Find the first current-constituent table by its name and admission headers."""
    rows = []
    for table in BeautifulSoup(html, "lxml").find_all("table"):
        first = table.find("tr")
        headers = [cell.get_text(" ", strip=True).casefold() for cell in first.find_all("th")] if first else []
        i_name = next((i for i, text in enumerate(headers) if text.startswith("name")), None)
        i_since = next((i for i, text in enumerate(headers) if "seit" in text or "aufnahme" in text), None)
        if i_name is None or i_since is None:
            continue
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td", recursive=False)
            if cells:
                name, link, _ = parse_cell(cells[i_name])
                since = parse_german_date(cells[i_since].get_text(" ", strip=True))
                rows.append((index_code, name, link, since))
        break
    return pd.DataFrame(rows, columns=CURRENT_COLS)


def _expand_rows(rows: Iterable[Tag]) -> Iterable[dict[int, tuple[Tag, bool]]]:
    """Yield table-grid slots as (source cell, is_origin_row), expanding spans."""
    carried: dict[int, tuple[Tag, int]] = {}
    for tr in rows:
        grid = {column: (cell, False) for column, (cell, _) in carried.items()}
        carried = {column: (cell, remaining - 1) for column, (cell, remaining) in carried.items()
                   if remaining > 1}
        column = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while column in grid:
                column += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for offset in range(colspan):
                grid[column + offset] = (cell, True)
                if rowspan > 1:
                    carried[column + offset] = (cell, rowspan - 1)
            column += colspan
        yield grid


def parse_changes(html: str, index_code: str) -> pd.DataFrame:
    """Parse span-expanded changes, emitting company cells only on their origin row."""
    rows = []
    for table in BeautifulSoup(html, "lxml").find_all("table"):
        first = table.find("tr")
        headers = [cell.get_text(" ", strip=True).casefold() for cell in first.find_all(["th", "td"])] if first else []
        i_removed = next((i for i, text in enumerate(headers) if "ausgeschiedene" in text), None)
        i_added = next((i for i, text in enumerate(headers) if "aufgenommene" in text), None)
        if i_removed is None or i_added is None:
            continue
        i_date = next(i for i, text in enumerate(headers) if "datum" in text)
        for grid in _expand_rows(table.find_all("tr")[1:]):
            date_slot = grid.get(i_date)
            date = parse_german_date(date_slot[0].get_text(" ", strip=True) if date_slot else None)
            if pd.isna(date):
                continue
            sides = []
            for column in (i_removed, i_added):
                slot = grid.get(column)
                value = parse_cell(slot[0]) if slot and slot[1] else (None, None, None)
                sides.append(value if value[0] else (None, None, None))
            removed, added = sides
            if removed[0] is not None or added[0] is not None:
                rows.append((date, *removed, *added, index_code))
        break
    return pd.DataFrame(rows, columns=CHANGE_COLS)


def derive_sdax_changes(mdax_changes: pd.DataFrame) -> pd.DataFrame:
    """Invert MDAX's SDAX migration markers into the known SDAX events."""
    rows = []
    for row in mdax_changes.itertuples(index=False):
        if row.removed_other == "SDAX":
            rows.append((row.date, None, None, None, row.removed_name, row.removed_link, None, "sdax"))
        if row.added_other == "SDAX":
            rows.append((row.date, row.added_name, row.added_link, None, None, None, None, "sdax"))
    return pd.DataFrame(rows, columns=CHANGE_COLS)


def name_key(text: str) -> str:
    """Normalize company text without inferring undocumented name changes."""
    text = re.sub(r"\([^)]*\)", "", _FOOTNOTE.sub("", text))
    text = _LEGAL_FORM.sub("", text)
    text = text.translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                                       "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def load_redirects(obj: dict) -> dict[str, str]:
    """Merge API normalization and redirect maps, preferring redirect entries."""
    return {**obj.get("normalized", {}), **obj.get("redirects", {})}


def canonical_title(title: str, redirects: dict[str, str]) -> str:
    """Follow at most five title normalization/redirect hops."""
    for _ in range(5):
        if title not in redirects:
            break
        title = redirects[title]
    return title


def load_aliases(path: Path = ALIASES_PATH) -> pd.DataFrame:
    """Read the manually evidenced identity aliases."""
    return pd.read_csv(path, usecols=["alias_text", "canonical_title", "evidence"])


def load_event_identities(path: Path = EVENT_IDENTITIES_PATH) -> pd.DataFrame:
    """Read the manually evidenced identities for individual change events."""
    identities = pd.read_csv(path, usecols=["index_code", "date", "side", "text", "company_key", "evidence"])
    identities["date"] = pd.to_datetime(identities["date"])
    return identities


def resolve_keys(cells: Iterable[tuple[str | None, str | None]], redirects: dict[str, str],
                 aliases: pd.DataFrame) -> dict[tuple[str | None, str | None], str]:
    """Join name/article nodes and assign each component a deterministic key."""
    cells = list(cells)
    parents: dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != node:
            node, parents[node] = parents[node], root
        return root

    def union(left: str, right: str) -> None:
        parents[find(left)] = find(right)

    for text, link in cells:
        if text is None:
            continue
        node = "N:" + name_key(text)
        find(node)
        if link:
            union(node, "L:" + canonical_title(link, redirects))
    for row in aliases.itertuples(index=False):
        union("N:" + name_key(row.alias_text), "L:" + canonical_title(row.canonical_title, redirects))
    representatives: dict[str, str] = {}
    for node in parents:
        root = find(node)
        representatives[root] = min(node, representatives.get(root, node))  # L: sorts before N:
    result = {(text, link): representatives[find("N:" + name_key(text))][2:]
              for text, link in cells if text is not None}
    unresolved = sorted({key for cell, key in result.items()
                         if representatives[find("N:" + name_key(cell[0]))].startswith("N:")})
    if unresolved:
        log.warning("Names without a linked identity (retained as name keys): %s", unresolved)
    return result


def build_de_membership(changes: pd.DataFrame, current: pd.DataFrame,
                        keys: dict[tuple[str | None, str | None], str],
                        event_identities: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Replay per-company index events and reconcile with current constituents."""
    overrides = {}
    if event_identities is not None:
        for row in event_identities.itertuples(index=False):
            count = 0
            if row.index_code in INDEXES and row.side in ("added", "removed"):
                names = changes.loc[changes["index_code"].eq(row.index_code)
                                    & changes["date"].eq(row.date), f"{row.side}_name"]
                count = names.dropna().map(name_key).eq(name_key(row.text)).sum()
            if count != 1:
                raise ValueError(f"Event identity row {row} matched {count} events; expected exactly one")
            overrides[row.index_code, row.date, row.side, name_key(row.text)] = row.company_key
    out = []
    diag = {"rows": 0, "double_add": 0, "double_remove": 0, "current_without_events": 0,
            "start_unknown": [], "since_mismatch": [], "inconsistent_current": [], "orphans": [],
            "same_day_swap": [], "event_identity": [],
            "coverage_start": {}}
    tolerance = pd.Timedelta(days=config.SETTINGS.de_membership.since_tolerance_days)
    for index_code in INDEXES:
        history = changes.loc[changes["index_code"] == index_code]
        if not history.empty:
            diag["coverage_start"][index_code] = history["date"].min()
        events: dict[str, list[tuple[pd.Timestamp, int, str]]] = {}
        for row in history.itertuples(index=False):
            for name, link, kind in ((row.added_name, row.added_link, 1),
                                     (row.removed_name, row.removed_link, -1)):
                if name is not None:
                    side = "added" if kind == 1 else "removed"
                    key = overrides.get((index_code, row.date, side, name_key(name)))
                    if key is None:
                        key = keys[name, link]
                    else:
                        diag["event_identity"].append((index_code, row.date, side, key))
                    events.setdefault(key, []).append((row.date, kind, name))
        now = {keys[row.name, row.link]: (row.name, row.since)
               for row in current.loc[current["index_code"] == index_code].itertuples(index=False)}
        for key in sorted(set(events) | set(now)):
            evs = sorted(events.get(key, []), key=lambda event: (event[0], -event[1]))
            kinds_by_date: dict[pd.Timestamp, set[int]] = {}
            for date, kind, _ in evs:
                kinds_by_date.setdefault(date, set()).add(kind)
            swap_dates = {date for date, kinds in kinds_by_date.items() if len(kinds) == 2}
            intervals = []
            open_start = None
            recorded = False
            last_end = None
            identity = (index_code, key)
            for date, kind, _ in evs:
                if date in swap_dates:
                    diag["same_day_swap"].append((*identity, date))
                    if open_start is None:
                        open_start = date if recorded else FOREVER
                        if not recorded:
                            recorded = True
                            if index_code == "sdax":
                                diag["start_unknown"].append(identity)
                    swap_dates.remove(date)
                elif len(kinds_by_date[date]) == 2:
                    continue
                elif kind == 1:
                    if open_start is None:
                        open_start = date
                    else:
                        diag["double_add"] += 1
                elif open_start is not None:
                    intervals.append((open_start, date))
                    recorded, last_end, open_start = True, date, None
                elif not recorded:
                    intervals.append((FOREVER, date))
                    recorded, last_end = True, date
                    if index_code == "sdax":
                        diag["start_unknown"].append(identity)
                else:
                    diag["double_remove"] += 1
            if key in now:
                name, since = now[key]
                if open_start is not None:
                    if not pd.isna(since) and abs(open_start - since) > tolerance:
                        diag["since_mismatch"].append(identity)
                else:
                    open_start = since if not pd.isna(since) and (last_end is None or since >= last_end) else last_end
                    if open_start is None:
                        open_start = FOREVER
                    if evs:
                        diag["inconsistent_current"].append(identity)
                    else:
                        diag["current_without_events"] += 1
                intervals.append((open_start, pd.NaT))
            else:
                name = evs[-1][2]
                if open_start is not None:
                    diag["orphans"].append(identity)
            out.extend((key, name, index_code, start, end) for start, end in intervals
                       if pd.isna(end) or start != end)
    membership = pd.DataFrame(out, columns=MEMBERSHIP_COLS)
    for column in ("start_date", "end_date"):
        membership[column] = pd.to_datetime(membership[column])
    membership = membership.sort_values(["index_code", "company_key", "start_date"]).reset_index(drop=True)
    diag["rows"] = len(membership)
    return membership, diag


def membership_on(membership: pd.DataFrame, date) -> pd.DataFrame:
    """Return the companies and German cap buckets active on a date."""
    date = pd.Timestamp(date)
    active = membership.loc[(membership["start_date"] <= date)
                            & (membership["end_date"].isna() | (date < membership["end_date"])),
                            ["company_key", "index_code"]].copy()
    active["cap_bucket"] = active["index_code"].map(config.MARKETS["de"].cap_bucket)
    return active.reset_index(drop=True)


def fetch_pages(refresh: bool = False) -> dict[str, str]:
    """Fetch configured REST HTML pages, reusing fresh UTF-8 disk caches."""
    _require_de()
    limiter = store.RateLimiter(config.SETTINGS.de_membership.wiki_rps)
    sess = store.session()
    cache = config.CACHE_DIR / "wikipedia_de"
    cache.mkdir(parents=True, exist_ok=True)
    pages = {}
    for key, url in config.DE_WIKI_PAGES.items():
        path = cache / f"{key}.html"
        fresh = path.exists() and time.time() - path.stat().st_mtime < config.SETTINGS.de_membership.cache_hours * 3600
        if fresh and not refresh:
            pages[key] = path.read_text(encoding="utf-8")
        else:
            resp = store.get(sess, url, limiter)
            if resp.status_code != 200:
                raise RuntimeError(f"Wikipedia HTTP {resp.status_code} for {url}")
            pages[key] = resp.text
            path.write_text(resp.text, encoding="utf-8")
    return pages


def fetch_redirects(titles: Iterable[str]) -> dict:
    """Resolve unique sorted article targets in API-sized, rate-limited batches."""
    _require_de()
    titles = sorted(set(titles))
    batch_size = config.SETTINGS.de_membership.redirect_batch
    limiter = store.RateLimiter(config.SETTINGS.de_membership.wiki_rps)
    sess = store.session()
    result = {"normalized": {}, "redirects": {}}
    for start in range(0, len(titles), batch_size):
        batch = titles[start:start + batch_size]
        url = (f"{config.DE_WIKI_API}?action=query&redirects=1&format=json&formatversion=2&titles="
               + "|".join(quote(title, safe="") for title in batch))
        resp = store.get(sess, url, limiter)
        if resp.status_code != 200:
            raise RuntimeError(f"Wikipedia HTTP {resp.status_code} for {url}")
        query = resp.json().get("query", {})
        for kind in result:
            result[kind].update({row["from"]: row["to"] for row in query.get(kind, [])})
    return result


def update_de_membership(refresh: bool = False) -> dict:
    """Reconstruct membership and replace only the active German warehouse table. Fetches the Wikipedia pages and the
    redirect API at most every `FetchCadence.membership_refresh_days` unless `refresh`."""
    _require_de()
    if store.table_exists(TABLE) and not store.fetch_due(
            "de_membership", config.SETTINGS.fetch_cadence.membership_refresh_days, refresh):
        return {"skipped": True, "reason": f"fetched {store.fetch_age_days('de_membership')}d ago", "table": TABLE}
    pages = fetch_pages(refresh=refresh)
    current = pd.concat([parse_current(pages[index], index) for index in INDEXES], ignore_index=True)
    dax = parse_changes(pages["dax"], "dax")
    mdax = parse_changes(pages["mdax_history"], "mdax")
    changes = pd.concat([dax, mdax, derive_sdax_changes(mdax)], ignore_index=True)
    cells = list(current[["name", "link"]].itertuples(index=False, name=None))
    for side in ("removed", "added"):
        cells.extend(changes[[f"{side}_name", f"{side}_link"]].itertuples(index=False, name=None))
    redirects = load_redirects(fetch_redirects(link for _, link in cells if link))
    keys = resolve_keys(cells, redirects, load_aliases())
    membership, diag = build_de_membership(changes, current, keys, event_identities=load_event_identities())
    store.replace_table(TABLE, membership)
    store.record_fetch("de_membership")
    counts = {key: len(value) if isinstance(value, list) else value
              for key, value in diag.items() if isinstance(value, (int, list))}
    return {"table": TABLE, **counts}
