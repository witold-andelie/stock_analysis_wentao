"""R4 · Classifying Filing Text (OPM: SD2.1).

A deterministic, explainable text layer on top of the 8-K item codes used by
`features.edgar_features`: no LLM, no paid API. For every filing accepted inside an event
window (previous close 16:00 ET .. next morning 09:00 ET) we

  1. pick the main press-release exhibit from the filing index (EX-99.1, else EX-99, else the
     primary document), convert it to text and cache it gzip-compressed,
  2. flag lexicon phrases (guidance cut / raise, going concern, priced offering, restatement,
     clinical failure ...) with a negation guard and a matched snippet for explainability,
  3. score the tone with the Loughran-McDonald (LM) dictionary,
  4. aggregate per event (`attach_filing_text`) into point-in-time `ft_*` columns and a refined
     category `edgar_category_text`.

Storage decision: the warehouse table `filing_text` keeps flags + tone + short snippets (small,
queryable); the capped plain text is also kept as `cache/filing_text/<accession>.txt.gz` (~5-8 kB
per filing) so the lexicon can be refined offline without re-downloading (`refit`). Set
`TextConfig.keep_raw=False` to skip it.

CLI:  python -m driftrev.filing_text ingest [--max N] [--no-earnings]
      python -m driftrev.filing_text refit          # re-run the lexicon on the cached texts
      python -m driftrev.filing_text study
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlsplit

import numpy as np
import pandas as pd

from . import config, store

log = logging.getLogger(__name__)

TABLE = "filing_text"
SEC_HOST = "www.sec.gov"


# --------------------------------------------------------------------------------------
# Settings (kept here because config.py is owned by another module; move to config.py at integration)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TextConfig:
    crash_families: tuple = ("shock_crash", "earnings_crash")
    crash_forms_prefix: tuple = ("8-K", "424B", "NT 10-")       # all years
    earnings_families: tuple = ("earnings_up", "earnings_down")
    earnings_from: str = "2022-01-01"                          # validation + test period
    earnings_item: str = "2.02"
    max_filings: int = 5000            # budget per run (resumable)
    max_chars: int = 60_000            # text kept / analysed per filing
    max_bytes: int = 4_000_000         # never read more than this from one document
    workers: int = 6
    keep_raw: bool = True
    snippet_chars: int = 110           # context each side of a match
    # 8-K items whose substance lives in the primary document, not in an EX-99 press release
    primary_items: tuple = ("1.01", "1.02", "1.03", "2.03", "2.04", "2.05", "2.06", "3.01", "4.01", "4.02", "5.02")


TEXT = TextConfig()
CACHE = config.CACHE_DIR / "filing_text"
_LIMITER = store.RateLimiter(config.SEC_MAX_RPS)

FLAG_NAMES = (
    "guidance_cut", "guidance_raise", "guidance_reaffirm", "going_concern", "offering_priced",
    "restatement", "material_weakness", "investigation", "clinical_failure", "exec_departure",
    "covenant", "impairment", "layoffs", "bankruptcy", "beat", "miss",
)
TONE_COLS = ("lm_negative_pct", "lm_positive_pct", "lm_uncertainty_pct", "lm_net_tone")
OFFER_COLS = ("offering_price", "last_price", "offering_discount")
FINAL_STATUS = ("ok", "no_index", "no_doc")  # 'error' rows are retried


# --------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------
_OBJ = r"(?:(?:its|our|the|their|previously|issued|provided|prior|full[- ]year|fiscal|year|annual|financial|\d{4}|revenue|earnings|eps|sales|and|or|net|income|adjusted|ebitda|per|share|quarterly|first|second|third|fourth|quarter|[-,])\s+){0,8}"
_GUIDE = r"(?:outlook|guidance|forecasts?|projections?|financial targets|expectations)"
_GUIDE_STRICT = r"(?:outlook|guidance|forecasts?|financial targets)"
_DOWN = r"(?:lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|cut(?:s|ting)?|trim(?:s|med|ming)?|slash(?:es|ed|ing)?|decreas(?:e|es|ed|ing)|revis(?:e|es|ed|ing)\s+(?:\w+\s+){0,3}?downward|downward(?:ly)?\s+revis\w+)"
_UP = r"(?:rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|boost(?:s|ed|ing)?|lift(?:s|ed|ing)?|rais\w+|upward(?:ly)?\s+revis\w+|revis(?:e|es|ed|ing)\s+(?:\w+\s+){0,3}?upward)"

# (flag, regex). All patterns run on lower-cased, whitespace-normalised text.
_PATTERNS: dict[str, list[str]] = {
    "guidance_cut": [
        rf"\b{_DOWN}\s+{_OBJ}{_GUIDE}\b",
        rf"\b{_GUIDE_STRICT}\s+(?:has been|was|is being|were)\s+(?:lowered|reduced|cut|revised downward)",
        r"\bnow (?:expects?|anticipates?|projects?|forecasts?|sees|estimates)\b[^.]{0,220}?\b(?:below|lower than|less than|down from|reduced from|decline of|decrease)\b",
        rf"\bwithdr(?:aw|aws|awn|awing|ew)\s+(?:its |our |the |previously |issued |provided |financial |full[- ]year |all )*{_GUIDE}\b",
        rf"\b(?:suspend(?:s|ed|ing)?|discontinu(?:e|es|ed|ing))\s+(?:its |our |the |financial |full[- ]year )*{_GUIDE_STRICT}\b",
        r"\bno longer (?:expects?|anticipates?|provides?|providing|gives?|giving)\b[^.]{0,80}\b(?:guidance|outlook|forecast)",
        r"\bbelow (?:the )?(?:low[- ]end|bottom|low end) of (?:its |our |the |previous |prior )*(?:guidance|outlook|range)",
    ],
    "guidance_raise": [
        rf"\b{_UP}\s+{_OBJ}{_GUIDE_STRICT}\b",
        rf"\b{_GUIDE_STRICT}\s+(?:has been|was|is being|were)\s+(?:raised|increased|lifted|boosted)",
        r"\bnow (?:expects?|anticipates?|projects?|forecasts?|sees|estimates)\b[^.]{0,220}?\b(?:above|higher than|greater than|up from|exceed\w*|increase of)\b",
        r"\babove (?:the )?(?:high[- ]end|top|high end) of (?:its |our |the |previous |prior )*(?:guidance|outlook|range)",
    ],
    "guidance_reaffirm": [
        rf"\b(?:reaffirm|reiterat|re-affirm|maintain|confirm)\w*\s+(?:its |our |the |previously |issued |provided |full[- ]year |fiscal |annual |financial |\d{{4}} )*{_GUIDE_STRICT}\b",
        r"\bguidance (?:remains|is) (?:unchanged|in line)\b",
    ],
    "going_concern": [
        r"\bsubstantial doubt\b[^.]{0,200}?\bgoing concern\b",
        r"\bgoing concern\b[^.]{0,120}?\bsubstantial doubt\b",
        r"\bdoubt (?:about|regarding|as to) (?:its|our|the company'?s) ability to continue as a going concern\b",
    ],
    "offering_priced": [
        r"\bpric(?:es|ed|ing)\s+(?:of\s+)?(?:its |an? |the |upsized |underwritten |proposed |previously announced |\$[\d.,]+ ?(?:million|billion|m|b)? )*(?:public offering|offering|registered direct|private placement|follow-on|common stock offering|equity offering|convertible)",
        r"\bat[- ]the[- ]market (?:offering|equity|program|sales agreement)",
        r"\bregistered direct (?:offering|placement)",
        r"\b(?:we|the company) (?:are|is) offering [\d,.]+ ?(?:million )?(?:shares|units|common)",
        r"\bunderwriting agreement\b[^.]{0,80}\bshares\b",
    ],
    "restatement": [
        r"\brestat(?:e|es|ed|ing|ement|ements)\b",
        r"\bnon-?reliance\b",
        r"\bshould no longer be relied upon\b",
    ],
    "material_weakness": [r"\bmaterial weakness(?:es)?\b"],
    "investigation": [
        r"\bsubpoenas?\b",
        r"\bcivil investigative demand\b",
        r"\bwells notice\b",
        r"\bformal order of (?:private )?investigation\b",
        r"\b(?:sec|doj|department of justice|securities and exchange commission|ftc|fda|attorney general)\b[^.]{0,60}\b(?:investigation|inquiry)\b",
        r"\binvestigation (?:by|from) the (?:sec|securities and exchange commission|department of justice|doj)\b",
        r"\b(?:securities )?class action\b",
        r"\bshort[- ]seller report\b",
    ],
    "clinical_failure": [
        r"\b(?:did not|failed to|has not|not) (?:meet|achieve|reach|demonstrate|show)\b[^.]{0,80}\b(?:primary endpoint|primary end point|statistical significance|co-primary)",
        r"\bcomplete response letter\b",
        r"\breceived a crl\b",
        r"\bclinical hold\b",
        r"\brefusal to file\b|\brefuse[sd]? to file\b",
        r"\bdiscontinu\w+\s+(?:the |its |our |all |further |clinical |development of |enrollment in |dosing in )*(?:trial|study|program|development|phase \w+)",
        r"\bfailed (?:the |its )?(?:phase \w+ )?(?:trial|study)\b",
    ],
    "exec_departure": [
        r"\b(?:chief executive officer|ceo|chief financial officer|cfo|chief operating officer|coo|president)\b[^.]{0,90}\b(?:resign\w*|depart\w*|step(?:s|ped|ping)? (?:down|aside)|terminat\w+|retir\w+|removed|ousted)\b",
        r"\b(?:resign\w*|terminat\w+|removal|departure|ousted)\b[^.]{0,60}\bas (?:the )?(?:company'?s )?(?:chief executive officer|ceo|chief financial officer|cfo|chief operating officer|president)\b",
    ],
    "covenant": [
        r"\bwaiver\b[^.]{0,80}\bcovenants?\b",
        r"\bcovenants?\b[^.]{0,40}\b(?:breach\w*|violation\w*|default\w*|non-?compliance)\b",
        r"\bnot in compliance with\b[^.]{0,80}\bcovenants?\b",
        r"\bevents? of default\b",
        r"\bforbearance agreement\b",
        r"\bcross-?default\b",
        r"\bin default (?:under|of|on)\b",
    ],
    "impairment": [
        r"\b(?:goodwill |intangible |asset |long-lived asset |non-?cash )?impairments?(?: charges?| of| losses| expense)",
        r"\bwrite-?downs?\b|\bwrite-?offs?\b|\bwritten (?:down|off)\b",
    ],
    "layoffs": [
        r"\breduction in (?:force|workforce|headcount)\b|\bworkforce reduction\b|\blayoffs?\b|\blay(?:ing)? off\b",
        r"\brestructuring (?:plan|charges?|program|actions?|initiative)\b",
        r"\b(?:eliminat\w+|reduc\w+)\s+(?:approximately |about )?\d[\d,]*(?: ?%| percent)? of (?:its |our |the )?(?:workforce|employees|positions|headcount)",
        r"\bplant closures?\b|\bstore closures?\b|\bfacility closures?\b",
    ],
    "bankruptcy": [r"\bchapter 11\b", r"\bvoluntary petitions?\b[^.]{0,80}\b(?:bankruptcy|chapter)\b", r"\bchapter 7\b"],
    "beat": [
        r"\brecord (?:\w+[- ]){0,3}?(?:revenues?|net sales|sales|earnings|results|bookings|eps|net income|profit|ebitda|operating income|free cash flow)\b",
        r"\b(?:exceed(?:ed|s|ing)?|surpass(?:ed|es|ing)?|beat|beats|topped)\s+(?:analyst |consensus |market |street |our |prior |the high end of )*(?:expectations|estimates|consensus|guidance)\b",
        r"\b(?:above|ahead of) (?:analyst |consensus |market |street )*(?:expectations|estimates|consensus)\b",
        r"\bstrongest (?:quarter|year|results)\b",
    ],
    "miss": [
        r"\bbelow (?:analyst |consensus |market |street |our |prior |previous |management )*(?:expectations|estimates|consensus|forecasts?)\b",
        r"\bfell short of\b|\bfall(?:ing)? short of\b|\bmissed (?:analyst |consensus |market )*(?:expectations|estimates|consensus)",
        r"\bdisappointing\b|\bshortfall\b",
        r"\b(?:weaker|lower|softer|worse) than (?:expected|anticipated|forecast)\b",
    ],
}
_COMPILED = {k: [re.compile(p) for p in v] for k, v in _PATTERNS.items()}

# a match is dropped when one of these appears shortly before it (same sentence)
_NEGATORS = re.compile(r"\b(?:no|not|never|neither|nor|without|cannot|n't|unlikely|hasn't|haven't|doesn't|don't|didn't|alleviat\w+|remediat\w+|resolved|eliminat\w+ the)\b(?:\W+\w+){0,5}\W*$")
_NEG_SKIP = {"layoffs", "beat", "miss", "guidance_reaffirm", "offering_priced", "bankruptcy", "investigation", "exec_departure"}
# flags where the negator window must only look inside the sentence, never at a previous one
_SENT_BREAK = re.compile(r"[.;!?]")

_PRICE = re.compile(r"(?:public offering price|offering price|purchase price|price to the public|exercise price|price of)\s+(?:of\s+|is\s+|was\s+|will be\s+)?(?:us)?\$\s?(\d{1,4}(?:\.\d{1,4})?)\s+per\s+(?:share|unit|security|ads)")
_LAST = re.compile(r"last (?:reported )?(?:sale|sales|closing|trading)? ?price[^$]{0,120}?\$\s?(\d{1,4}(?:\.\d{1,4})?)")
_LAST2 = re.compile(r"closing (?:sale )?price[^$]{0,120}?\$\s?(\d{1,4}(?:\.\d{1,4})?)")


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - 60):start]
    brk = [m.end() for m in _SENT_BREAK.finditer(window)]
    if brk:
        window = window[brk[-1]:]
    return bool(_NEGATORS.search(window))


def find_flags(text: str, snippet_chars: int = 110) -> tuple[dict[str, int], dict[str, str]]:
    """Return ({flag: 0/1}, {flag: snippet}) for already-normalised text."""
    flags, snips = {}, {}
    for name in FLAG_NAMES:
        hit = None
        for rx in _COMPILED[name]:
            for m in rx.finditer(text):
                if name not in _NEG_SKIP and _negated(text, m.start()):
                    continue
                hit = m
                break
            if hit:
                break
        flags[name] = int(hit is not None)
        if hit:
            snips[name] = text[max(0, hit.start() - snippet_chars):hit.end() + snippet_chars]
    # a pure "reaffirm" statement next to a cut/raise on the same object is not neutral: keep all three, aggregation decides
    return flags, snips


def parse_offering_discount(text: str) -> tuple[float, float, float]:
    """(offering price, last close, discount) or NaN. Discount = 1 - offering/last (positive = priced below the last close)."""
    p, last = _PRICE.search(text), _LAST.search(text) or _LAST2.search(text)
    off = float(p.group(1)) if p else np.nan
    lst = float(last.group(1)) if last else np.nan
    disc = 1 - off / lst if off == off and lst == lst and lst > 0 and 0 < off / lst < 3 else np.nan
    return off, lst, disc


# --------------------------------------------------------------------------------------
# Tone: Loughran-McDonald master dictionary
# --------------------------------------------------------------------------------------
LM_URL = "https://drive.google.com/uc?export=download&id=17CmUZM9hGUdGYjCXcjQLyybjTrcjrhik"  # mirror of the SRAF master CSV
LM_SOURCE_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"  # licence: free for academic / research use, cite LM (2011)

# Curated fallback (subset of LM categories, used only when the dictionary cannot be downloaded)
_CURATED = {
    "negative": """loss losses decline declined declines decrease decreased decreasing weak weaker weakness adverse adversely
        impairment impaired impairments default defaults defaulted litigation lawsuit lawsuits terminate terminated termination
        restate restated restatement deficiency deficiencies delay delayed delays shortfall shortfalls fail failed failure failures
        negative negatively harm harmed damage damages deteriorate deteriorated deterioration difficult difficulties difficulty
        disappoint disappointing disappointed investigation violation violations breach breaches bankruptcy insolvency
        unfavorable underperform underperformed volatile volatility writedown writeoff closure closures downturn downgrade
        challenging challenges concern concerns doubt doubts penalty penalties recall recalls slowdown slump sued suspend
        suspended suspension withdraw withdrawn withdrawal worsen worsened worse worst warn warned warning dilution dilutive""".split(),
    "positive": """gain gains improve improved improvement improvements improving strong stronger strength strengthen best better
        beneficial benefit benefits efficient effective excellent exceed exceeded exceeding favorable good great growth
        outperform outperformed profitable profitability progress record success successful successfully surpass surpassed
        achieve achieved achievement achievements opportunity opportunities leading leadership innovative reliable
        enhance enhanced enable enabled momentum robust solid upturn upgrade""".split(),
    "uncertainty": """approximate approximately believe believes could depend depends fluctuate fluctuations may might possible
        possibly predict risk risks risky uncertain uncertainties uncertainty unknown unpredictable unforeseen volatile
        volatility likelihood doubt anticipate anticipated assume assumed assumption contingent expose exposed exposure
        suggest probable perhaps variable vary varying""".split(),
}
_LM_CACHE: dict | None = None
_WORD = re.compile(r"[a-z]+")


def load_lm(download: bool = True) -> dict:
    """{'negative': set, 'positive': set, 'uncertainty': set, 'source': str}. Cached in DATA_DIR/cache."""
    global _LM_CACHE
    if _LM_CACHE is not None:
        return _LM_CACHE
    path = config.CACHE_DIR / "lm_words.json.gz"
    try:
        if path.exists():
            d = json.loads(gzip.decompress(path.read_bytes()))
            _LM_CACHE = {k: set(d[k]) for k in ("negative", "positive", "uncertainty")} | {"source": d["source"]}
            return _LM_CACHE
        if download:
            import requests

            resp = requests.get(LM_URL, timeout=60, headers={"User-Agent": config.USER_AGENT})
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), usecols=["Word", "Negative", "Positive", "Uncertainty"])
            d = {
                "negative": sorted(df.loc[df["Negative"] > 0, "Word"].str.lower()),
                "positive": sorted(df.loc[df["Positive"] > 0, "Word"].str.lower()),
                "uncertainty": sorted(df.loc[df["Uncertainty"] > 0, "Word"].str.lower()),
                "source": "loughran-mcdonald-master-dictionary",
            }
            if len(d["negative"]) > 1000:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(json.dumps(d).encode()))
                _LM_CACHE = {k: set(d[k]) for k in ("negative", "positive", "uncertainty")} | {"source": d["source"]}
                return _LM_CACHE
    except Exception as exc:  # network / parse problem -> curated fallback
        log.warning("Loughran-McDonald dictionary unavailable (%s); using the embedded curated list", exc)
    _LM_CACHE = {k: set(v) for k, v in _CURATED.items()} | {"source": "curated-fallback"}
    return _LM_CACHE


def lm_tone(text: str, lm: dict | None = None) -> dict[str, float]:
    """Percent of words in each LM list plus net tone (pos - neg) / (pos + neg). Text: any case."""
    lm = lm or load_lm()
    words = _WORD.findall(text.lower())
    n = len(words)
    if n == 0:
        return dict.fromkeys(TONE_COLS, np.nan)
    neg = sum(w in lm["negative"] for w in words)
    pos = sum(w in lm["positive"] for w in words)
    unc = sum(w in lm["uncertainty"] for w in words)
    return {
        "lm_negative_pct": 100 * neg / n,
        "lm_positive_pct": 100 * pos / n,
        "lm_uncertainty_pct": 100 * unc / n,
        "lm_net_tone": (pos - neg) / (pos + neg) if pos + neg else 0.0,
    }


# --------------------------------------------------------------------------------------
# SEC fetch helpers (idea from tradepulse/ingest/edgar.py: index table -> exhibit URL)
# --------------------------------------------------------------------------------------
def index_url(cik: int, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.htm"


def parse_index(index_html: str) -> list[dict]:
    """Rows of the 'Document Format Files' table: seq, description, type, url."""
    from bs4 import BeautifulSoup

    out = []
    for table in BeautifulSoup(index_html, "lxml").find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if "Type" not in headers or "Document" not in headers:
            continue
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) != len(headers):
                continue
            link = cells[headers.index("Document")].find("a", href=True)
            if not link:
                continue
            href = parse_qs(urlsplit(link["href"]).query).get("doc", [link["href"]])[0]
            url = urljoin("https://www.sec.gov", href)
            parts = urlsplit(url)
            if parts.hostname != SEC_HOST or not parts.path.startswith("/Archives/"):
                continue
            out.append({
                "seq": cells[headers.index("Seq")].get_text(strip=True) if "Seq" in headers else "",
                "type": cells[headers.index("Type")].get_text(strip=True),
                "url": url,
            })
    return out


def choose_exhibit(docs: list[dict], form: str) -> tuple[str, str] | None:
    """Main press-release document: EX-99.1, else EX-99, else another EX-99.x, else the primary document.
    Returns (type, url). Non-8-K forms (424B*, NT 10-K/Q) always use the primary document."""
    def pick(pred):
        for d in docs:
            if pred(d["type"].upper()) and not d["url"].lower().endswith((".jpg", ".png", ".gif", ".pdf", ".xml", ".xsd")):
                return d["type"], d["url"]
        return None

    if form.startswith("8-K"):
        hit = pick(lambda t: t == "EX-99.1") or pick(lambda t: t == "EX-99") or pick(lambda t: t.startswith("EX-99"))
        if hit:
            return hit
    primary = pick(lambda t: t == form.upper()) or pick(lambda t: t.startswith(form.upper().split("/")[0]))
    if primary:
        return primary
    return pick(lambda t: not t.startswith("EX-") and not t.startswith("GRAPHIC") and not t.startswith("XML"))


def primary_document(docs: list[dict], form: str) -> tuple[str, str] | None:
    for d in docs:
        if d["type"].upper() == form.upper() and not d["url"].lower().endswith((".xml", ".xsd")):
            return d["type"], d["url"]
    return None


def html_to_text(html: str, max_chars: int = TEXT.max_chars) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for node in soup(["script", "style"]):
        node.decompose()
    return " ".join(soup.get_text(" ").split())[:max_chars]


# --------------------------------------------------------------------------------------
# Windows and target selection
# --------------------------------------------------------------------------------------
def window_filings(events: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Every (event, filing) pair with the filing accepted in (prev close 16:00 ET, next morning 09:00 ET],
    exactly the window of features.edgar_features. Columns: event_idx + the filing columns."""
    fl = filings.dropna(subset=["accepted_et"]).sort_values("accepted_et")
    by_cik = {k: g for k, g in fl.groupby("cik")}
    parts = []
    ev = events.dropna(subset=["cik"])
    for cik, evs in ev.groupby("cik"):
        g = by_cik.get(int(cik))
        if g is None:
            continue
        start = (pd.to_datetime(evs["prev_date"]).fillna(evs["date"] - pd.Timedelta(days=1)) + pd.Timedelta(hours=16)).values
        end = (pd.to_datetime(evs["next_date"]).fillna(evs["date"] + pd.Timedelta(days=1)) + pd.Timedelta(hours=9)).values
        t = g["accepted_et"].values
        lo, hi = np.searchsorted(t, start, "right"), np.searchsorted(t, end, "right")
        for idx, a, b in zip(evs.index, lo, hi):
            if b > a:
                w = g.iloc[a:b].copy()
                w["event_idx"] = idx
                parts.append(w)
    if not parts:
        return pd.DataFrame(columns=["event_idx", *filings.columns])
    return pd.concat(parts, ignore_index=True)


def select_targets(events: pd.DataFrame, filings: pd.DataFrame, done: set[str] | None = None,
                   cfg: TextConfig = TEXT, include_earnings: bool = True, max_filings: int | None = None) -> pd.DataFrame:
    """Filings to fetch, crash families first (newest events first so recent data lands early), then the
    2022+ earnings 8-K Item 2.02 filings. `done` = accessions already stored with a final status."""
    done = done or set()
    cols = ["ticker", "date", "cik", "prev_date", "next_date", "event_family"]
    if events.empty or filings.empty or not set(cols) <= set(events.columns):
        return pd.DataFrame(columns=["accession", "cik", "form", "items", "accepted_et", "priority"])
    ev = events[cols]
    out = []
    crash = ev[ev["event_family"].isin(cfg.crash_families)]
    w = window_filings(crash, filings)
    if not w.empty:
        w = w[w["form"].str.startswith(cfg.crash_forms_prefix)]
        w["priority"] = 0
        out.append(w)
    if include_earnings:
        earn = ev[ev["event_family"].isin(cfg.earnings_families) & (ev["date"] >= pd.Timestamp(cfg.earnings_from))]
        w = window_filings(earn, filings)
        if not w.empty:
            w = w[w["form"].str.startswith("8-K") & w["items"].fillna("").str.contains(cfg.earnings_item, regex=False)]
            w["priority"] = 1
            out.append(w)
    if not out:
        return pd.DataFrame(columns=["accession", "cik", "form", "items", "accepted_et", "priority"])
    t = pd.concat(out, ignore_index=True)
    t = t.sort_values(["priority", "accepted_et"], ascending=[True, False]).drop_duplicates(["accession", "cik"])
    t = t[~t["accession"].isin(done)]
    t = t[["accession", "cik", "form", "items", "accepted_et", "priority"]].reset_index(drop=True)
    return t.head(max_filings if max_filings is not None else cfg.max_filings)


# --------------------------------------------------------------------------------------
# Analysing one filing
# --------------------------------------------------------------------------------------
def analyse_text(text: str, main_text: str | None = None, cfg: TextConfig = TEXT, lm: dict | None = None) -> dict:
    """Flags/snippets from `text` (all fetched text), tone from `main_text` (the main exhibit, default = text)."""
    norm = normalise(text)
    flags, snips = find_flags(norm, cfg.snippet_chars)
    off, last, disc = parse_offering_discount(norm)
    tone = lm_tone(main_text if main_text is not None else text, lm)
    return {**flags, **tone, "offering_price": off, "last_price": last, "offering_discount": disc,
            "snippets": json.dumps(snips, ensure_ascii=False), "n_chars": len(norm)}


def _cache_path(accession: str):
    return CACHE / f"{accession}.txt.gz"


def _download_text(sess, url: str, cfg: TextConfig) -> str | None:
    resp = store.get(sess, url, _LIMITER)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    body = resp.content[: cfg.max_bytes]
    html = body.decode(resp.encoding or "utf-8", errors="replace") if resp.encoding else body.decode("utf-8", errors="replace")
    return html_to_text(html, cfg.max_chars)


def process_filing(sess, row: dict, cfg: TextConfig = TEXT, lm: dict | None = None) -> dict:
    """Fetch + analyse one filing. Never raises: failures become a status row (`error` is retried later)."""
    base = {"accession": row["accession"], "cik": int(row["cik"]), "form": row["form"], "accepted_et": row["accepted_et"],
            "fetched_at": pd.Timestamp.now()}
    try:
        resp = store.get(sess, index_url(row["cik"], row["accession"]), _LIMITER)
        if resp.status_code == 404:
            return {**base, "status": "no_index", "exhibit_type": None}
        resp.raise_for_status()
        docs = parse_index(resp.text)
        choice = choose_exhibit(docs, row["form"])
        if not choice:
            return {**base, "status": "no_doc", "exhibit_type": None}
        etype, url = choice
        main = _download_text(sess, url, cfg)
        if main is None:
            return {**base, "status": "no_doc", "exhibit_type": etype}
        text = main
        items = str(row.get("items") or "")
        if row["form"].startswith("8-K") and etype.upper().startswith("EX-") and any(i in items for i in cfg.primary_items):
            prim = primary_document(docs, row["form"])
            if prim:
                extra = _download_text(sess, prim[1], cfg)
                if extra:
                    text = (main + " " + extra)[: 2 * cfg.max_chars]
        if cfg.keep_raw:
            CACHE.mkdir(parents=True, exist_ok=True)
            with gzip.open(_cache_path(row["accession"]), "wt", encoding="utf-8", compresslevel=6) as fh:
                fh.write(json.dumps({"exhibit_type": etype, "main": main, "all": text}))
        res = analyse_text(text, main, cfg, lm)
        return {**base, "status": "ok", "exhibit_type": etype, **res}
    except Exception as exc:
        log.warning("filing_text %s failed: %s", row["accession"], exc)
        return {**base, "status": "error", "exhibit_type": None}


def done_accessions() -> set[str]:
    if not store.table_exists(TABLE):
        return set()
    with store.connect() as con:
        return {r[0] for r in con.execute(f"select accession from {TABLE} where status in ('ok','no_index','no_doc')").fetchall()}


def _save(rows: list[dict]) -> None:
    if rows:
        df = pd.DataFrame(rows)
        for c in (*FLAG_NAMES, "n_chars"):
            if c in df:
                df[c] = df[c].astype("Int64")
        store.upsert(TABLE, df, ["accession", "cik"])


def ingest(max_filings: int | None = None, include_earnings: bool = True, cfg: TextConfig = TEXT,
           batch: int = 200, max_seconds: float | None = None) -> dict:
    """Fetch + classify the next `max_filings` unfetched target filings. Resumable: call again to continue."""
    t0 = time.time()
    ev = store.read_table("events")
    fl = store.read_table("filings")
    if ev.empty or fl.empty:   # first run: `events` is built by the transform step that follows ingest
        return {"targets": 0, "skipped": True, "reason": "no events yet; filing text starts on the next run"}
    done = done_accessions()
    targets = select_targets(ev, fl, done, cfg, include_earnings, max_filings)
    lm = load_lm()
    log.info("filing_text: %d targets (LM source %s)", len(targets), lm["source"])
    sess = store.session()
    rows, stats = [], {"targets": len(targets), "ok": 0, "no_index": 0, "no_doc": 0, "error": 0}
    recs = targets.to_dict("records")
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for i, res in enumerate(pool.map(lambda r: process_filing(sess, r, cfg, lm), recs), 1):
            stats[res["status"]] += 1
            rows.append(res)
            if len(rows) >= batch:
                _save(rows)
                rows = []
                log.info("filing_text %d/%d %s (%.0fs)", i, len(recs), {k: v for k, v in stats.items() if k != "targets"}, time.time() - t0)
            if max_seconds and time.time() - t0 > max_seconds:
                log.info("time budget reached")
                pool.shutdown(wait=False, cancel_futures=True)
                break
    _save(rows)
    stats["seconds"] = round(time.time() - t0)
    return stats


def refit(cfg: TextConfig = TEXT) -> int:
    """Re-run lexicon + tone on the cached raw texts (after editing the patterns); no network."""
    existing = store.read_table(TABLE)
    if existing.empty:
        return 0
    lm = load_lm()
    rows, n = [], 0
    for rec in existing[existing["status"] == "ok"].to_dict("records"):
        p = _cache_path(rec["accession"])
        if not p.exists():
            continue
        d = json.loads(gzip.decompress(p.read_bytes()))
        rows.append({**{k: rec[k] for k in ("accession", "cik", "form", "accepted_et", "fetched_at", "status", "exhibit_type")},
                     **analyse_text(d["all"], d["main"], cfg, lm)})
        n += 1
    _save(rows)
    return n


# --------------------------------------------------------------------------------------
# Point-in-time aggregation per event
# --------------------------------------------------------------------------------------
FT_COLUMNS = (
    ["ft_n_filings_window", "ft_n_text", "ft_text_available"]
    + [f"ft_{f}" for f in FLAG_NAMES]
    + ["ft_lm_negative_pct", "ft_lm_positive_pct", "ft_lm_uncertainty_pct", "ft_lm_net_tone", "ft_offering_discount"]
)
TARGET_FORMS = ("8-K", "424B", "NT 10-")
CATEGORY_TEXT_VALUES = (
    "earnings_guidance_cut", "earnings_guidance_raise", "earnings_guidance_mixed", "earnings_other",
    "going_concern", "clinical_failure", "offering_priced", "investigation",
)


def refine_category(row: pd.Series) -> str:
    """edgar_category_text for one event (row needs edgar_category + ft_* flags)."""
    cat = row.get("edgar_category")
    if not isinstance(cat, str) or not cat:  # events without a CIK have no EDGAR category
        cat = "none"
    if cat == "bankruptcy_delisting":
        return cat
    if row["ft_going_concern"]:
        return "going_concern"
    if row["ft_clinical_failure"]:
        return "clinical_failure"
    if cat == "restatement_auditor":
        return cat
    if row["ft_offering_priced"] and cat in ("offering", "earnings", "none", "regfd_other_8k", "deal", "mgmt_change"):
        return "offering_priced"
    if cat == "earnings":
        cut, up = row["ft_guidance_cut"], row["ft_guidance_raise"]
        if cut and up:
            return "earnings_guidance_mixed"
        if cut:
            return "earnings_guidance_cut"
        if up:
            return "earnings_guidance_raise"
        return "earnings_other"
    if row["ft_investigation"]:
        return "investigation"
    return cat


def attach_filing_text(events: pd.DataFrame, filing_text: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Per-event text features, aligned to events.index.

    events      ticker/date/cik/prev_date/next_date (+ edgar_category for the refined category)
    filing_text rows of the warehouse table `filing_text` (status ok rows carry flags / tone)
    filings     warehouse `filings` (accession, cik, form, items, accepted_et)

    Point-in-time: only filings accepted in (prev close 16:00 ET, next morning 09:00 ET]. Flags are 1 if ANY
    filing of the window has them (0 when no text was fetched; check `ft_text_available`); tone comes from the
    main exhibit (largest EX-99 8-K text, else the largest text). Returns ft_* columns and edgar_category_text."""
    out = pd.DataFrame(0, index=events.index, columns=list(FT_COLUMNS), dtype=float)
    for c in ("ft_lm_negative_pct", "ft_lm_positive_pct", "ft_lm_uncertainty_pct", "ft_lm_net_tone", "ft_offering_discount"):
        out[c] = np.nan
    w = window_filings(events, filings[filings["form"].str.startswith(TARGET_FORMS)]) if len(events) else pd.DataFrame()
    if not w.empty:
        n_win = w.groupby("event_idx").size()
        out.loc[n_win.index, "ft_n_filings_window"] = n_win
    if not w.empty and filing_text is not None and not filing_text.empty:
        ok = filing_text[filing_text["status"] == "ok"].drop_duplicates(["accession", "cik"])
        j = w.merge(ok.drop(columns=[c for c in ("form", "accepted_et", "cik") if c in ok and c != "cik"], errors="ignore"),
                    on=["accession", "cik"], how="inner")
        if not j.empty:
            g = j.groupby("event_idx")
            out.loc[g.size().index, "ft_n_text"] = g.size()
            out.loc[g.size().index, "ft_text_available"] = 1
            flags = g[list(FLAG_NAMES)].max()
            for f in FLAG_NAMES:
                out.loc[flags.index, f"ft_{f}"] = flags[f].fillna(0).astype(float)
            disc = g["offering_discount"].max()
            out.loc[disc.index, "ft_offering_discount"] = disc
            j["_main"] = j["exhibit_type"].fillna("").str.upper().str.startswith("EX-99") * 10**9 + j["n_chars"].fillna(0).astype(float)
            main = j.sort_values("_main", ascending=False).drop_duplicates("event_idx").set_index("event_idx")
            for c in ("lm_negative_pct", "lm_positive_pct", "lm_uncertainty_pct", "lm_net_tone"):
                out.loc[main.index, f"ft_{c}"] = main[c].astype(float)
    int_cols = ["ft_n_filings_window", "ft_n_text", "ft_text_available"] + [f"ft_{f}" for f in FLAG_NAMES]
    out[int_cols] = out[int_cols].fillna(0).astype(int)
    cat = events["edgar_category"] if "edgar_category" in events else pd.Series("none", index=events.index)
    tmp = out.assign(edgar_category=cat.values)
    out["edgar_category_text"] = tmp.apply(refine_category, axis=1) if len(tmp) else pd.Series(dtype=object)
    return out


# --------------------------------------------------------------------------------------
# Event study (evidence)
# --------------------------------------------------------------------------------------
PERIODS = {"train<=2021": ("1900-01-01", "2021-12-31"), "valid2022-23": ("2022-01-01", "2023-12-31"), "test2024+": ("2024-01-01", "2100-01-01"),
           "all": ("1900-01-01", "2100-01-01")}


def _boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(x) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _stats(sub: pd.DataFrame, threshold: float) -> dict:
    r = sub["growth_future_10d"].dropna() - 1
    rs = sub["growth_future_10d_vs_spy"].dropna()
    rs = rs - 1 if len(rs) and rs.abs().median() > 0.5 else rs  # tolerate either "ratio" or "excess" convention
    lo, hi = _boot_ci(r.values)
    return {
        "n": len(r), "base_rate": float((sub["growth_future_10d"].dropna() >= threshold).mean()) if len(r) else np.nan,
        "mean_ret_10d": float(r.mean()) if len(r) else np.nan, "median_ret_10d": float(r.median()) if len(r) else np.nan,
        "mean_vs_spy": float(rs.mean()) if len(rs) else np.nan, "ci_lo": lo, "ci_hi": hi,
    }


def event_study(df: pd.DataFrame, threshold: float = 1.02) -> pd.DataFrame:
    """df: events + ft_* + edgar_category_text + event_family + targets. One row per
    (group kind, group, family-group, period)."""
    d = df.dropna(subset=["growth_future_10d"]).copy()
    d["family_group"] = np.where(d["event_family"].isin(("shock_crash", "earnings_crash")), "crash", "earnings")
    d.loc[d["event_family"].isin(("earnings_up", "earnings_down")), "family_group"] = "earnings_post"
    groups: list[tuple[str, str, pd.Series]] = [("all", "all_events", pd.Series(True, index=d.index))]
    for f in FLAG_NAMES:
        groups.append(("flag", f"ft_{f}=1", d[f"ft_{f}"] == 1))
        groups.append(("flag", f"ft_{f}=0 (text read)", (d[f"ft_{f}"] == 0) & (d["ft_text_available"] == 1)))
    for col in ("edgar_category", "edgar_category_text"):  # events without a CIK carry None
        d[col] = d[col].fillna("none").astype(str)
    for c in sorted(d["edgar_category_text"].unique()):
        groups.append(("category_text", c, d["edgar_category_text"] == c))
    for c in sorted(d["edgar_category"].unique()):
        groups.append(("category_items", c, d["edgar_category"] == c))
    # the guidance contrast inside earnings releases, where text was read
    er = (d["edgar_category"] == "earnings") & (d["ft_text_available"] == 1)
    groups += [("guidance", "earnings+cut", er & (d["ft_guidance_cut"] == 1) & (d["ft_guidance_raise"] == 0)),
               ("guidance", "earnings+raise", er & (d["ft_guidance_raise"] == 1) & (d["ft_guidance_cut"] == 0)),
               ("guidance", "earnings+neither", er & (d["ft_guidance_raise"] == 0) & (d["ft_guidance_cut"] == 0))]
    rows = []
    for kind, name, mask in groups:
        for fam in ("crash", "earnings_post", "all"):
            sel = mask if fam == "all" else mask & (d["family_group"] == fam)
            for per, (a, b) in PERIODS.items():
                sub = d[sel & (d["date"] >= pd.Timestamp(a)) & (d["date"] <= pd.Timestamp(b))]
                if len(sub) == 0:
                    continue
                rows.append({"kind": kind, "group": name, "family": fam, "period": per, **_stats(sub, threshold)})
    return pd.DataFrame(rows)


def load_study_frame() -> pd.DataFrame:
    """events (needed columns) + ft_* attached from the warehouse."""
    with store.connect(read_only=True) as con:
        ev = con.execute(
            "select ticker, date, cik, prev_date, next_date, event_family, edgar_category, growth_future_10d, growth_future_10d_vs_spy from events"
        ).df()
    ft = store.read_table(TABLE)
    fl = store.read_table("filings")
    ev = ev[ev["event_family"] != "momentum_breakout"].reset_index(drop=True)
    return pd.concat([ev, attach_filing_text(ev, ft, fl)], axis=1)


def run_study() -> pd.DataFrame:
    df = load_study_frame()
    res = event_study(df, config.SETTINGS.target.threshold)
    out = config.ARTIFACTS_DIR / "filing_text_study.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    df.to_parquet(config.ARTIFACTS_DIR / "filing_text_events.parquet", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ingest", "refit", "study"])
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.cmd == "ingest":
        print(ingest(a.max, not a.no_earnings, max_seconds=a.max_seconds))
    elif a.cmd == "refit":
        print({"refit": refit()})
    else:
        print(run_study().head(40).to_string())


if __name__ == "__main__":
    main()
