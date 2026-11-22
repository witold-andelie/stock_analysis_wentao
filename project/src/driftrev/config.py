"""Central configuration for the Drift-or-Reversal event-driven strategy.

Every tunable number lives here (or in an environment variable that overrides it),
so the research notebooks, the CLI pipeline, the API and the dashboard all agree.
Values marked ASSUMPTION are judgement calls documented in the README.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: read project/.env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Market:
    """One tradable market. R7.1: `us` is implemented, `de` is declared only (R7.2+)."""

    code: str
    name: str
    exchange_mic: str
    timezone: str
    run_time: str
    currency: str
    index_set: tuple[str, ...]
    cap_bucket_by_index: tuple[tuple[str, str], ...]
    benchmarks: tuple[tuple[str, str], ...]
    cost_bps_one_way: tuple[tuple[str, float], ...]
    data_subdir: str
    implemented: bool

    def cap_bucket(self, index_code: str) -> str | None:
        """Cap bucket of an index code, or None if the code is not in this market."""
        return dict(self.cap_bucket_by_index).get(index_code)

    def cost_bps(self, cap_bucket: str) -> float:
        """One-way cost in bps for a cap bucket; unknown buckets get the "mid" value."""
        costs = dict(self.cost_bps_one_way)
        return costs.get(cap_bucket, costs["mid"])


MARKETS: dict[str, Market] = {
    "us": Market(
        code="us",
        name="US equities (S&P 1500)",
        exchange_mic="XNYS",
        timezone="America/New_York",
        run_time="22:30",
        currency="USD",
        index_set=("sp500", "sp400", "sp600"),
        cap_bucket_by_index=(("sp500", "large"), ("sp400", "mid"), ("sp600", "small")),
        benchmarks=(("SPY", "spy"), ("IWM", "iwm"), ("MDY", "mdy")),
        cost_bps_one_way=(("large", 8.0), ("mid", 15.0), ("small", 25.0)),
        data_subdir="",
        implemented=True,
    ),
    "de": Market(
        code="de",
        name="German equities (Xetra: DAX + MDAX + SDAX)",
        exchange_mic="XETR",
        timezone="Europe/Berlin",
        run_time="18:30",
        currency="EUR",
        index_set=("dax", "mdax", "sdax"),
        cap_bucket_by_index=(("dax", "large"), ("mdax", "mid"), ("sdax", "small")),
        benchmarks=(("^GDAXI", "dax"), ("^MDAXI", "mdax"), ("^SDAXI", "sdax")),
        # ASSUMPTION (R7.1): Xetra event-day spreads; revisit in R7.4
        cost_bps_one_way=(("large", 10.0), ("mid", 20.0), ("small", 35.0)),
        data_subdir="de",
        implemented=False,
    ),
}


def get_market(code: str) -> Market:
    """Look up a market by code, ignoring case and surrounding whitespace."""
    code = code.strip().lower()
    if code not in MARKETS:
        raise ValueError(f"unknown market {code!r}; valid: {', '.join(sorted(MARKETS))}")
    return MARKETS[code]


class MarketNotImplementedError(RuntimeError):
    """Raised when an entry point is asked to run a market whose `implemented` is False."""


def require_implemented_market(market: Market | None = None) -> Market:
    """Return `market` (default: MARKET) if implemented, else raise MarketNotImplementedError."""
    if market is None:
        market = MARKET
    if not market.implemented:
        raise MarketNotImplementedError(
            f"market {market.code!r} is planned (roadmap R7.2+) and not implemented yet; set DRIFTREV_MARKET=us"
        )
    return market


MARKET: Market = get_market(_env("DRIFTREV_MARKET", "us"))


# --------------------------------------------------------------------------------------
# Storage. Data never goes to GitHub: everything lives under DATA_DIR (default
# project/data/, git-ignored). Point DRIFTREV_DATA_DIR at a mounted drive, a Docker
# volume or a synced Google-Drive folder to keep it elsewhere.
# --------------------------------------------------------------------------------------
DATA_ROOT = Path(_env("DRIFTREV_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
DATA_DIR = DATA_ROOT / MARKET.data_subdir if MARKET.data_subdir else DATA_ROOT
WAREHOUSE = DATA_DIR / "warehouse.duckdb"          # incremental raw + feature tables
CACHE_DIR = DATA_DIR / "cache"                      # raw API payloads (EDGAR / Wikimedia)
SNAPSHOT_DIR = DATA_DIR / "snapshot"                # "file" regime: ready-made parquet files
ARTIFACTS_DIR = DATA_DIR / "artifacts"              # models, predictions, trades, metrics
HISTORY_DIR = ARTIFACTS_DIR / "history"             # daily copies of the signal book (pruned)
SITE_DIR = DATA_DIR / "site"                        # static results site (GitHub Pages)
LOG_DIR = DATA_DIR / "logs"

# Download a ready-made snapshot (events dataset + price paths) for the "file" regime.
# Upload `data/snapshot/*.parquet` (made by `driftrev snapshot`) to Google Drive / S3 and
# put the direct-download URL of the zip here.
SNAPSHOT_URL = _env("DRIFTREV_SNAPSHOT_URL", "")

# Polite identification for SEC EDGAR / Wikimedia (both require a contact User-Agent).
# Set your own: SEC_USER_AGENT="Your Name your@email".
USER_AGENT = _env("SEC_USER_AGENT", "driftrev-research research@example.org")  # SEC needs "name email"
SEC_MAX_RPS = float(_env("SEC_MAX_RPS", "8"))       # SEC fair-access limit is 10 req/s
FILING_TEXT_MAX_SECONDS = float(_env("DRIFTREV_FILING_TEXT_MAX_SECONDS", "1200"))  # R4 fetch budget per run

# --------------------------------------------------------------------------------------
# Universe and history
# --------------------------------------------------------------------------------------
START_DATE = _env("DRIFTREV_START", "2014-01-01")
UNIVERSE_INDEXES = ("sp500", "sp400", "sp600")      # S&P 1500 = large + mid + small caps
MAX_TICKERS = int(_env("DRIFTREV_MAX_TICKERS", "0"))  # 0 = all (use e.g. 50 for a smoke test)

MARKET_TICKERS = {
    "SPY": "spy", "IWM": "iwm", "MDY": "mdy",
    "^VIX": "vix", "^VIX3M": "vix3m",
}
SECTOR_ETFS = {
    "Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
FRED_SERIES = {
    "DGS10": "dgs10", "DGS5": "dgs5", "DGS2": "dgs2", "DFF": "fedfunds",
    "BAA10Y": "credit_spread", "T10Y2Y": "t10y2y",
}


@dataclass(frozen=True)
class EventRule:
    """What counts as an event (one row of the ML dataset).

    Two families share one dataset:
      * earnings  - the first session that can react to an 8-K Item 2.02 earnings release
                    (HW1 idea: short-horizon momentum + post-earnings drift)
      * shock     - a non-earnings crash day (HW5 Q3 idea: panic vs permanent damage)
      * breakout  - a momentum breakout after a consolidation (strategy guide: rolling-window
                    pressure & release, Variant B), see rolling_window.py
    """

    max_ret_1d: float = -0.10        # close-to-close fall of at least 10%
    min_volume_mult: float = 2.0     # volume >= 2x the 20-day average (ASSUMPTION; HW5 said 3x)
    min_price: float = 3.0           # skip penny prices
    min_dollar_volume: float = 2e6   # 20d avg dollar volume, keeps it tradeable
    earnings_lookback_days: int = 3  # 8-K 2.02 accepted up to 3 days before still counts


@dataclass(frozen=True)
class Target:
    """Prediction target. Entry = next session's OPEN (D+1), exit = CLOSE of D+horizon."""

    horizon: int = 10
    threshold: float = 1.02          # bounce = growth >= +2% (covers ~2x round-trip cost)


@dataclass(frozen=True)
class Split:
    train_end: str = "2021-12-31"
    valid_end: str = "2023-12-31"    # test = 2024-01-01 .. today (strictly out of sample)
    # Purge: a 10-session label looks ~14 calendar days ahead, so the last rows before a boundary have labels
    # that are realised inside the next block. With purge=True those rows get split="purged" and are used by
    # nothing (not for fitting, thresholds or metrics). purge_days = 10 sessions ~ 14 calendar days + 2 safety.
    purge: bool = True
    purge_days: int = 16


@dataclass(frozen=True)
class Costs:
    """One-way trading cost in basis points (commission + half-spread + slippage).
    Small/mid caps right after a crash have wide spreads: ASSUMPTION 15 bps per side."""

    bps_one_way: float = 15.0

    @property
    def round_trip(self) -> float:
        return 2 * self.bps_one_way / 1e4


@dataclass(frozen=True)
class CapCosts:
    """Cap-aware one-way trading cost in bps (research item: small caps pay more after a crash).
    ASSUMPTION, justified: S&P 500 names trade with ~1-3 bps half-spread, but on an event day
    (earnings gap, crash) spreads widen several-fold, plus commission and slippage -> 8 bps;
    mid caps keep the original 15 bps (S&P 400 spreads ~5-10 bps normal, wider on events);
    small caps (S&P 600) have 15-30 bps half-spreads on event days and thinner books -> 25 bps.
    `enabled=False` restores the flat `Costs.bps_one_way` for comparison."""

    enabled: bool = True
    large: float = 8.0
    mid: float = 15.0
    small: float = 25.0

    def bps(self, cap_bucket, flat: float) -> float:
        if not self.enabled:
            return flat
        return {"large": self.large, "mid": self.mid, "small": self.small}.get(str(cap_bucket), self.mid)


@dataclass(frozen=True)
class Research:
    """Research switches (decided on VALIDATION, see pipeline.train_and_backtest / research.py)."""

    # which feature set feeds the production model: pooled | cs_keep_market | cs_drop_market
    # VALIDATION winner of the model comparison (b2): stock-level + cross-sectional, market-level inputs removed
    feature_set: str = "cs_drop_market"
    # production learner layout: pooled | per_family
    model_layout: str = "pooled"
    # model book composition: families / EDGAR categories the MODEL never trades (guide arms are a separate sleeve)
    book_exclude_families: tuple = ("momentum_breakout",)    # VALIDATION: Sharpe 0.29 -> 0.63
    book_exclude_categories: tuple = ("offering",)           # VALIDATION: +offering -> 0.69; mgmt_change NOT excluded (val precision 71%)
    # HW4: cap on average trades per day when choosing the probability threshold (None = off)
    max_trades_per_day: float | None = 5.0   # VALIDATION: Sharpe 0.29 (uncapped) -> 0.97 at cap 5
    # R2 walk-forward
    wf_first_year: int = 2018
    wf_purge_days: int = 16          # label horizon 10 sessions ~ 14 calendar days, +2 safety
    wf_model: str = "lgbm"           # learner refit each year
    sklearn_max_tested: str = "1.7"  # HW4 lesson: RF probabilities change from 1.8


@dataclass(frozen=True)
class Strategy:
    initial_capital: float = 10_000.0
    max_positions: int = 5
    stop_loss: float = 0.15          # exit if price trades 15% below entry
    take_profit: float = 0.25        # take gains at +25%
    hold_days: int = 10
    min_prob: float = 0.55           # probability rule threshold (tuned on VALIDATION)
    pause_in_stressed_regime: bool = True


@dataclass(frozen=True)
class Regime:
    """Fragility monitor (borrowed from TradePulse): each trigger adds a point."""

    trend_ma_days: int = 200
    vix_term_ratio_trigger: float = 1.0
    corr_window_days: int = 20
    corr_pctile_trigger: float = 0.90
    breadth_trigger: float = 0.40
    credit_change_trigger: float = 0.25   # Baa-10y spread +25 bp over 20 days (HY OAS history is no longer free)
    elevated_min_points: int = 2
    stressed_min_points: int = 4


@dataclass(frozen=True)
class Retention:
    """How long each class of local data file lives (housekeeping.py, SD7). Days unless noted.
    Raw tables in the warehouse are never pruned: the full history is the research dataset."""

    edgar_root_days: float = 2          # CIK*.json roots are re-fetched every refresh anyway
    edgar_history_days: float = 180     # older submission pages (re-downloadable)
    filing_text_raw_days: float = 30    # raw exhibit text; flags/tone/snippets stay in the warehouse
    signal_history_days: int = 60       # data/artifacts/history/YYYY-MM-DD/
    snapshot_zips_keep: int = 1
    run_log_max_lines: int = 1000
    log_max_bytes: int = 5_000_000
    # GitHub side (cleanup.yml reads these through `python -m driftrev.housekeeping --print-github`)
    gh_cache_days: int = 3
    gh_artifact_days: int = 30
    gh_run_days: int = 30


@dataclass(frozen=True)
class Membership:
    """R1 historical index membership (membership.py) + HW1 Q3 correction state + HW1 Q4 consensus."""

    restrict_events: bool = True      # events only while the stock was a member of the S&P 1500
    merge_gap_days: int = 10          # index migrations (400 -> 500) may leave a few days between rows
    price_tail_days: int = 45         # keep removed names' prices this long after removal (targets need 10 sessions)
    min_overlap_days: int = 20        # removed ticker needs >= this many price days inside its membership window
    wiki_rps: float = 4.0             # Wikimedia etiquette
    sec_search_rps: float = 4.0       # EDGAR company-search etiquette (SEC limit is 10 rps)
    name_match_min: float = 0.5       # token overlap between change-table name and EDGAR name (ticker-reuse guard)


@dataclass(frozen=True)
class Correction:
    threshold: float = 0.05           # SPY >= 5% below its running all-time high = correction (HW1 Q3)


@dataclass(frozen=True)
class Consensus:
    window_sessions: int = 3          # earnings release within +-3 sessions of the event date
    refresh_days: float = 7.0         # skip tickers refreshed more recently than this
    limit: int = 60                   # get_earnings_dates(limit=...) rows per ticker (~15 years of quarters)
    workers: int = 4
    max_failures: int = 40            # stop a run after this many consecutive rate-limit-looking failures


@dataclass(frozen=True)
class Settings:
    event: EventRule = field(default_factory=EventRule)
    target: Target = field(default_factory=Target)
    split: Split = field(default_factory=Split)
    costs: Costs = field(default_factory=Costs)
    strategy: Strategy = field(default_factory=Strategy)
    regime: Regime = field(default_factory=Regime)
    retention: Retention = field(default_factory=Retention)
    cap_costs: CapCosts = field(default_factory=CapCosts)
    research: Research = field(default_factory=Research)
    membership: Membership = field(default_factory=Membership)
    correction: Correction = field(default_factory=Correction)
    consensus: Consensus = field(default_factory=Consensus)
    random_state: int = 42


SETTINGS = Settings()


def ensure_dirs() -> None:
    for path in (DATA_DIR, CACHE_DIR, SNAPSHOT_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
