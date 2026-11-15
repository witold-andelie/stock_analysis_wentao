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


# --------------------------------------------------------------------------------------
# Storage. Data never goes to GitHub: everything lives under DATA_DIR (default
# project/data/, git-ignored). Point DRIFTREV_DATA_DIR at a mounted drive, a Docker
# volume or a synced Google-Drive folder to keep it elsewhere.
# --------------------------------------------------------------------------------------
DATA_DIR = Path(_env("DRIFTREV_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
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
    "DGS10": "dgs10", "DGS2": "dgs2", "DFF": "fedfunds",
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


@dataclass(frozen=True)
class Costs:
    """One-way trading cost in basis points (commission + half-spread + slippage).
    Small/mid caps right after a crash have wide spreads: ASSUMPTION 15 bps per side."""

    bps_one_way: float = 15.0

    @property
    def round_trip(self) -> float:
        return 2 * self.bps_one_way / 1e4


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
    signal_history_days: int = 60       # data/artifacts/history/YYYY-MM-DD/
    snapshot_zips_keep: int = 1
    run_log_max_lines: int = 1000
    log_max_bytes: int = 5_000_000
    # GitHub side (cleanup.yml reads these through `python -m driftrev.housekeeping --print-github`)
    gh_cache_days: int = 3
    gh_artifact_days: int = 30
    gh_run_days: int = 30


@dataclass(frozen=True)
class Settings:
    event: EventRule = field(default_factory=EventRule)
    target: Target = field(default_factory=Target)
    split: Split = field(default_factory=Split)
    costs: Costs = field(default_factory=Costs)
    strategy: Strategy = field(default_factory=Strategy)
    regime: Regime = field(default_factory=Regime)
    retention: Retention = field(default_factory=Retention)
    random_state: int = 42


SETTINGS = Settings()


def ensure_dirs() -> None:
    for path in (DATA_DIR, CACHE_DIR, SNAPSHOT_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
