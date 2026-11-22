"""German events and research through the shared feature and model pipelines."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import pandas as pd

from . import (
    artifacts,
    config,
    consensus,
    dataset,
    de_context,
    de_market,
    de_membership,
    de_universe,
    features,
    macro,
    pipeline,
    prices,
    store,
)
from .de_market import _require_de

FEATURE_UNIVERSE_COLS = ["ticker", "company", "sector", "sub_industry", "cap_bucket", "cik", "wikipedia_title",
                         "sector_etf", "is_current_member", "price_source"]


def membership_for_features(membership_de: pd.DataFrame, universe_de: pd.DataFrame) -> pd.DataFrame:
    """Map German company keys and index codes to the shared ticker membership schema."""
    mapped = membership_de.merge(universe_de[["company_key", "symbol"]], on="company_key", how="inner")
    mapped = mapped.loc[mapped["symbol"].notna() & mapped["symbol"].ne("")]
    return (mapped.rename(columns={"symbol": "ticker", "index_code": "index_name"})
            [["ticker", "index_name", "start_date", "end_date"]].reset_index(drop=True))


def universe_for_features(universe_de: pd.DataFrame) -> pd.DataFrame:
    """One row per resolved symbol, with unknown German sector and filing metadata."""
    uni = universe_de.loc[universe_de["symbol"].notna() & universe_de["symbol"].ne("")].copy()
    uni = uni.drop_duplicates("symbol").rename(columns={"symbol": "ticker", "name": "company"})
    uni = uni.assign(sector="unknown", sub_industry=None, cik=None, wikipedia_title=None, sector_etf=None)
    return uni[FEATURE_UNIVERSE_COLS].reset_index(drop=True)


def german_earnings(cons: pd.DataFrame, sessions) -> dict[str, pd.DataFrame]:
    """Date German releases while preserving the original ET-naive acceptance timestamp."""
    releases = consensus.release_session_dates(cons, sessions, config.MARKETS["de"])
    return {ticker: rows[["date", "release_et"]].rename(columns={"release_et": "earn_accepted_et"}).reset_index(drop=True)
            for ticker, rows in releases.groupby("ticker")}


def ingest_de(refresh: bool = False) -> dict:
    """Update German market series, rates, membership, universe, prices and consensus EPS."""
    _require_de()
    report = {"market": de_market.update_de_market(), "rates": de_market.update_de_rates()}
    report["membership"] = de_membership.update_de_membership(refresh=refresh)
    report["universe"] = de_universe.update_de_universe()
    universe_de = store.read_table(de_universe.TABLE)
    symbols = universe_de.loc[universe_de["price_source"].eq("xetra"), "symbol"].dropna()
    report["consensus"] = consensus.update_consensus(sorted(set(symbols[symbols.ne("")])))
    return report


def transform_de(save: bool = True) -> dict[str, pd.DataFrame]:
    """Build German events, daily panel and context, optionally replacing warehouse tables."""
    _require_de()
    universe_de = store.read_table(de_universe.TABLE)
    membership_de = store.read_table(de_membership.TABLE)
    uni = universe_for_features(universe_de)
    membership = membership_for_features(membership_de, universe_de)
    px = prices.load_prices()
    market = prices.load_market()
    macro_wide = macro.load_macro_wide()
    ctx = de_context.market_context_de(market, macro_wide)
    consensus_eps = consensus.load_consensus()
    earnings_by_ticker = german_earnings(consensus_eps, ctx.index)
    ev, panel, ctx = features.build_all(
        uni, px, market, macro_wide, filings=pd.DataFrame(), sue=pd.DataFrame(), pageviews=pd.DataFrame(),
        membership=membership, consensus_eps=consensus_eps, filing_text=None, ctx=ctx,
        earnings_by_ticker=earnings_by_ticker,
        cap_buckets=dict(config.MARKETS["de"].cap_bucket_by_index),
        liquid_by_membership=uni.loc[uni["price_source"].ne("xetra"), "ticker"],
        min_dollar_volume=config.MARKETS["de"].min_dollar_volume,
        breakout_min_median_turnover=config.MARKETS["de"].breakout_min_median_turnover)
    ctx = ctx.rename_axis("date").reset_index()
    if save:
        store.replace_table("events", ev)
        store.replace_table("daily_panel", panel)
        store.replace_table("market_ctx", ctx)
    return {"events": ev, "panel": panel, "market": market, "universe": uni, "ctx": ctx}


def load_from_warehouse_de() -> dict[str, pd.DataFrame]:
    """Load the transformed German warehouse with the shared feature universe schema."""
    _require_de()
    return {"events": store.read_table("events"), "panel": store.read_table("daily_panel"),
            "market": prices.load_market(), "universe": universe_for_features(store.read_table(de_universe.TABLE)),
            "ctx": store.read_table("market_ctx")}


def event_summary(events: pd.DataFrame) -> dict:
    """JSON-compatible counts and date coverage for the German event dataset."""
    if events.empty:
        return {"events": 0, "by_family": {}, "by_cap_bucket": {}, "by_year": {},
                "first_date": "", "last_date": ""}
    dates = pd.to_datetime(events["date"])
    return {
        "events": int(len(events)),
        "by_family": {str(k): int(v) for k, v in events["event_family"].value_counts().items()},
        "by_cap_bucket": {str(k): int(v) for k, v in events["cap_bucket"].value_counts().items()},
        "by_year": {int(k): int(v) for k, v in dates.dt.year.value_counts().sort_index().items()},
        "first_date": dates.min().date().isoformat(), "last_date": dates.max().date().isoformat(),
    }


def run_de(refresh: bool = False) -> dict:
    """Update German raw inputs, transform them and report event counts."""
    report = ingest_de(refresh=refresh)
    data = transform_de()
    report["transform"] = event_summary(data["events"])
    return report


def research_de(tune: bool = False, with_walkforward: bool = True) -> dict:
    """Train and backtest the transformed German warehouse and persist research metrics."""
    _require_de()
    events = store.read_table("events")
    panel = store.read_table("daily_panel")
    if events.empty or panel.empty:
        raise RuntimeError("run transform_de first")
    market = prices.load_market()
    config.ensure_dirs()
    ml, feats = dataset.prepare(events)
    metrics = artifacts.load_metrics()
    metrics.update(pipeline.train_and_backtest(ml, feats, panel, market, tune=tune, with_walkforward=with_walkforward))
    metrics.update({"market": "de", "generated_at": datetime.now(UTC).isoformat(),
                    "data_through": pd.Timestamp(panel["date"].max()).date().isoformat(),
                    "n_events": int(len(events)), "n_panel_rows": int(len(panel)), "mode": "warehouse"})
    artifacts.save_metrics(metrics)
    return metrics


def main() -> None:
    """Run ingestion and transformation, or transform/research the existing German warehouse."""
    ap = argparse.ArgumentParser(description="German event pipeline (crash, breakout and earnings events)")
    ap.add_argument("--transform-only", action="store_true", help="reuse warehouse inputs without downloading")
    ap.add_argument("--research", action="store_true", help="train and backtest the transformed warehouse without downloading")
    ap.add_argument("--tune", action="store_true", help="re-run the research hyper-parameter grid search")
    args = ap.parse_args()
    _require_de()
    logging.basicConfig(level=logging.INFO)
    if args.research:
        report = research_de(tune=args.tune)
    else:
        report = event_summary(transform_de()["events"]) if args.transform_only else run_de()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
