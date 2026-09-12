"""
Stock Market Analytics Zoomcamp - Module 1 Homework (2026 cohort) solution.

Covers:
  Q1: S&P 500 companies from Wikipedia         -> year with most index additions (from 2020)
  Q2: World indices YTD (2026-01-01..2026-08-21) vs S&P 500
  Q3: S&P 500 market corrections (drawdown >= 5%) since 1950
  Q4: AMZN earnings surprise -> 2-day return analysis
  Q5: (in report.md / README - free text)

All raw data is cached under data/raw/ as parquet, so the script can be re-run
offline. Analysis outputs go to data/outputs/ (CSV, Power BI friendly) and the
final answers are printed to stdout.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ----------------------------------------------------------------------------
# paths / helpers
# ----------------------------------------------------------------------------
DATA = Path(__file__).parent.parent / "data"
RAW = DATA / "raw"
OUT = DATA / "outputs"
for p in (RAW, OUT):
    p.mkdir(parents=True, exist_ok=True)


def cached(name: str, func, **kwargs):
    """Load parquet cache if present, else run func() and cache it."""
    path = RAW / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  [cache] {name}.parquet loaded ({len(df)} rows)")
        return df
    df = func(**kwargs)
    df.to_parquet(path)
    print(f"  [fetch] {name}.parquet saved ({len(df)} rows)")
    return df


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ----------------------------------------------------------------------------
# Q1 - S&P 500 additions (Wikipedia)
# ----------------------------------------------------------------------------
def q1():
    print("=" * 70)
    print("Q1 | S&P 500 companies - index additions")
    print("=" * 70)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def load_wiki():
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        # the constituents table is parseable directly from the raw HTML list too
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]
        return df

    sp500 = cached("q1_sp500_wiki", load_wiki)

    # normalize column names (they differ depending on pandas version)
    col_map = {}
    for c in sp500.columns:
        cl = c.lower()
        if "symbol" in cl or "ticker" in cl:
            col_map[c] = "ticker"
        elif "security" in cl or "company" in cl or "name" in cl:
            col_map[c] = "name"
        elif "date" in cl and ("added" in cl or "first" in cl):
            col_map[c] = "added"
    sp500 = sp500.rename(columns=col_map)
    need = {"ticker", "name", "added"}
    if not need.issubset(sp500.columns):
        print("  [!] unexpected columns:", list(sp500.columns))
        return
    sp500 = sp500[["ticker", "name", "added"]].copy()

    # fallback: parse "YYYY-MM-DD" from the added date
    sp500["added_dt"] = pd.to_datetime(sp500["added"], errors="coerce")
    sp500["added_year"] = sp500["added_dt"].dt.year
    print(
        f"\n  current constituents: {len(sp500)}; with additions data: {sp500['added_year'].notna().sum()}"
    )

    yearly = (
        sp500.dropna(subset=["added_year"])
        .groupby("added_year")["ticker"]
        .count()
        .rename("count")
        .reset_index()
    )
    yearly.to_csv(OUT / "q1_additions_by_year.csv", index=False)

    recent = yearly[yearly["added_year"] >= 2020].sort_values("count", ascending=False)
    best = recent.iloc[0]
    print("\n  Additions starting from 2020:")
    print(recent.to_string(index=False))
    print(
        f"\n  >>> Q1 answer: {int(best['added_year'])} had the highest number of additions ({int(best['count'])})"
    )

    # bonus: how many current members have been in the index > 20 years?
    cutoff = date.today().year - 20
    senior = sp500[sp500["added_year"] <= cutoff]
    print(
        f"  >>> Q1 bonus: {len(senior)} current members added in {cutoff} or earlier "
        f"(i.e. in the index for more than 20 years)"
    )

    # sanitize for Power BI (removes emoji that breaks Excel/pbix encoding on some locales)
    out = sp500.copy()
    out["added"] = out["added"].astype(str)
    out[["ticker", "name", "added", "added_dt", "added_year"]].to_csv(
        OUT / "q1_sp500_constituents.csv", index=False
    )


# ----------------------------------------------------------------------------
# Q2 - World indices YTD vs S&P 500
# ----------------------------------------------------------------------------
WORLD_INDICES = {
    "US - S&P 500": "^GSPC",
    "China - Shanghai Composite": "000001.SS",
    "Hong Kong - HANG SENG": "^HSI",
    "Australia - S&P/ASX 200": "^AXJO",
    "India - Nifty 50": "^NSEI",
    "Canada - S&P/TSX Composite": "^GSPTSE",
    "Germany - DAX": "^GDAXI",
    "UK - FTSE 100": "^FTSE",
    "Japan - Nikkei 225": "^N225",
    "Mexico - IPC": "^MXX",
    "Brazil - Ibovespa": "^BVSP",
}


def q2():
    print("\n" + "=" * 70)
    print("Q2 | World indices YTD (2026-01-01 .. 2026-08-21) vs S&P 500")
    print("=" * 70)

    start = "2026-01-01"
    end = "2026-08-21"

    def load_ytd():
        data = yf.download(
            list(WORLD_INDICES.values()),
            start=start,
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        closes = {}
        for ticker in WORLD_INDICES.values():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    closes[ticker] = data[ticker]["Close"].dropna()
                else:
                    closes[ticker] = data["Close"].dropna()
            except KeyError:
                print(f"  [warn] no data for {ticker}")
        return pd.DataFrame(closes)

    ytd = cached("q2_world_indices_ytd", load_ytd)
    ytd = ytd.sort_index()

    # only keep rows within the asked window
    ytd = ytd.loc[(ytd.index >= start) & (ytd.index <= end)]

    rows = []
    for label, ticker in WORLD_INDICES.items():
        s = ytd[ticker].dropna()
        if len(s) == 0:
            print(f"  [warn] {label} ({ticker}) empty")
            continue
        first = s.iloc[0]
        last = s.iloc[-1]
        ret = last / first - 1
        rows.append(
            {
                "country_index": label,
                "ticker": ticker,
                "first_close": round(float(first), 2),
                "last_close": round(float(last), 2),
                "first_date": s.index[0].date().isoformat(),
                "last_date": s.index[-1].date().isoformat(),
                "ytd_return": ret,
            }
        )
    df = pd.DataFrame(rows).sort_values("ytd_return", ascending=False)
    df["ytd_return_pct"] = (df["ytd_return"] * 100).round(2)
    df.to_csv(OUT / "q2_world_indices_ytd.csv", index=False)

    print("\n  YTD returns (as of last trading day <= 2026-08-21):")
    print(
        df[["country_index", "ytd_return_pct", "first_date", "last_date"]].to_string(
            index=False
        )
    )

    us_ret = df.loc[df["ticker"] == "^GSPC", "ytd_return"].iloc[0]
    better = df[df["ytd_return"] > us_ret]
    print(f"\n  US S&P 500 YTD: {fmt_pct(us_ret)}")
    print(f"  >>> Q2 answer: {len(better)} of {len(df)} indexes beat the S&P 500 YTD")
    names = better["country_index"].tolist()
    print(f"      outperformers: {names}")

    # bonus: 3 / 5 / 10 year performance
    def horizon_return(ticker, years):
        start_d = f"{date.today().year - years}-01-01"
        h = yf.Ticker(ticker).history(start=start_d, auto_adjust=False)
        h = h["Close"].dropna()
        if len(h) == 0:
            return np.nan
        return h.iloc[-1] / h.iloc[0] - 1

    if "--skip-bonus" not in sys.argv:
        print("\n  Bonus: 3 / 5 / 10 year returns (ignoring FX):")
        bonus_rows = []
        for label, ticker in WORLD_INDICES.items():
            r3, r5, r10 = (horizon_return(ticker, y) for y in (3, 5, 10))
            bonus_rows.append(
                {
                    "country_index": label,
                    "ticker": ticker,
                    "r3y": r3,
                    "r5y": r5,
                    "r10y": r10,
                }
            )
        bdf = pd.DataFrame(bonus_rows)
        for yrs, col in ((3, "r3y"), (5, "r5y"), (10, "r10y")):
            us_val = bdf.loc[bdf["ticker"] == "^GSPC", col].iloc[0]
            n = int((bdf[col] > us_val).sum())
            print(f"    {yrs}y: S&P 500 {fmt_pct(us_val)} | {n} indexes beat it")
        bdf.to_csv(OUT / "q2_bonus_long_term_returns.csv", index=False)


# ----------------------------------------------------------------------------
# Q3 - S&P 500 corrections
# ----------------------------------------------------------------------------
def q3():
    print("\n" + "=" * 70)
    print("Q3 | S&P 500 corrections (drawdown >= 5%) since 1950")
    print("=" * 70)

    def load_spx():
        t = yf.Ticker("^GSPC")
        h = t.history(start="1950-01-01", auto_adjust=False)
        return h

    spx = cached("q3_spx_daily", load_spx)
    close = spx["Close"].dropna()

    # 1) running all-time high
    highs = close.cummax()

    # 2) all-time high events (close sets a new record)
    new_high = close[close == highs]

    # 3) for each pair of consecutive ATHs: min close in between
    ath_dates = new_high.index
    corrections = []
    for i in range(len(ath_dates) - 1):
        d0, d1 = ath_dates[i], ath_dates[i + 1]
        seg = close.loc[d0:d1]
        low = seg.min()
        low_dt = seg.idxmin()
        dd = 1 - low / close.loc[d0]  # drawdown from the previous high
        corrections.append(
            {
                "peak_date": d0.date().isoformat(),
                "peak_close": round(float(close.loc[d0]), 2),
                "trough_date": low_dt.date().isoformat(),
                "trough_close": round(float(low), 2),
                "drawdown_pct": float(dd * 100),
                "duration_days": int((low_dt - d0).days),
            }
        )
    corr = pd.DataFrame(corrections)

    # 4) filter >= 5% drawdown
    sig = corr[corr["drawdown_pct"] >= 5.0].copy()
    sig.to_csv(OUT / "q3_corrections_all.csv", index=False)

    # 5) percentiles
    out = {
        "metric": ["drawdown_pct", "duration_days"],
    }
    for q in (0.25, 0.50, 0.75):
        out[f"q{q:.2f}"] = [
            round(float(sig["drawdown_pct"].quantile(q)), 2),
            round(float(sig["duration_days"].quantile(q)), 1),
        ]
    stat = pd.DataFrame(out)
    print("\n  25th / 50th / 75th percentiles of significant corrections (>=5%):")
    print(stat.to_string(index=False))
    stat.to_csv(OUT / "q3_percentiles.csv", index=False)

    med_dd = float(sig["drawdown_pct"].median())
    med_dur = float(sig["duration_days"].median())
    print(
        f"\n  >>> Q3 answer: median drawdown = {med_dd:.2f}%  (median duration = {med_dur:.0f} days)"
    )
    print(f"      number of significant corrections since 1950: {len(sig)}")

    # sanity: top-10 largest by drawdown
    top = sig.nlargest(10, "drawdown_pct")
    print("\n  sanity check - top 10 corrections by drawdown:")
    print(top.to_string(index=False))


# ----------------------------------------------------------------------------
# Q4 - AMZN earnings surprise analysis
# ----------------------------------------------------------------------------
def q4():
    print("\n" + "=" * 70)
    print("Q4 | AMZN earnings surprise -> 2-day return")
    print("=" * 70)

    ticker = "AMZN"
    tk = yf.Ticker(ticker)

    def load_earn():
        dates = tk.get_earnings_dates(limit=40)
        dates = dates.reset_index()
        dates.columns = [str(c).lower().replace(" ", "_") for c in dates.columns]
        return dates

    def load_px():
        return tk.history(start="2014-01-01", auto_adjust=False)

    earn = cached(f"q4_{ticker.lower()}_earnings", load_earn)
    px = cached(f"q4_{ticker.lower()}_prices", load_px)

    # normalize earnings table columns
    earn.columns = [str(c).strip() for c in earn.columns]
    date_col = [c for c in earn.columns if "earnings_date" in c or c == "date"][0]
    rep_col = [c for c in earn.columns if "reported_eps" in c][0]
    surp_col = [c for c in earn.columns if "surprise" in c][0]
    earn = earn.rename(
        columns={
            date_col: "earnings_date",
            rep_col: "reported_eps",
            surp_col: "surprise_pct",
        }
    )
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
    earn["surprise_pct"] = pd.to_numeric(earn["surprise_pct"], errors="coerce")
    has_report = earn["reported_eps"].notna()
    print(
        f"\n  earnings entries: {len(earn)} (past with data: {int(has_report.sum())})"
    )

    # 1) 3-trading-day windows: return = Close(Day3) / Close(Day1) - 1
    close = px["Close"].dropna().sort_index()
    ret2 = (close.shift(-2) / close - 1).rename("return_2d")  # aligned to Day1
    ret2 = ret2.to_frame("return_2d")
    ret2["day1"] = ret2.index
    ret2["day3_date"] = ret2.index + pd.Timedelta(
        days=0
    )  # placeholder; real day3 below

    # map each earnings date to a trading day: if the announcement date is not a
    # trading day, use the NEXT trading day (typical for after-hours releases)
    trading_days = close.index
    ann = pd.Series(
        earn["earnings_date"].values, index=earn["earnings_date"].values
    ).sort_index()
    day2 = []
    for d in earn["earnings_date"]:
        pos = trading_days.searchsorted(d)
        if pos >= len(trading_days):
            day2.append(pd.NaT)
        else:
            day2.append(trading_days[pos])
    earn["day2_trading"] = pd.to_datetime(day2)

    m = pd.merge(
        earn[["earnings_date", "day2_trading", "reported_eps", "surprise_pct"]],
        ret2.rename(columns={"index": "day1"}),
        left_on="day2_trading",
        right_on="day1",
        how="inner",
    )
    m["return_2d_pct"] = m["return_2d"] * 100

    # 2) filter positive surprises
    pos = m[m["surprise_pct"] > 0].copy()
    print("\n  earnings events matched to 2-day windows:")
    show = pos[
        [
            "earnings_date",
            "day2_trading",
            "reported_eps",
            "surprise_pct",
            "return_2d_pct",
        ]
    ]
    print(show.to_string(index=False))
    show.to_csv(OUT / "q4_amzn_positive_surprises.csv", index=False)

    if len(pos) > 0:
        med = float(pos["return_2d"].median())
        med_pct = med * 100
    else:
        med, med_pct = float("nan"), float("nan")

    print(
        f"\n  >>> Q4 answer: median 2-day return after POSITIVE surprises = {med_pct:.2f}% "
        f"(based on {len(pos)} events)"
    )

    # 3) correlation between surprise magnitude and 2-day return
    if len(m) >= 3:
        corr_all = m[["surprise_pct", "return_2d"]].corr().iloc[0, 1]
        corr_pos = (
            pos[["surprise_pct", "return_2d"]].corr().iloc[0, 1]
            if len(pos) >= 3
            else np.nan
        )
        print(
            f"      correlation(surprise, 2d-return): all events = {corr_all:+.3f} | "
            f"positive-only = {corr_pos:+.3f}"
        )
        pd.DataFrame(
            {
                "metric": ["corr_all_events", "corr_positive_only"],
                "value": [corr_all, corr_pos],
            }
        ).to_csv(OUT / "q4_correlations.csv", index=False)
        # keep full matched table for Power BI
        m.to_csv(OUT / "q4_amzn_earnings_returns_full.csv", index=False)


# ----------------------------------------------------------------------------
# run everything
# ----------------------------------------------------------------------------
def main():
    q1()
    q2()
    q3()
    q4()
    print("\nDone. Data + outputs written to:", DATA.resolve())


if __name__ == "__main__":
    main()
