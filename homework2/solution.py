"""
Stock Market Analytics Zoomcamp - Module 2 Homework (2026 cohort) solution.

Covers (spec: cohorts/2026/homework2.md):
  Q1: Withdrawn IPOs by Company Type (iposcoop.com/ipos-recently-filed)
      -> class with highest total withdrawn value + value in $M
  Q2: Median Sharpe ratio (as of 2026-09-11) for 2025 IPOs with
      Offer Date < 2025-09-01 and non-zero Return
      growth_252d = Close/Close.shift(252),
      volatility = Close.rolling(30).std()*sqrt(252),
      Sharpe = (growth_252d - 0.05)/volatility
  Q3: Fixed-months holding strategy: future_growth_1m..12m
      (21/42/.../252 trading days), median-max month
  Q4: RSI<30 oversold strategy on precomputed parquet
      (gdown id 1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-), 2000-01-01..2025-06-01,
      net = 1000*(growth_future_30d-1).sum()
  Q5: free text (see README.md)

Raw downloads are cached under data/raw/ as parquet; analysis outputs
go to data/outputs/ (CSV + answers.json). Re-runs are offline-friendly.
"""

from __future__ import annotations

import json
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------------------------------------------------------
# paths / helpers
# ----------------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "data"
RAW = DATA / "raw"
OUT = DATA / "outputs"
for p in (RAW, OUT):
    p.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

ANSWERS: dict = {}


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


def _num(x):
    """Round floats for answers.json; map NaN/inf -> None (valid JSON)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 4)


def parse_price(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in ("", "-", "nan", "NaN", "None"):
        return np.nan
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def classify_company(name: str) -> str:
    """Company Type per HW2 spec order (case-insensitive substring match)."""
    n = str(name).lower()
    if "technologies" in n:
        return "Technologies"
    if "acquisition corp" in n or "acquisition corporation" in n or "corp" in n:
        return "Acquisition Corp"
    if "inc" in n or "incorporated" in n:
        return "Inc."
    if "group" in n:
        return "Group"
    if "ltd" in n or "limited" in n:
        return "Limited"
    if "holdings" in n or "holding" in n:
        return "Holdings"
    return "Other"


# ----------------------------------------------------------------------------
# Q1 - Withdrawn IPOs by Company Type
# ----------------------------------------------------------------------------
def q1():
    print("=" * 70)
    print("Q1 | Withdrawn IPOs by Company Type (iposcoop recently-filed)")
    print("=" * 70)

    def load_recently_filed():
        r = requests.get(
            "https://www.iposcoop.com/ipos-recently-filed/",
            headers=HEADERS,
            timeout=60,
        )
        r.raise_for_status()
        return pd.read_html(StringIO(r.text))[0]

    filed = cached("q1_iposcoop_recently_filed", load_recently_filed)
    print(f"  recently-filed rows: {len(filed)}")

    w = filed[filed["Expected To Trade"] == "Withdrawn"].copy()
    # spec scope: withdrawn before Sep 11, 2026
    w["File Date"] = pd.to_datetime(w["File Date"], errors="coerce")
    w = w[w["File Date"] < "2026-09-11"].copy()
    print(f"  withdrawn rows (before 2026-09-11): {len(w)}")

    w["Company Type"] = w["Company"].apply(classify_company)
    w["low"] = w["Price Low"].apply(parse_price)
    w["high"] = w["Price High"].apply(parse_price)
    w["Avg_price"] = (w["low"] + w["high"]) / 2
    w["Shares (millions)"] = pd.to_numeric(w["Shares (millions)"], errors="coerce")
    est = (
        w["Est $ Vol (millions)"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    w["est_vol"] = pd.to_numeric(est, errors="coerce")
    w["Shares_offered_value"] = (w["Shares (millions)"] * w["Avg_price"]).where(
        (w["Shares (millions)"] * w["Avg_price"]).notna(), w["est_vol"]
    )

    agg = (
        w.groupby("Company Type")["Shares_offered_value"]
        .sum()
        .reset_index()
        .sort_values("Shares_offered_value", ascending=False)
    )
    agg.to_csv(OUT / "q1_withdrawn_by_company_type.csv", index=False)
    w.to_csv(OUT / "q1_withdrawn_details.csv", index=False)

    print("\n  Total withdrawn value by Company Type ($M):")
    print(agg.to_string(index=False))
    best = agg.iloc[0]
    print(
        f"\n  >>> Q1 answer: '{best['Company Type']}' highest total "
        f"= ${float(best['Shares_offered_value']):.2f}M"
    )
    ANSWERS["q1"] = {
        "class": str(best["Company Type"]),
        "total_value_m": _num(float(best["Shares_offered_value"])),
        "n_withdrawn": int(len(w)),
    }


# ----------------------------------------------------------------------------
# Q2/Q3 shared: 2025 IPO list + yfinance OHLCV
# ----------------------------------------------------------------------------
def load_2025_pricings() -> pd.DataFrame:
    def _dl():
        r = requests.get(
            "https://www.iposcoop.com/2025-pricings/", headers=HEADERS, timeout=60
        )
        r.raise_for_status()
        return pd.read_html(StringIO(r.text))[0]

    return cached("q2_iposcoop_2025_pricings", _dl)


def filtered_2025_tickers() -> pd.DataFrame:
    df = load_2025_pricings()
    print(f"  2025 pricings rows: {len(df)}")
    df["Offer Date"] = pd.to_datetime(df["Offer Date"], errors="coerce")
    ret = pd.to_numeric(
        df["Return"].astype(str).str.replace("%", "", regex=False), errors="coerce"
    )
    df["_ret"] = ret
    filt = df[(df["Offer Date"] < "2025-09-01") & (df["_ret"] != 0)].copy()
    print(f"  filtered (Offer Date < 2025-09-01, Return != 0): {len(filt)}")
    filt.to_csv(OUT / "q2_2025_ipos_filtered.csv", index=False)
    return filt


def download_ipo_prices(tickers: list[str]) -> pd.DataFrame:
    """Long OHLCV frame with one row per (ticker, date). Cached."""

    def _dl():
        # single bulk download (fast); missing/delisted tickers come back empty
        data = yf.download(
            tickers,
            period="max",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        frames = []
        if isinstance(data.columns, pd.MultiIndex):
            have = set(data.columns.get_level_values(0))
            for t in tickers:
                if t not in have:
                    continue
                try:
                    px = data[t].dropna(how="all").copy()
                except KeyError:
                    continue
                if px.empty or "Close" not in px.columns:
                    continue
                px["ticker"] = t
                px["date"] = px.index
                frames.append(px.reset_index(drop=True))
                time.sleep(0.05)
        else:  # single ticker fallback
            px = data.copy()
            px["ticker"] = tickers[0]
            px["date"] = px.index
            frames.append(px.reset_index(drop=True))
        if not frames:
            return pd.DataFrame()
        long = pd.concat(frames, ignore_index=True)
        long["date"] = pd.to_datetime(long["date"])
        return long.sort_values(["ticker", "date"]).reset_index(drop=True)

    # cache key depends on ticker set size; store generic file
    path = RAW / "q2_q3_ipo_prices.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        have = set(df["ticker"].unique())
        missing = [t for t in tickers if t not in have]
        print(f"  [cache] q2_q3_ipo_prices.parquet ({len(df)} rows, "
              f"{len(have)} tickers, {len(missing)} tickers missing from cache)")
        if not missing:
            return df
        print(f"  downloading {len(missing)} uncached tickers ...")
        extra = _dl.__wrapped__() if hasattr(_dl, "__wrapped__") else None
        # simple: re-download everything missing in one go
        data = yf.download(
            missing, period="max", interval="1d", group_by="ticker",
            auto_adjust=False, threads=True,
        )
        frames = []
        if isinstance(data.columns, pd.MultiIndex):
            have2 = set(data.columns.get_level_values(0))
            for t in missing:
                if t not in have2:
                    continue
                try:
                    px = data[t].dropna(how="all").copy()
                except KeyError:
                    continue
                if px.empty or "Close" not in px.columns:
                    continue
                px["ticker"] = t
                px["date"] = px.index
                frames.append(px.reset_index(drop=True))
        if frames:
            df = pd.concat([df, *frames], ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
            df.to_parquet(path)
        return df
    df = _dl()
    df.to_parquet(path)
    print(f"  [fetch] q2_q3_ipo_prices.parquet saved ({len(df)} rows)")
    return df


def q2():
    print("\n" + "=" * 70)
    print("Q2 | Median Sharpe ratio (as of 2026-09-11) for 2025 IPOs")
    print("=" * 70)

    filt = filtered_2025_tickers()
    tickers = filt["Symbol"].astype(str).str.strip().unique().tolist()
    px = download_ipo_prices(tickers)
    print(f"  price rows: {len(px)}, tickers with data: {px['ticker'].nunique()}")

    px = px.sort_values(["ticker", "date"]).copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    g = px.groupby("ticker")["Close"]
    px["growth_252d"] = g.transform(lambda s: s / s.shift(252))
    px["volatility"] = g.transform(lambda s: s.rolling(30).std() * np.sqrt(252))
    px["Sharpe"] = (px["growth_252d"] - 0.05) / px["volatility"]

    asof = px[px["date"] == "2026-09-11"].copy()
    # if 2026-09-11 missing for some (holiday/delisted), fall back is NOT used:
    # spec pins the exact trading day 2026-09-11 (a Friday).
    print(f"  rows on 2026-09-11: {len(asof)}")
    valid = asof.dropna(subset=["growth_252d"])
    print(f"  with growth_252d defined (reached 252d milestone): {len(valid)}")

    desc = asof[["growth_252d", "Sharpe"]].describe()
    print("\n  describe() on 2026-09-11:")
    print(desc.to_string())
    desc.to_csv(OUT / "q2_sharpe_describe_2026-09-11.csv")

    med = float(asof["Sharpe"].median())
    mean = float(asof["Sharpe"].mean())
    med_g = float(asof["growth_252d"].median())
    mean_g = float(asof["growth_252d"].mean())
    print(f"\n  growth_252d median={med_g:.4f} mean={mean_g:.4f}")
    print(f"  >>> Q2 answer: median Sharpe (2026-09-11) = {med:.4f} (mean={mean:.4f})")

    top_g = asof.nlargest(10, "growth_252d")[["ticker", "growth_252d", "Sharpe"]]
    top_s = asof.nlargest(10, "Sharpe")[["ticker", "growth_252d", "Sharpe"]]
    top_g.to_csv(OUT / "q2_top10_growth.csv", index=False)
    top_s.to_csv(OUT / "q2_top10_sharpe.csv", index=False)
    asof.to_csv(OUT / "q2_asof_2026-09-11.csv", index=False)

    ANSWERS["q2"] = {
        "asof": "2026-09-11",
        "n_tickers_filtered": int(len(filt)),
        "n_tickers_with_data": int(px["ticker"].nunique()),
        "n_rows_asof": int(len(asof)),
        "n_growth252_defined": int(valid["ticker"].nunique()),
        "median_sharpe": _num(med),
        "mean_sharpe": _num(mean),
        "median_growth_252d": _num(med_g),
        "mean_growth_252d": _num(mean_g),
    }
    return px


def q3(px: pd.DataFrame | None = None):
    print("\n" + "=" * 70)
    print("Q3 | Fixed-months holding strategy (median-max month)")
    print("=" * 70)

    if px is None:
        filt = filtered_2025_tickers()
        tickers = filt["Symbol"].astype(str).str.strip().unique().tolist()
        px = download_ipo_prices(tickers)

    px = px.sort_values(["ticker", "date"]).copy()
    px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
    for m in range(1, 13):
        px[f"future_growth_{m}m"] = px.groupby("ticker")["Close"].transform(
            lambda s, k=m: s.shift(-21 * k) / s
        )

    cols = [f"future_growth_{m}m" for m in range(1, 13)]
    first = px.groupby("ticker").first().reset_index()
    # inner join min_date records with full growth dataset on (ticker, date)
    min_dates = px.groupby("ticker")["date"].min().reset_index()
    entry = pd.merge(min_dates, px, on=["ticker", "date"], how="inner")
    print(f"  IPO entry rows (one per ticker): {len(entry)} (tickers: {entry['ticker'].nunique()})")

    desc = entry[cols].describe()
    print("\n  describe() of 12 future-growth columns:")
    print(desc.to_string())
    desc.to_csv(OUT / "q3_future_growth_describe.csv")

    medians = entry[cols].median()
    means = entry[cols].mean()
    comp = pd.DataFrame({"median": medians, "mean": means})
    comp.to_csv(OUT / "q3_median_mean_by_month.csv")
    print("\n  median / mean by holding period:")
    print(comp.to_string())

    best_m = int(medians.idxmax().split("_")[2].replace("m", ""))
    best_med = float(medians.max())
    print(f"\n  >>> Q3 answer: optimal holding = {best_m} month(s), "
          f"max median growth = {best_med:.4f}")
    entry.to_csv(OUT / "q3_ipo_entry_growth.csv", index=False)

    ANSWERS["q3"] = {
        "optimal_months": best_m,
        "max_median_growth": _num(best_med),
        "medians": {str(k): _num(v) for k, v in medians.items()},
        "n_ipos": int(entry["ticker"].nunique()),
    }


# ----------------------------------------------------------------------------
# Q4 - RSI oversold strategy on precomputed parquet
# ----------------------------------------------------------------------------
def q4():
    print("\n" + "=" * 70)
    print("Q4 | RSI<30 oversold strategy (2000-01-01..2025-06-01)")
    print("=" * 70)

    def _dl():
        try:
            import gdown
        except ImportError as e:
            raise RuntimeError("gdown is required for Q4: pip install gdown") from e
        file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
        tmp = RAW / "_q4_tmp_download.parquet"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", str(tmp), quiet=False
        )
        df = pd.read_parquet(tmp, engine="pyarrow")
        tmp.unlink(missing_ok=True)
        return df

    df = cached("q4_precomputed", _dl)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    sel = df[
        (df["rsi"] < 30)
        & (df["Date"] >= "2000-01-01")
        & (df["Date"] <= "2025-06-01")
    ].copy()
    print(f"  oversold trades (RSI<30): {len(sel)}")
    avg = float((sel["growth_future_30d"] - 1).mean())
    win = float((sel["growth_future_30d"] > 1).mean())
    net = float(1000 * (sel["growth_future_30d"] - 1).sum())
    print(f"  avg 30d return: {avg * 100:.2f}% | win rate: {win * 100:.2f}%")
    print(f"  >>> Q4 answer: net income = ${net:,.2f} (~${net / 1000:.1f}K)")

    sel[["Ticker", "Date", "rsi", "growth_future_30d"]].to_csv(
        OUT / "q4_oversold_trades.csv", index=False
    )
    sel["growth_future_30d"].describe().to_csv(OUT / "q4_return_describe.csv")

    ANSWERS["q4"] = {
        "rsi_threshold": 30,
        "n_trades": int(len(sel)),
        "avg_30d_return": _num(avg),
        "win_rate": _num(win),
        "net_income_usd": _num(net),
        "net_income_k": _num(net / 1000),
    }


# ----------------------------------------------------------------------------
def main():
    q1()
    px = None
    try:
        px = q2()
    except Exception as e:
        print(f"  [!] Q2 failed: {e}")
    try:
        q3(px)
    except Exception as e:
        print(f"  [!] Q3 failed: {e}")
    try:
        q4()
    except Exception as e:
        print(f"  [!] Q4 failed: {e}")

    with open(OUT / "answers.json", "w", encoding="utf-8") as f:
        json.dump(ANSWERS, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("\nDone. Answers written to:", (OUT / "answers.json").resolve())
    print(json.dumps(ANSWERS, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
