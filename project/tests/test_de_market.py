"""Offline German market ingestion and point-in-time realised volatility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftrev import config, de_market, indicators, prices, store

FIXTURE = Path(__file__).parent / "fixtures" / "bundesbank_bund10y_sample.csv"


@pytest.fixture
def de_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "de"
    monkeypatch.setattr(config, "MARKET", config.MARKETS["de"])
    for name, path in (
        ("DATA_DIR", data_dir),
        ("WAREHOUSE", data_dir / "warehouse.duckdb"),
        ("CACHE_DIR", data_dir / "cache"),
        ("SNAPSHOT_DIR", data_dir / "snapshot"),
        ("ARTIFACTS_DIR", data_dir / "artifacts"),
    ):
        monkeypatch.setattr(config, name, path)
    return data_dir


def test_parse_bundesbank_fixture(de_env):
    df = de_market.parse_bundesbank_csv(FIXTURE.read_text(encoding="utf-8"), "bund10y")
    assert list(df.columns) == ["date", "value", "series"]
    assert len(df) == 15
    assert df["date"].dtype == np.dtype("datetime64[ns]")
    assert df["value"].dtype == np.dtype("float64")
    assert df.iloc[0]["date"] == pd.Timestamp("2014-01-02")
    assert df.iloc[0]["value"] == 2.11
    assert df.iloc[-1]["date"] == pd.Timestamp("2026-11-20")
    assert df.iloc[-1]["value"] == 3.39
    assert df["series"].eq("bund10y").all()
    assert not df.isna().any().any()
    assert df["date"].is_monotonic_increasing
    assert df.index.equals(pd.RangeIndex(15))


def test_parse_bundesbank_crlf_and_empty(de_env):
    text = FIXTURE.read_text(encoding="utf-8")
    expected = de_market.parse_bundesbank_csv(text, "bund10y")
    actual = de_market.parse_bundesbank_csv(text.replace("\n", "\r\n"), "bund10y")
    pd.testing.assert_frame_equal(actual, expected)
    metadata = "\n".join(text.splitlines()[:9])
    empty = de_market.parse_bundesbank_csv(metadata, "bund10y")
    assert empty.empty
    assert list(empty.columns) == ["date", "value", "series"]
    assert empty["date"].dtype == np.dtype("datetime64[ns]")
    assert empty["value"].dtype == np.dtype("float64")


def test_bundesbank_url(de_env):
    series_key = next(iter(config.DE_RATE_SERIES))
    assert de_market.bundesbank_url(series_key, "2014-01-01") == (
        "https://api.statistiken.bundesbank.de/rest/data/"
        "BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"
        "?format=csv&lang=en&startPeriod=2014-01-01"
    )


def test_de_functions_refuse_us_market(de_env, monkeypatch):
    monkeypatch.setattr(config, "MARKET", config.MARKETS["us"])

    def unexpected_download(*args, **kwargs):
        pytest.fail("German ingestion attempted a download for the US market")

    monkeypatch.setattr(de_market, "fetch_bundesbank", unexpected_download)
    monkeypatch.setattr(prices, "_download", unexpected_download)
    for update in (de_market.update_de_rates, de_market.update_de_market):
        with pytest.raises(RuntimeError, match=r"DRIFTREV_MARKET=de \(active market: us\)"):
            update()
    assert not config.WAREHOUSE.exists()
    assert not de_env.exists()


def test_update_de_rates_incremental(de_env, monkeypatch):
    sample = de_market.parse_bundesbank_csv(FIXTURE.read_text(encoding="utf-8"), "bund10y")
    starts = []

    def fake_fetch(series_key: str, start: str) -> pd.DataFrame:
        assert series_key in config.DE_RATE_SERIES
        starts.append(start)
        return sample.loc[sample["date"] >= pd.Timestamp(start)].copy()

    monkeypatch.setattr(de_market, "fetch_bundesbank", fake_fetch)
    assert de_market.update_de_rates() == {"table": "macro", "rows_written": 15}
    assert starts == [config.START_DATE]
    assert len(store.read_table("macro")) == 15
    assert de_market.update_de_rates() == {"table": "macro", "rows_written": 5}
    expected_start = (pd.Timestamp("2026-11-20") - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    assert expected_start == "2026-09-21"
    assert starts == [config.START_DATE, expected_start]
    stored = store.read_table("macro", "order by date")
    assert len(stored) == 15
    assert not stored.duplicated(["series", "date"]).any()
    pd.testing.assert_frame_equal(stored, sample)


def test_update_de_market_uses_benchmarks(de_env, monkeypatch):
    requested = []

    def fake_download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        requested.extend(tickers)
        assert start == config.START_DATE
        return pd.DataFrame([
            {"ticker": ticker, "date": date, "open": 100.0, "high": 102.0,
             "low": 99.0, "close": 101.0, "volume": 1000.0, "dividends": 0.0, "splits": 0.0}
            for ticker in tickers for date in pd.bdate_range(start, periods=3)
        ])

    monkeypatch.setattr(prices, "_download", fake_download)
    assert de_market.update_de_market() == {
        "table": "market", "rows_written": 9, "failed": [], "resplit": [],
    }
    assert requested == ["^GDAXI", "^MDAXI", "^SDAXI"]
    stored = store.read_table("market")
    assert len(stored) == 9
    assert stored.groupby("ticker").size().to_dict() == {ticker: 3 for ticker in requested}


def test_realised_vol_known_value():
    returns = np.tile([0.01, -0.01], 20)
    close = pd.Series(100 * np.exp(np.r_[0, returns.cumsum()]))
    vol = indicators.realised_vol(close, 20)
    expected = 100 * np.sqrt(252) * 0.01 * np.sqrt(20 / 19)
    assert vol.iloc[:20].isna().all()
    assert vol.iloc[20:].notna().all()
    np.testing.assert_allclose(vol.iloc[20:], expected, atol=1e-9, rtol=0)


def test_realised_vol_point_in_time():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(rng.normal(0, 0.01, 150).cumsum()))
    pd.testing.assert_series_equal(indicators.realised_vol(close.iloc[:100]),
                                   indicators.realised_vol(close).iloc[:100])


def test_realised_vol_term():
    returns = np.tile([0.01, -0.01], 60)
    close = pd.Series(100 * np.exp(np.r_[0, returns.cumsum()]))
    term = indicators.realised_vol_term(close)
    expected = np.sqrt(20 / 19) / np.sqrt(60 / 59)
    assert term.iloc[:60].isna().all()
    assert term.iloc[60:].notna().all()
    np.testing.assert_allclose(term.iloc[60:], expected, atol=1e-9, rtol=0)
