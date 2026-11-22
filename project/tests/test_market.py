"""Market definitions, isolated environment selection and entry-point guards."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from zoneinfo import ZoneInfo

import pytest

from driftrev import config


def _run_python(tmp_path: Path, code: str, market: str | None = None,
                schedule: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["DRIFTREV_DATA_DIR"] = str(tmp_path / "data")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    for name, value in (("DRIFTREV_MARKET", market), ("DRIFTREV_SCHEDULE_TIME", schedule)):
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run([sys.executable, "-c", dedent(code)], env=env, capture_output=True, text=True)


def test_default_market_is_us(tmp_path):
    result = _run_python(tmp_path, """
        from driftrev import config

        assert config.MARKET.code == "us"
        assert config.DATA_DIR == config.DATA_ROOT
        assert config.WAREHOUSE == config.DATA_ROOT / "warehouse.duckdb"
    """)
    assert result.returncode == 0, result.stderr


def test_us_market_matches_existing_constants():
    market = config.MARKETS["us"]
    assert market.index_set == config.UNIVERSE_INDEXES
    assert dict(market.benchmarks) == {ticker: config.MARKET_TICKERS[ticker] for ticker in ("SPY", "IWM", "MDY")}
    costs = config.CapCosts()
    assert market.cost_bps("large") == costs.large
    assert market.cost_bps("mid") == costs.mid
    assert market.cost_bps("small") == costs.small
    assert market.timezone == "America/New_York"
    assert market.run_time == "22:30"
    assert market.implemented is True


def test_de_market_definition():
    market = config.MARKETS["de"]
    assert market.code == "de"
    assert market.name == "German equities (Xetra: DAX + MDAX + SDAX)"
    assert market.exchange_mic == "XETR"
    assert market.timezone == "Europe/Berlin"
    assert market.run_time == "18:30"
    assert market.currency == "EUR"
    assert market.index_set == ("dax", "mdax", "sdax")
    assert market.cap_bucket_by_index == (("dax", "large"), ("mdax", "mid"), ("sdax", "small"))
    assert market.benchmarks == (("^GDAXI", "dax"), ("^MDAXI", "mdax"), ("^SDAXI", "sdax"))
    assert market.cost_bps_one_way == (("large", 10.0), ("mid", 20.0), ("small", 35.0))
    assert market.data_subdir == "de"
    assert market.implemented is True
    assert market.cap_bucket("sdax") == "small"
    assert market.cap_bucket("sp500") is None
    assert market.cost_bps("unknown") == 20.0


def test_get_market_normalises_and_rejects():
    assert config.get_market(" DE ").code == "de"
    with pytest.raises(ValueError, match="valid: de, us"):
        config.get_market("xx")


def test_de_selection_uses_own_data_dir(tmp_path):
    result = _run_python(tmp_path, """
        from driftrev import config

        assert config.MARKET.code == "de"
        assert config.DATA_DIR == config.DATA_ROOT / "de"
    """, market="de")
    assert result.returncode == 0, result.stderr


def test_invalid_market_env_fails_at_import(tmp_path):
    result = _run_python(tmp_path, "import driftrev.config", market="xx")
    assert result.returncode != 0
    assert "unknown market 'xx'" in result.stderr


def test_require_implemented_market(monkeypatch):
    assert config.require_implemented_market(config.MARKETS["us"]) is config.MARKETS["us"]
    market = replace(config.MARKETS["de"], code="xx", implemented=False)
    monkeypatch.setattr(config, "MARKET", market)
    with pytest.raises(config.MarketNotImplementedError, match="R7.2"):
        config.require_implemented_market()
    with pytest.raises(config.MarketNotImplementedError, match="R7.2"):
        config.require_implemented_market(market)


def test_pipeline_refuses_unimplemented_market(tmp_path):
    result = _run_python(tmp_path, """
        from dataclasses import replace
        from unittest.mock import patch

        from driftrev import config, pipeline

        with patch.object(config, "MARKET", replace(config.MARKETS["de"], code="xx", implemented=False)):
            try:
                pipeline.run_pipeline(mode="update")
            except config.MarketNotImplementedError as exc:
                print(type(exc).__name__)
                raise SystemExit(0)
        raise AssertionError("pipeline accepted an unimplemented market")
    """, market="de")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MarketNotImplementedError"
    assert not list((tmp_path / "data").rglob("warehouse.duckdb"))


def test_scheduler_uses_market_time(tmp_path):
    result = _run_python(tmp_path, """
        from driftrev import scheduler

        assert scheduler.TZ.key == "Europe/Berlin"
        assert scheduler.schedule_time() == "18:30"
    """, market="de")
    assert result.returncode == 0, result.stderr
    result = _run_python(tmp_path, """
        from driftrev import scheduler

        print(scheduler.schedule_time())
    """, market="de", schedule="07:15")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "07:15"


def test_next_run_in_berlin():
    from driftrev.scheduler import next_run

    berlin = ZoneInfo("Europe/Berlin")
    result = next_run(datetime(2026, 11, 20, 19, 0, tzinfo=berlin), "18:30")
    assert result == datetime(2026, 11, 23, 18, 30, tzinfo=berlin)
    assert result.tzinfo == berlin
