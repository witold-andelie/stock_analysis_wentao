"""Blank environment settings use defaults without enabling broker opt-ins."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

from driftrev import broker, broker_ibkr, config, scheduler


def _run_python(tmp_path: Path, code: str, **settings: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["DRIFTREV_DATA_DIR"] = str(tmp_path / "data")
    env["DRIFTREV_MARKET"] = "us"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name, value in settings.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run([sys.executable, "-c", dedent(code)], env=env, capture_output=True, text=True)


def test_empty_sec_user_agent_uses_default(tmp_path):
    for value in (None, "", "   "):
        result = _run_python(tmp_path, """
            from driftrev import config

            assert config.USER_AGENT == "driftrev-research research@example.org"
        """, SEC_USER_AGENT=value)
        assert result.returncode == 0, result.stderr
    result = _run_python(tmp_path, """
        from driftrev import config

        assert config.USER_AGENT == "Jane Doe jane@example.com"
    """, SEC_USER_AGENT=" Jane Doe jane@example.com ")
    assert result.returncode == 0, result.stderr


def test_empty_data_dir_and_market_use_defaults(tmp_path):
    for value in (None, "", "   "):
        result = _run_python(tmp_path, """
            from driftrev import config

            assert config.DATA_ROOT == config.PROJECT_ROOT / "data"
            assert config.MARKET.code == "us"
        """, DRIFTREV_DATA_DIR=value, DRIFTREV_MARKET=value)
        assert result.returncode == 0, result.stderr


def test_empty_numeric_settings_use_defaults(tmp_path):
    for value in (None, "", "   "):
        result = _run_python(tmp_path, """
            from driftrev import config

            assert config.MAX_TICKERS == 0
            assert config.SEC_MAX_RPS == 8.0
            assert config.START_DATE == "2014-01-01"
        """, DRIFTREV_MAX_TICKERS=value, SEC_MAX_RPS=value, DRIFTREV_START=value)
        assert result.returncode == 0, result.stderr


def test_schedule_time_empty(monkeypatch):
    monkeypatch.delenv("DRIFTREV_SCHEDULE_TIME", raising=False)
    assert scheduler.schedule_time() == config.MARKET.run_time
    for value in ("", "   "):
        monkeypatch.setenv("DRIFTREV_SCHEDULE_TIME", value)
        assert scheduler.schedule_time() == config.MARKET.run_time
    for value in ("07:15", " 07:15 "):
        monkeypatch.setenv("DRIFTREV_SCHEDULE_TIME", value)
        assert scheduler.schedule_time() == "07:15"


def test_ibkr_empty_connection_settings(monkeypatch):
    calls = []

    class IB:
        def connect(self, host, port, clientId, timeout, readonly, account):
            calls.append(("connect", host, port, clientId, timeout, readonly, account))

        def managedAccounts(self):
            return ["DU1234567"]

        def disconnect(self):
            calls.append(("disconnect",))

    monkeypatch.setattr(broker_ibkr, "_ib_api", lambda: SimpleNamespace(IB=IB))
    monkeypatch.setenv("IBKR_ENABLED", "true")
    monkeypatch.setenv("IBKR_PAPER", "true")
    for value in (None, "", "   "):
        for name in ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID", "IBKR_ACCOUNT"):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        assert broker_ibkr.configured()
        with broker_ibkr._connect() as (_, account):
            assert account == "DU1234567"
        assert calls[-2:] == [("connect", "127.0.0.1", 4002, 17, 10, False, ""), ("disconnect",)]
    for name, value in (("IBKR_HOST", " gateway "), ("IBKR_PORT", " 7497 "),
                        ("IBKR_CLIENT_ID", " 42 "), ("IBKR_ACCOUNT", " DU7654321 ")):
        monkeypatch.setenv(name, value)
    assert broker_ibkr.configured()
    with broker_ibkr._connect() as (_, account):
        assert account == "DU7654321"
    assert calls[-2:] == [("connect", "gateway", 7497, 42, 10, False, "DU7654321"), ("disconnect",)]


def test_opt_ins_still_off_when_empty(monkeypatch):
    monkeypatch.setenv("DRIFTREV_ALLOW_ORDERS", "")
    monkeypatch.setenv("IBKR_ENABLED", "")
    monkeypatch.setenv("IBKR_PORT", "")
    assert broker._allowed() is False
    assert broker_ibkr._allowed() is False
    assert broker_ibkr.configured() is False
