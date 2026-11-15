"""SD7: retention deletes exactly the expired files; the static report builds from artifacts."""

import json
import os
import time

import pytest

from driftrev import config, housekeeping, report


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for name, sub in (("DATA_DIR", ""), ("CACHE_DIR", "cache"), ("ARTIFACTS_DIR", "artifacts"),
                      ("HISTORY_DIR", "artifacts/history"), ("LOG_DIR", "logs"), ("SITE_DIR", "site"),
                      ("SNAPSHOT_DIR", "snapshot")):
        p = tmp_path / sub if sub else tmp_path
        p.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, p)
    monkeypatch.setattr(config, "WAREHOUSE", tmp_path / "none.duckdb")
    return tmp_path


def _touch(p, age_days):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))
    return p


def test_retention_deletes_only_expired(data_dir):
    edgar = config.CACHE_DIR / "edgar"
    old_root = _touch(edgar / "CIK0000000001.json", 3)
    new_root = _touch(edgar / "CIK0000000002.json", 0.5)
    old_page = _touch(edgar / "CIK0000000001-submissions-001.json", 200)
    new_page = _touch(edgar / "CIK0000000002-submissions-001.json", 10)
    old_day = config.HISTORY_DIR / time.strftime("%Y-%m-%d", time.localtime(time.time() - 90 * 86400))
    new_day = config.HISTORY_DIR / time.strftime("%Y-%m-%d")
    _touch(old_day / "signals_latest.parquet", 90)
    _touch(new_day / "signals_latest.parquet", 0)
    old_txt = _touch(config.CACHE_DIR / "filing_text" / "00" / "a.txt.gz", 40)
    new_txt = _touch(config.CACHE_DIR / "filing_text" / "00" / "b.txt.gz", 5)
    z_old = _touch(data_dir / "driftrev_snapshot_old.zip", 10)
    z_new = _touch(data_dir / "driftrev_snapshot.zip", 1)
    (config.ARTIFACTS_DIR / "run_log.jsonl").write_text("\n".join("{}" for _ in range(1500)) + "\n")

    dry = housekeeping.run(dry_run=True)
    assert dry["expired"] == 5 and old_root.exists()          # dry run deletes nothing

    out = housekeeping.run()
    assert out["expired"] == 5 and not old_txt.exists() and new_txt.exists()
    assert not old_root.exists() and not old_page.exists() and not old_day.exists() and not z_old.exists()
    assert new_root.exists() and new_page.exists() and new_day.exists() and z_new.exists()
    assert len((config.ARTIFACTS_DIR / "run_log.jsonl").read_text().splitlines()) == config.SETTINGS.retention.run_log_max_lines


def test_report_builds_site(art_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SITE_DIR", tmp_path / "site")
    (art_dir / "metrics.json").write_text(json.dumps({
        "data_through": "2026-11-13", "regime": {"state": "normal", "points": 0},
        "model": {"name": "rf", "threshold": 0.55, "test_auc": 0.56},
        "strategies": [{"name": "model_top5", "cagr": 0.12, "sharpe": 1.1, "max_drawdown": -0.1, "n_trades": 50}],
        "benchmarks": [{"name": "SPY_buy_hold", "cagr": 0.15, "sharpe": 1.2, "max_drawdown": -0.18}],
        "guide": {"verdict": "release does NOT add value"}}))
    out = report.build()
    site = tmp_path / "site"
    assert out["signals"] == 3
    html = (site / "index.html").read_text()
    assert "Drift or Reversal" in html and "AAA" in html and "release does NOT add value" in html
    assert json.loads((site / "signals.json").read_text())[0]["ticker"] == "AAA"
    assert "| ticker" in (site / "summary.md").read_text() or "ticker" in (site / "summary.md").read_text()
