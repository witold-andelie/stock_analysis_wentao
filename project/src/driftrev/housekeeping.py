"""Housekeeping: delete expired local data files according to `config.Retention` (OPM SD7).

What gets deleted (raw warehouse tables are never pruned, because the full history is the dataset):

* EDGAR root submission JSONs (`CIK##########.json`), which every refresh re-downloads anyway
* EDGAR history pages (`CIK...-submissions-NNN.json`) older than `edgar_history_days`
* daily signal archives `artifacts/history/YYYY-MM-DD/` older than `signal_history_days`
* snapshot zips beyond the newest `snapshot_zips_keep`
* run-log lines beyond `run_log_max_lines`, and `logs/*.log` above `log_max_bytes` (keeps the tail)
* then a DuckDB CHECKPOINT so freed pages are reclaimed

    python -m driftrev.housekeeping              # delete
    python -m driftrev.housekeeping --dry-run    # only list what would go
    python -m driftrev.housekeeping --print-github   # retention numbers for cleanup.yml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

from . import artifacts, config, store

log = logging.getLogger(__name__)
R = config.SETTINGS.retention
_ROOT = re.compile(r"^CIK\d{10}\.json$")
_PAGE = re.compile(r"^CIK\d{10}-submissions-\d+\.json$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _age_days(p: Path, now: float) -> float:
    return (now - p.stat().st_mtime) / 86400


def plan(now: float | None = None) -> list[tuple[Path, str]]:
    """Return [(path, reason)] of everything that is expired. Pure: deletes nothing."""
    now = now or time.time()
    out: list[tuple[Path, str]] = []
    edgar = config.CACHE_DIR / "edgar"
    if edgar.exists():
        for p in edgar.iterdir():
            if _ROOT.match(p.name) and _age_days(p, now) > R.edgar_root_days:
                out.append((p, f"edgar root json > {R.edgar_root_days} d"))
            elif _PAGE.match(p.name) and _age_days(p, now) > R.edgar_history_days:
                out.append((p, f"edgar history page > {R.edgar_history_days} d"))
    ft_dir = config.CACHE_DIR / "filing_text"
    if ft_dir.exists():
        for p in ft_dir.rglob("*"):
            if p.is_file() and _age_days(p, now) > R.filing_text_raw_days:
                out.append((p, f"raw filing text > {R.filing_text_raw_days} d"))
    if config.HISTORY_DIR.exists():
        cutoff = time.strftime("%Y-%m-%d", time.localtime(now - R.signal_history_days * 86400))
        for d in config.HISTORY_DIR.iterdir():
            if d.is_dir() and _DAY.match(d.name) and d.name < cutoff:
                out.append((d, f"signal archive > {R.signal_history_days} d"))
    zips = sorted(config.DATA_DIR.glob("driftrev_snapshot*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    out += [(p, f"snapshot zip beyond newest {R.snapshot_zips_keep}") for p in zips[R.snapshot_zips_keep:]]
    return out


def _trim_run_log() -> int:
    p = config.ARTIFACTS_DIR / artifacts.RUN_LOG
    if not p.exists():
        return 0
    lines = p.read_text().splitlines()
    if len(lines) <= R.run_log_max_lines:
        return 0
    p.write_text("\n".join(lines[-R.run_log_max_lines:]) + "\n")
    return len(lines) - R.run_log_max_lines


def _trim_logs() -> int:
    trimmed = 0
    if config.LOG_DIR.exists():
        for p in config.LOG_DIR.glob("*.log"):
            if p.stat().st_size > R.log_max_bytes:
                data = p.read_bytes()[-R.log_max_bytes // 2:]
                p.write_bytes(data)
                trimmed += 1
    return trimmed


def run(dry_run: bool = False) -> dict:
    expired = plan()
    freed = 0
    for path, reason in expired:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
        freed += size
        log.info("%s %s (%s)", "would delete" if dry_run else "delete", path, reason)
        if not dry_run:
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    report = {"expired": len(expired), "freed_mb": round(freed / 1e6, 1), "dry_run": dry_run,
              "by_reason": {}}
    for _, reason in expired:
        report["by_reason"][reason] = report["by_reason"].get(reason, 0) + 1
    if not dry_run:
        report["run_log_lines_dropped"] = _trim_run_log()
        report["logs_trimmed"] = _trim_logs()
        if config.WAREHOUSE.exists():
            try:
                with store.connect() as con:
                    con.execute("checkpoint")
            except Exception as exc:  # another process may hold the lock; not fatal
                report["checkpoint_error"] = str(exc)
    return report


def archive_signals(data_through: str) -> Path | None:
    """Keep a dated copy of today's signal book (pruned after `signal_history_days`)."""
    src = artifacts.path("signals")
    if not src.exists():
        return None
    day = config.HISTORY_DIR / data_through
    day.mkdir(parents=True, exist_ok=True)
    for name in ("signals", "events"):
        p = artifacts.path(name)
        if p.exists():
            shutil.copy2(p, day / p.name)
    return day


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Delete expired driftrev data files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-github", action="store_true", help="print GitHub retention settings as KEY=VALUE")
    a = ap.parse_args()
    if a.print_github:
        print(f"GH_CACHE_DAYS={R.gh_cache_days}\nGH_ARTIFACT_DAYS={R.gh_artifact_days}\nGH_RUN_DAYS={R.gh_run_days}")
        return
    print(json.dumps(run(a.dry_run), indent=2))


if __name__ == "__main__":
    main()
