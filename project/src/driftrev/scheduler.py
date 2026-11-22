"""Long-running scheduler for Docker: runs the pipeline Mon-Fri at DRIFTREV_SCHEDULE_TIME
in the local exchange time of DRIFTREV_MARKET (US default 22:30 New York;
DE 18:30 Berlin).

  python -m driftrev.scheduler          # loop forever
  python -m driftrev.scheduler --once   # run now and exit
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger("driftrev.scheduler")
TZ = ZoneInfo(config.MARKET.timezone)


def schedule_time() -> str:
    return os.environ.get("DRIFTREV_SCHEDULE_TIME", "").strip() or config.MARKET.run_time


def next_run(now: datetime, hhmm: str) -> datetime:
    """Next Mon-Fri occurrence of hh:mm strictly after `now` (tz-aware, local exchange time)."""
    h, m = (int(x) for x in hhmm.split(":"))
    cand = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if cand <= now:
        cand += timedelta(days=1)
    while cand.weekday() >= 5:
        cand += timedelta(days=1)
    return cand


def run_once() -> None:
    try:
        config.require_implemented_market()
        from . import pipeline  # lazy import: core module

        log.info("pipeline start")
        log.info("pipeline done: %s", pipeline.run_pipeline(mode="update"))
    except Exception:  # keep the loop alive whatever happens
        log.exception("pipeline failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("active market: %s", config.MARKET.code)
    if args.once:
        run_once()
        return
    hhmm = schedule_time()
    while True:
        target = next_run(datetime.now(TZ), hhmm)
        log.info("next run at %s", target.isoformat())
        while (left := (target - datetime.now(TZ)).total_seconds()) > 0:
            time.sleep(min(left, 60))
        run_once()


if __name__ == "__main__":
    main()
