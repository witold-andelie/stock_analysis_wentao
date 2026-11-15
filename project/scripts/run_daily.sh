#!/usr/bin/env bash
# Daily pipeline run for cron. Logs to data/logs/cron.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
{ echo "=== $(date -Is) start"; python -m driftrev.pipeline --mode update; echo "=== $(date -Is) done"; } >> data/logs/cron.log 2>&1
