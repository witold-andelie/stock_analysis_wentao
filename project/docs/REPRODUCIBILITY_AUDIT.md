# Reproducibility audit (peer-reviewer walk-through)

Fresh `git clone https://github.com/witold-andelie/stock_analysis_wentao.git` (public state at audit time),
fresh Python 3.11 venv, README section 11 followed literally, macOS arm64, Docker 29, home network.
MCP / agent checks that need code not yet pushed (`get_research`, `/v1/research`) were run against the local working tree.

## Pass / fail table

| # | Step | Result | Time |
|---|---|---|---|
| 1 | `git clone` | PASS | 1 s |
| 2 | `python3.11 -m venv` + `pip install -r requirements.txt` (pip cache warm; 1.0 GB venv) | PASS | 41 s |
| 3 | `cp .env.example .env`, set `SEC_USER_AGENT` | PASS (see F1) | <1 s |
| 4 | Smoke: `DRIFTREV_MAX_TICKERS=50 python -m driftrev.pipeline --mode update --retrain --tune` | PASS, exit 0, data_through 2026-11-13, test AUC 0.536 | 121 s (README says ~3 min) |
| 5 | `python -m pytest tests -q` (public tree) | PASS, 27 passed (README says 27) | 2 s |
| 6 | `docker compose up -d --build` | PASS, 4 containers up (build was fully layer-cached; cold slim build measured 71 s) | 3 s cached |
| 7 | API `:8000/health` 200; `/v1/signals/latest` 401 without token, 200 with | PASS | |
| 8 | MCP `:8001` unauthenticated POST 401; authenticated `initialize` + `tools/list` (9 tools) + `tools/call get_regime` | PASS | |
| 9 | Dashboard `:8501` HTTP 200, `/_stcore/health` ok | PASS | |
| 10 | Scheduler container stays up, logs `next run at ...22:30:00-05:00` | PASS | |
| 11 | Same compose flow with the `.env.example` value `DRIFTREV_DATA_DIR=./data` | FAIL then fixed (F2): containers ran with an empty data dir, `/health` = `no_data` | |
| 12 | `docker compose down` | PASS | |
| 13 | MCP stdio with the `mcp` Python client, public tree: `get_latest_signals` | PASS | 2 s |
| 14 | MCP stdio, local tree: `get_latest_signals` + `get_research` | PASS (research tables empty: audit data has no research outputs) | |
| 15 | `agent/grok_agent_example.py` without `XAI_API_KEY` (public version) | FAIL: `KeyError` traceback (F3). Local version now exits with a message | |
| 16 | Agent tool dispatch vs. API with a fake xAI server (tool call, bad token, dead API) | PASS: errors are fed back to the "model", no crash | |
| 17 | Every GET route in `routes.json` through `call_api` | 1 FAIL: `get_backtest_summary` -> HTTP 500 (F4); others OK | |
| 18 | Local slim image (`requirements-runtime.txt`): api, mcp (auth + initialize + tools), dashboard (Streamlit AppTest: 0 exceptions), scheduler | PASS | |
| 19 | Local full suite | PASS, 79 passed | 3 s |

Not done: notebooks (explicitly out of scope), full-universe run, `--mode file` snapshot run, a real xAI call, a cold-cache `docker build` of the public Dockerfile.

## Friction points

- **F1** README section 11 says "set `SEC_USER_AGENT`" but not that `.env` is read from `project/.env` only (via `config.py`, `parents[2]`); running from another directory still works, fine. Minor.
- **F2 (real bug)** `.env.example` has `DRIFTREV_DATA_DIR=./data`. `docker-compose.yml` uses `env_file: .env`, which overrides the image's `DRIFTREV_DATA_DIR=/data`, so the containers read `/app/data` (does not exist) instead of the mounted volume: API reports `no_data`. The comment in `.env.example` says "/data in Docker" but nothing enforces it. Fixed on our side in `docker-compose.yml` by pinning `environment: DRIFTREV_DATA_DIR: /data` (wins over `env_file`). Verified with `docker compose config`.
- **F3** Public `grok_agent_example.py` raised a bare `KeyError: 'XAI_API_KEY'`. Local version now prints a one-line message and exits 1; also turns xAI HTTP errors and network errors into short messages, and accepts `XAI_API_URL` for testing against a fake server.
- **F4 (real bug, public AND local tree)** `GET /v1/backtest/summary` (and MCP `get_backtest_summary`) return HTTP 500, `ValueError: Out of range float values are not JSON compliant`: `metrics.json` is written with `NaN` (`avg_r_multiple`, `top5_profit_share`, benchmark `sharpe/sortino/calmar`) and `service.backtest_summary` returns it unsanitised. Seen with the 50-ticker smoke data. Proposed fix below.
- **F5** README section 11 has no Docker section; compose users must know to bring their own data (named volume `driftrev-data` starts empty, so `/health` says `no_data` until the pipeline runs inside a container, or data is copied/bind-mounted). Proposed text below.
- **F6** The public repo has no `requirements-dev.txt` / `ruff.toml` (they exist locally and in `ci.yml`): a reviewer cloning the public repo cannot run lint. Push them.
- **F7** `docker compose up` cold build time is not stated; the image was 1.79 GB (jupyter etc.). Runtime requirements file brings it to 1.37 GB on arm64 (amd64 wheels may be larger).
- **F8** MCP stdio server prints `Processing request of type ...` and a pydantic `IncompleteFieldDefinitionWarning` on stderr; harmless but noisy for MCP host logs (library behavior; `logging` level could be raised).
- **F9** The `mcp` compose service has no healthcheck (api and dashboard do); `docker compose ps` shows it as just "Up". The endpoint needs a bearer token, so a healthcheck would have to send it.
- **F10** Smoke run is ~2 min on this machine, README says ~3 min: fine, but it depends on live yfinance/SEC downloads (results differ by day; AUC 0.536 here).

## Proposed fixes for the lead (read-only files)

**F4: `src/driftrev/artifacts.py`** (sanitise NaN/inf on read; also covers `model_info` and the MCP tool):

```diff
+import math
+
+
+def _nan_to_none(obj):
+    if isinstance(obj, float) and not math.isfinite(obj):
+        return None
+    if isinstance(obj, dict):
+        return {k: _nan_to_none(v) for k, v in obj.items()}
+    if isinstance(obj, list):
+        return [_nan_to_none(v) for v in obj]
+    return obj
+
+
 def load_metrics() -> dict:
     p = config.ARTIFACTS_DIR / METRICS
-    return json.loads(p.read_text()) if p.exists() else {}
+    return _nan_to_none(json.loads(p.read_text())) if p.exists() else {}
```

Add a regression test: write a `metrics.json` containing `NaN` in `strategies[0].report_card.avg_r_multiple`, call `/v1/backtest/summary`, expect 200 and `null`.

**F5: `README.md` section 11**, after the smoke-run block:

```diff
+# Docker (api :8000, mcp :8001, dashboard :8501, scheduler). Set DRIFTREV_API_TOKEN in .env first.
+docker compose up -d --build
+curl localhost:8000/health        # "no_data" until the pipeline has run in the volume:
+docker compose exec scheduler python -m driftrev.pipeline --mode update
+docker compose down
```

**F2 (optional, `.env.example`)**: change the comment to `# set to /data only if you run containers without compose`. Not needed now that compose pins it.

**F6**: `git add project/requirements-dev.txt project/ruff.toml` and mention `pip install -r requirements-dev.txt` in section 11 before `pytest`/`ruff`.

**F8 (optional, `src/driftrev/mcp_server.py` stdio branch)**: `logging.getLogger("mcp").setLevel(logging.WARNING)`.
