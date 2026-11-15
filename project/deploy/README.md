# Deploying the API (R5, optional)

These are **templates**. Nothing here is deployed by this repository and CI never runs them.
CI only publishes the image (`.github/workflows/image.yml` -> `ghcr.io/<owner>/driftrev:latest` and `:sha-<short>`).
After the first publish, make the GHCR package public (or give the platform a pull token).

| Target | File | Storage for /data | Daily pipeline |
|---|---|---|---|
| Google Cloud Run | `cloudrun/service.yaml`, `cloudrun/deploy.sh` | GCS bucket via GCS FUSE volume | Cloud Run Job + Cloud Scheduler (22:30 ET Mon-Fri) |
| Fly.io | `fly/fly.toml` | Fly volume `driftrev_data` | `driftrev.scheduler` beside the API |
| Render | `render/render.yaml` | Render persistent disk (paid plan) | `driftrev.scheduler` beside the API |

## Required configuration
- `DRIFTREV_API_TOKEN` (required, long random string; without it the API answers 503)
- `SEC_USER_AGENT` (required for the pipeline, `"Your Name you@example.com"`)
- optional: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DRIFTREV_WEBHOOK_URL`, `NOTIFY_MIN_TIER`
- optional broker (paper only): `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`; keep `DRIFTREV_ALLOW_ORDERS=false`
- `DRIFTREV_DATA_DIR=/data`. Put every secret in the platform's secret store, never in the manifest.

## Persistent storage
The DuckDB warehouse and artifacts (models, signals, metrics) live under `/data` and must be on a volume,
otherwise each restart starts with no data (`/health` then reports `no_data`). DuckDB is single-writer:
run exactly one instance (`maxScale 1`, `numInstances: 1`, one Fly machine). Object-store FUSE mounts (Cloud Run)
are slower than a local disk: acceptable for the API, and for a nightly job.
A first run needs a bootstrap: either run `python -m driftrev.pipeline --mode update` once inside the instance
(~45 min for the full S&P 1500), or point `DRIFTREV_SNAPSHOT_URL` at the weekly snapshot and run `--mode file`.

## Health check
`GET /health` is public and returns `status` (`ok` or `no_data`), `data_through`, `last_run_status`. All manifests use it.
`no_data` still returns HTTP 200 so the platform keeps the instance up while you bootstrap.

## Pointing a 24/7 Grok bot (or any agent) at it
- OpenAPI: import `https://<host>/openapi.json`, header `Authorization: Bearer <DRIFTREV_API_TOKEN>`.
- Function calling: `DRIFTREV_API_URL=https://<host> DRIFTREV_API_TOKEN=... XAI_API_KEY=... python agent/grok_agent_example.py "today's signals"`.
- MCP over HTTP needs the second port (8001) as a separate service: run the same image with
  `python -m driftrev.mcp_server --transport streamable-http --host 0.0.0.0 --port 8001`; the single-service templates above expose the REST API only.
- Poll shortly after 22:30 ET and check `data_through` in `/health` is today before reporting.

## Image size
`requirements-runtime.txt` drops jupyter/nbconvert/ipykernel/pytest/matplotlib/seaborn from the image
(1.79 GB -> about 1.37 GB on arm64). Add a `.env` only locally; it is excluded by `.dockerignore`.
