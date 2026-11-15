# Connecting a 24/7 personal agent

The pipeline writes artifacts; the API (`:8000`) and MCP server (`:8001`) serve them.
Three ways to wire an agent (Grok bot, Claude, ChatGPT, n8n, ...):

## (a) OpenAPI import
Point the agent's "import tools from OpenAPI" feature at `https://<host>/openapi.json`
and set the header `Authorization: Bearer <DRIFTREV_API_TOKEN>`. Operation IDs are clean
(`get_latest_signals`, `get_ticker`, ...).

## (b) Function calling (`agent/tools.json`)
`tools.json` is an OpenAI/xAI-format tool list; `routes.json` maps each tool to a method + path.
Regenerate with `python -m driftrev.agent_tools`. A working loop:

```bash
export XAI_API_KEY=... XAI_MODEL=grok-4 DRIFTREV_API_URL=http://localhost:8000 DRIFTREV_API_TOKEN=...
python agent/grok_agent_example.py "What are today's best signals?"
```

## (c) MCP
- Local (stdio): command `python -m driftrev.mcp_server --transport stdio` with `PYTHONPATH=src`.
- Remote: `python -m driftrev.mcp_server --transport streamable-http --host 0.0.0.0 --port 8001`,
  URL `https://<host>:8001/mcp`, header `Authorization: Bearer <token>`.

## Exposing it safely
- Always set `DRIFTREV_API_TOKEN` (long random string). Without it the API answers 503.
- Do not open ports to the internet: use Tailscale (`tailscale serve 8000`) or a Cloudflare Tunnel
  (`cloudflared tunnel --url http://localhost:8000`) with TLS.
- Order submission is off by default: `/v1/orders/submit` needs `confirm=true` AND
  `DRIFTREV_ALLOW_ORDERS=true`; it is not in tools.json. Keep `ALPACA_PAPER=true` until you trust it.

## Daily-brief prompt
> Call get_latest_signals(limit=10) and get_regime. Write a brief: regime, data date, the top
> signals (ticker, event type, probability, stop/take-profit, exit date), and which are
> STRONG_BUY/BUY. If the regime is STRESSED say new entries are paused. Add a short risk note.

## Schedule
The scheduler container refreshes data at 22:30 ET Mon-Fri. Have the agent poll
`GET /v1/signals/latest` shortly afterwards, e.g. cron `0 23 * * 1-5` (TZ=America/New_York),
check `/health` `data_through` is today before reporting.
