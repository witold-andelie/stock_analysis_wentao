"""Minimal Grok (xAI) tool-calling loop over the Drift-or-Reversal REST API.

  XAI_API_KEY=... DRIFTREV_API_URL=http://localhost:8000 DRIFTREV_API_TOKEN=... \
  python agent/grok_agent_example.py "What are today's best signals?"

xAI's chat completions API is OpenAI-compatible, so plain httpx is enough.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
TOOLS = json.loads((HERE / "tools.json").read_text())
ROUTES = json.loads((HERE / "routes.json").read_text())

XAI_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.environ.get("XAI_MODEL", "grok-4")
API = os.environ.get("DRIFTREV_API_URL", "http://localhost:8000").rstrip("/")
SYSTEM = ("You are a trading-research assistant. Use the tools to answer from real data, "
          "quote probabilities and the market regime, and remind the user this is research, "
          "not investment advice. Never claim to have placed orders.")


def call_api(name: str, args: dict) -> str:
    """Dispatch one tool call to our REST API; path params are substituted from args."""
    route = ROUTES[name]
    path, params = route["path"], dict(args)
    for key in list(params):
        if "{" + key + "}" in path:
            path = path.replace("{" + key + "}", str(params.pop(key)))
    headers = {"Authorization": f"Bearer {os.environ.get('DRIFTREV_API_TOKEN', '')}"}
    if route["method"] == "GET":
        r = httpx.get(API + path, params=params, headers=headers, timeout=60)
    else:
        r = httpx.post(API + path, json=params, headers=headers, timeout=60)
    return r.text[:20000]  # cap what we feed back to the model


def run(question: str, max_steps: int = 6) -> str:
    key = os.environ["XAI_API_KEY"]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = httpx.post(XAI_URL, headers={"Authorization": f"Bearer {key}"}, timeout=120,
                          json={"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto"})
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or ""
        for tc in calls:
            fn = tc["function"]
            try:
                out = call_api(fn["name"], json.loads(fn.get("arguments") or "{}"))
            except Exception as exc:  # feed errors back so the model can recover
                out = json.dumps({"error": str(exc)})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    return "Stopped: too many tool-call steps."


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:]) or "What are today's best signals?"))
