"""Notifying (OPM SD5, roadmap R3): push the day's signals to Telegram and/or a JSON webhook.

Configured entirely by environment variables; with none set, `notify()` is a no-op:

    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   Telegram Bot API sendMessage
    DRIFTREV_WEBHOOK_URL                   POST {"text": ..., "signals": [...], ...} (n8n, a Grok bot, Discord-style relays)
    NOTIFY_MIN_TIER                        lowest tier to include (default BUY → STRONG_BUY + BUY)

    python -m driftrev.notify            # send the current artifacts once
    python -m driftrev.notify --dry-run  # print the message only
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd
import requests

from . import artifacts

log = logging.getLogger(__name__)
ORDER = list(artifacts.TIERS)  # STRONG_BUY, BUY, WATCH, SKIP, AVOID, NO_CALL
MAX_LINES = 15


def _selected(sig: pd.DataFrame, min_tier: str) -> pd.DataFrame:
    if sig.empty or "tier" not in sig:
        return sig
    keep = ORDER[: ORDER.index(min_tier) + 1] if min_tier in ORDER else ORDER[:2]
    return sig[sig["tier"].isin(keep)].sort_values("prob", ascending=False)


def build_message(min_tier: str | None = None) -> dict:
    """Return {'text', 'signals', 'data_through', 'regime'} built from the artifact store only."""
    min_tier = (min_tier or os.environ.get("NOTIFY_MIN_TIER") or "BUY").upper()
    m = artifacts.load_metrics()
    sig = _selected(artifacts.load_frame("signals"), min_tier)
    reg = (m.get("regime") or {}).get("state", "unknown")
    head = f"Drift-or-Reversal · data through {m.get('data_through', '?')} · regime {reg}"
    if sig.empty:
        lines = [head, f"No signals at tier ≥ {min_tier} for the latest session."]
    else:
        lines = [head, f"{len(sig)} signal(s) at tier ≥ {min_tier} (entry: next open):"]
        for r in sig.head(MAX_LINES).itertuples(index=False):
            exit_by = pd.Timestamp(r.exit_by_date).strftime("%Y-%m-%d") if pd.notna(getattr(r, "exit_by_date", None)) else "?"
            lines.append(f"• {r.tier} {r.ticker} ({r.event_family}) p={r.prob:.2f} "
                         f"ref {r.entry_ref_price:.2f} stop {r.stop_loss_price:.2f} tp {r.take_profit_price:.2f} exit≤{exit_by}")
        if len(sig) > MAX_LINES:
            lines.append(f"… and {len(sig) - MAX_LINES} more")
    if str(reg).lower() == "stressed":
        lines.append("Regime STRESSED: new entries are paused by the risk-managed strategies.")
    lines.append("Research output, not investment advice.")
    return {"text": "\n".join(lines), "signals": artifacts.to_records(sig.head(MAX_LINES)),
            "data_through": m.get("data_through"), "regime": reg}


def configured() -> dict[str, bool]:
    return {"telegram": bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")),
            "webhook": bool(os.environ.get("DRIFTREV_WEBHOOK_URL"))}


def notify(dry_run: bool = False) -> dict:
    """Send to every configured channel. Never raises: failures are reported, not fatal."""
    channels = configured()
    if not any(channels.values()) and not dry_run:
        return {"sent": [], "skipped": "no channel configured"}
    msg = build_message()
    if dry_run:
        return {"sent": [], "dry_run": True, "text": msg["text"], "channels": channels}
    sent, errors = [], {}
    if channels["telegram"]:
        try:
            url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
            r = requests.post(url, json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": msg["text"],
                                         "disable_web_page_preview": True}, timeout=20)
            r.raise_for_status()
            sent.append("telegram")
        except Exception as exc:  # token is in the URL: never log the exception text verbatim
            errors["telegram"] = type(exc).__name__
    if channels["webhook"]:
        try:
            r = requests.post(os.environ["DRIFTREV_WEBHOOK_URL"], json=msg, timeout=20)
            r.raise_for_status()
            sent.append("webhook")
        except Exception as exc:
            errors["webhook"] = type(exc).__name__
    return {"sent": sent, "errors": errors, "n_signals": len(msg["signals"])}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Push the latest signals to Telegram / webhook")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = notify(dry_run=a.dry_run)
    print(out.pop("text", "")) if a.dry_run else None
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
