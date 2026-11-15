"""R3 Notifying: message content, tier filter, no-op without configuration, both channels."""

import json

from driftrev import notify


def _metrics(art):
    (art / "metrics.json").write_text(json.dumps({"data_through": "2026-11-13", "regime": {"state": "normal"}}))


def test_noop_without_configuration(art_dir, monkeypatch):
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DRIFTREV_WEBHOOK_URL"):
        monkeypatch.delenv(k, raising=False)
    assert notify.notify() == {"sent": [], "skipped": "no channel configured"}


def test_message_filters_tiers(art_dir, monkeypatch):
    _metrics(art_dir)
    msg = notify.build_message("BUY")
    assert "AAA" in msg["text"] and "BBB" in msg["text"] and "CCC" not in msg["text"]   # CCC is SKIP
    assert msg["data_through"] == "2026-11-13" and len(msg["signals"]) == 2
    assert "AAA" in notify.build_message("STRONG_BUY")["text"]
    assert "BBB" not in notify.build_message("STRONG_BUY")["text"]


def test_sends_to_both_channels(art_dir, monkeypatch):
    _metrics(art_dir)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t0k")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("DRIFTREV_WEBHOOK_URL", "https://example.invalid/hook")
    calls = []

    class Resp:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(notify.requests, "post", lambda url, json, timeout: calls.append((url, json)) or Resp())
    out = notify.notify()
    assert out["sent"] == ["telegram", "webhook"] and out["errors"] == {}
    assert calls[0][0].endswith("/bott0k/sendMessage") and calls[0][1]["chat_id"] == "42"
    assert calls[1][1]["data_through"] == "2026-11-13"


def test_failures_are_reported_not_raised(art_dir, monkeypatch):
    _metrics(art_dir)
    monkeypatch.setenv("DRIFTREV_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    def boom(*a, **k):
        raise ConnectionError("secret-in-url")

    monkeypatch.setattr(notify.requests, "post", boom)
    out = notify.notify()
    assert out["sent"] == [] and out["errors"] == {"webhook": "ConnectionError"}
