import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from driftrev import service
from driftrev.api import app


def test_health_public(art_dir):
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["artifacts_present"]["signals"]


def test_auth(art_dir, monkeypatch):
    c = TestClient(app)
    assert c.get("/v1/regime").status_code == 401
    assert c.get("/v1/regime", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/v1/regime", headers={"Authorization": "Bearer secret"}).status_code == 200
    monkeypatch.delenv("DRIFTREV_API_TOKEN")
    assert c.get("/v1/regime").status_code == 503
    monkeypatch.setenv("DRIFTREV_ALLOW_NO_AUTH", "true")
    assert c.get("/v1/regime").status_code == 200


def test_signals(client):
    j = client.get("/v1/signals/latest").json()
    assert j["count"] == 3 and j["regime"] == "NORMAL" and "AAA" in j["summary"]
    assert client.get("/v1/signals/latest?min_prob=0.65").json()["count"] == 1
    assert client.get("/v1/signals/latest?tier=BUY").json()["signals"][0]["ticker"] == "BBB"
    assert client.get("/v1/signals/latest?event_family=earnings").json()["count"] == 2


def test_events_ticker(client):
    assert client.get("/v1/events/recent?days=30").json()["count"] == 3
    assert client.get("/v1/events/recent?ticker=aaa").json()["count"] == 1
    j = client.get("/v1/tickers/AAA").json()
    assert j["universe"]["company"] == "A" and len(j["prices"]) == 60 and j["open_signal"]
    assert client.get("/v1/tickers/ZZZ").status_code == 404


def test_backtest_model_regime(client):
    j = client.get("/v1/backtest/summary").json()
    assert j["strategies"][0]["name"] == "prob"
    assert client.get("/v1/backtest/trades?limit=1").json()["count"] == 1
    assert client.get("/v1/backtest/equity?strategy=prob").json()["count"] > 0
    assert client.get("/v1/backtest/equity?weekly=false").json()["count"] == 60
    m = client.get("/v1/model?top_n=2&dimension=tier").json()
    assert len(m["feature_importance"]) == 2 and len(m["precision_breakdown"]) == 1
    assert client.get("/v1/regime").json()["regime"]["state"] == "NORMAL"
    assert len(client.get("/v1/runs").json()["runs"]) == 1
    assert len(client.get("/v1/agent/tools").json()) >= 10


def test_empty_artifacts_no_500(empty_art_dir):
    c = TestClient(app, headers={"Authorization": "Bearer secret"})
    assert c.get("/health").json()["status"] == "no_data"
    for p in ("/v1/signals/latest", "/v1/events/recent", "/v1/backtest/summary", "/v1/backtest/trades",
              "/v1/backtest/equity", "/v1/model", "/v1/regime", "/v1/runs"):
        assert c.get(p).status_code == 200, p
    assert c.get("/v1/tickers/AAA").status_code == 404


def test_pipeline_job(client, monkeypatch):
    calls = []
    fake = types.ModuleType("driftrev.pipeline")
    fake.run_pipeline = lambda mode="update", retrain=False: (time.sleep(0.3), calls.append((mode, retrain)), {"ok": 1})[2]
    monkeypatch.setitem(sys.modules, "driftrev.pipeline", fake)
    import driftrev
    monkeypatch.setattr(driftrev, "pipeline", fake, raising=False)
    service._JOBS.clear()
    r = client.post("/v1/pipeline/run", json={"mode": "file", "retrain": True})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert client.post("/v1/pipeline/run", json={}).status_code == 409
    for _ in range(50):
        j = client.get(f"/v1/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "succeeded" and j["result"] == {"ok": 1} and calls == [("file", True)]
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.post("/v1/pipeline/run", json={"mode": "bad"}).status_code == 422


def test_pipeline_job_failure(client, monkeypatch):
    import driftrev
    fake = types.ModuleType("driftrev.pipeline")

    def boom(mode="update", retrain=False):
        raise RuntimeError("boom")

    fake.run_pipeline = boom
    monkeypatch.setattr(driftrev, "pipeline", fake, raising=False)
    monkeypatch.setitem(sys.modules, "driftrev.pipeline", fake)
    service._JOBS.clear()
    jid = client.post("/v1/pipeline/run", json={}).json()["job_id"]
    for _ in range(50):
        j = client.get(f"/v1/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "failed" and "boom" in j["error"]


def test_orders(client, monkeypatch):
    j = client.post("/v1/orders/preview", json={"capital": 10000}).json()
    assert j["dry_run"] and j["count"] == 2
    assert client.post("/v1/orders/preview").status_code == 200
    assert client.post("/v1/orders/submit", json={"confirm": False}).status_code == 400
    assert client.post("/v1/orders/submit", json={"confirm": True}).status_code == 403
    assert client.get("/v1/portfolio").json()["configured"] is False
