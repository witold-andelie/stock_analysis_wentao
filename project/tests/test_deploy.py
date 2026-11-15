"""R5 deployment packaging: manifests parse and reference the right image / port / health path."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT.parent / ".github" / "workflows"


def _pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, ver = line.split("==")
            out[name.lower()] = ver
    return out


def test_runtime_requirements_subset_of_full_pins():
    full, rt = _pins(ROOT / "requirements.txt"), _pins(ROOT / "requirements-runtime.txt")
    assert rt and set(rt) <= set(full)
    assert all(full[k] == v for k, v in rt.items())
    for banned in ("jupyter", "nbconvert", "ipykernel", "pytest"):
        assert banned not in rt


def test_dockerfile_uses_runtime_requirements():
    assert "requirements-runtime.txt" in (ROOT / "Dockerfile").read_text()


def test_image_workflow():
    wf = yaml.safe_load((WORKFLOWS / "image.yml").read_text())
    on = wf.get("on") or wf[True]
    assert "workflow_dispatch" in on and on["push"]["branches"] == ["main"]
    assert on["push"]["paths"] == ["project/**"]
    pub = wf["jobs"]["publish"]
    assert pub["permissions"]["packages"] == "write"
    text = (WORKFLOWS / "image.yml").read_text()
    assert "ghcr.io/${owner}/driftrev" in text and ":latest" in text and "sha-" in text
    assert "docker/build-push-action" in text and "type=gha" in text
    assert wf["jobs"]["verify"]["needs"] == "publish"


def test_cloud_run_manifest():
    svc = yaml.safe_load((ROOT / "deploy/cloudrun/service.yaml").read_text())
    c = svc["spec"]["template"]["spec"]["containers"][0]
    assert "driftrev:latest" in c["image"]
    assert c["ports"][0]["containerPort"] == 8000
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"
    names = {e["name"] for e in c["env"]}
    assert {"DRIFTREV_API_TOKEN", "SEC_USER_AGENT", "DRIFTREV_DATA_DIR"} <= names
    assert svc["spec"]["template"]["metadata"]["annotations"]["autoscaling.knative.dev/maxScale"] == "1"
    assert c["volumeMounts"][0]["mountPath"] == "/data"
    assert "30 22 * * 1-5" in (ROOT / "deploy/cloudrun/deploy.sh").read_text()


def test_fly_manifest():
    import tomllib
    fly = tomllib.loads((ROOT / "deploy/fly/fly.toml").read_text())
    assert fly["build"]["image"].endswith("/driftrev:latest")
    assert fly["http_service"]["internal_port"] == 8000
    assert fly["http_service"]["checks"][0]["path"] == "/health"
    assert fly["mounts"]["destination"] == "/data"


def test_render_manifest():
    r = yaml.safe_load((ROOT / "deploy/render/render.yaml").read_text())
    svc = r["services"][0]
    assert svc["healthCheckPath"] == "/health"
    assert svc["image"]["url"].endswith("/driftrev:latest")
    assert svc["disk"]["mountPath"] == "/data"
    secrets = {e["key"] for e in svc["envVars"] if e.get("sync") is False}
    assert {"DRIFTREV_API_TOKEN", "SEC_USER_AGENT"} <= secrets


def test_no_secrets_committed_in_manifests():
    for p in (ROOT / "deploy").rglob("*"):
        if p.is_file():
            assert not re.search(r"(sk-|xai-)[A-Za-z0-9]{20,}", p.read_text()), p
