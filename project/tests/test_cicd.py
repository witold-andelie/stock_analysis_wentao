"""CI/CD contract (OPM SD7): workflows exist, keep data out of git, and use config.Retention."""

import re
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from driftrev import config

PROJECT = Path(__file__).resolve().parents[1]
WF = PROJECT.parent / ".github" / "workflows"
R = config.SETTINGS.retention


def _wf(name: str) -> str:
    return (WF / name).read_text()


def test_workflows_exist():
    for name in ("ci.yml", "daily.yml", "cleanup.yml"):
        assert (WF / name).exists(), name


def test_cleanup_retention_matches_config():
    text = _wf("cleanup.yml") + (PROJECT / "scripts" / "gh_cleanup.sh").read_text()
    for key, val in (("GH_CACHE_DAYS", R.gh_cache_days), ("GH_ARTIFACT_DAYS", R.gh_artifact_days),
                     ("GH_RUN_DAYS", R.gh_run_days)):
        found = set(re.findall(rf'{key}[:=]\s*"?\$?{{?{key}?:?-?(\d+)', text)) | set(re.findall(rf'{key}: "(\d+)"', text))
        assert found == {str(val)}, (key, found)


def test_daily_artifact_retention_matches_config():
    days = re.findall(r"retention-days:\s*(\d+)", _wf("daily.yml"))
    assert str(R.gh_artifact_days) in days


def test_daily_runs_after_us_close_and_never_commits_data():
    text = _wf("daily.yml")
    assert '"30 3 * * 2-6"' in text                       # 22:30 ET (EST) after Mon-Fri closes
    assert "git commit" not in text and "git push" not in text
    assert "actions/cache/save" in text and "upload-pages-artifact" in text


def test_data_dir_is_gitignored():
    assert "data/" in (PROJECT / ".gitignore").read_text().split()


def _steps(name: str) -> list[str]:
    return re.split(r"(?m)^      - ", _wf(name))[1:]


def _action_steps(name: str, action: str) -> list[str]:
    return [step for step in _steps(name) if re.search(rf"uses:\s*{re.escape(action)}@", step)]


def test_daily_de_exists_and_runs_after_xetra_close():
    assert (WF / "daily_de.yml").exists()
    text = _wf("daily_de.yml")
    assert re.findall(r'cron:\s*"([^"]+)"', text) == ["30 17 * * 1-5", "0 13 * * 0"]
    assert re.search(r"workflow_dispatch:\s*\n", text)
    assert re.search(r"DRIFTREV_MARKET:\s*de\b", text)
    assert re.search(r"\bgit\s+(?:add|commit|push)\b", text) is None
    assert "max_tickers" not in text and "SEC_USER_AGENT" not in text
    assert "snapshot" not in text.lower()
    assert re.search(r"timeout-minutes:\s*120\b", text)
    assert '"${{ github.event.schedule }}" = "0 13 * * 0"' in text
    assert '"${{ inputs.retrain }}" = "true"' in text
    assert "! -f data/de/artifacts/model.joblib" in text
    assert 'flags="--retrain"' in text
    assert '"${{ inputs.tune }}" = "true"' in text and 'flags="$flags --tune"' in text
    assert "python -m driftrev.pipeline --mode update ${{ steps.mode.outputs.flags }} | tee data/de/pipeline_result.json" in text
    assert "data/de/artifacts/metrics.json" in text and "data/de/site/summary.md" in text


def test_daily_de_caches_only_de_data():
    for action in ("restore", "save"):
        steps = [step for step in _action_steps("daily_de.yml", f"actions/cache/{action}")
                 if "key: driftrev-de-data-" in step]
        assert len(steps) == 1
        step = steps[0]
        paths = re.search(r"path:\s*\|\n((?:[ \t]+project/[^\n]+\n)+)", step)
        assert paths is not None
        assert paths[1].split() == ["project/data/de/warehouse.duckdb", "project/data/de/artifacts"]
        assert "key: driftrev-de-data-${{ github.run_id }}" in step
        if action == "restore":
            assert "restore-keys: driftrev-de-data-" in step
        else:
            assert "if: success()" in step


@pytest.mark.parametrize("name,own,other", [("daily.yml", "us", "de"), ("daily_de.yml", "de", "us")])
@pytest.mark.parametrize("markets", [("us",), ("de",), ("us", "de")])
def test_pages_site_composed_from_both_markets(name, own, other, markets, tmp_path):
    site_paths = {"us": "project/data/site", "de": "project/data/de/site"}
    saves = [step for step in _action_steps(name, "actions/cache/save") if f"key: driftrev-site-{own}-" in step]
    assert len(saves) == 1
    assert f"path: {site_paths[own]}" in saves[0]
    assert f"key: driftrev-site-{own}-${{{{ github.run_id }}}}" in saves[0]
    assert "if: success()" in saves[0]
    restores = [step for step in _action_steps(name, "actions/cache/restore") if f"key: driftrev-site-{other}-" in step]
    assert len(restores) == 1
    assert f"path: {site_paths[other]}" in restores[0]
    assert f"restore-keys: driftrev-site-{other}-" in restores[0]
    assert "fail-on-cache-miss: false" in restores[0]

    compose = next(step for step in _steps(name) if step.startswith("name: Compose Pages site\n"))
    script = dedent(compose.split("run: |\n", 1)[1])
    project = tmp_path / "project"
    project.mkdir()
    for market in markets:
        site = tmp_path / site_paths[market]
        (site / "assets").mkdir(parents=True)
        (site / "index.html").write_text(market)
        (site / "assets" / "report.css").write_text(f"{market} styles")
    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=project, check=True)
    for market, target in (("us", project / "pages"), ("de", project / "pages" / "de")):
        if market in markets:
            assert (target / "index.html").read_text() == market
            assert (target / "assets" / "report.css").read_text() == f"{market} styles"
        else:
            assert not (target / "index.html").exists()
    uploads = _action_steps(name, "actions/upload-pages-artifact")
    assert len(uploads) == 1
    assert re.search(r"path:\s*project/pages\s*\n", uploads[0])
    assert re.search(r"retention-days:\s*7\b", uploads[0])
    text = _wf(name)
    assert text.index(saves[0]) < text.index(restores[0]) < text.index(compose) < text.index(uploads[0])
    deploy = text.split("\n  deploy-pages:\n", 1)[1]
    assert re.search(r"concurrency:\s*\n\s+group:\s*pages-deploy\s*\n\s+cancel-in-progress:\s*false\b", deploy)


def test_daily_de_artifact_retention_matches_config():
    steps = _action_steps("daily_de.yml", "actions/upload-artifact")
    assert len(steps) == 1
    assert re.findall(r"retention-days:\s*(\d+)", steps[0]) == [str(R.gh_artifact_days)]
    assert "name: driftrev-de-results-${{ steps.pipe.outputs.data_through }}-${{ github.run_id }}" in steps[0]
    paths = re.search(r"path:\s*\|\n((?:[ \t]+project/[^\n]+\n)+)", steps[0])
    assert paths is not None
    assert all(path.startswith("project/data/de/") for path in paths[1].split())


def test_cleanup_covers_all_cache_prefixes_and_keeps_newest():
    text = (PROJECT / "scripts" / "gh_cleanup.sh").read_text()
    prefixes = re.search(r"cache_prefixes=\((.*?)\)", text, re.DOTALL)
    assert prefixes is not None
    assert set(prefixes[1].split()) == {"driftrev-data-", "driftrev-de-data-", "driftrev-site-us-", "driftrev-site-de-"}
    compact = re.sub(r"\s+", " ", text)
    assert 'for prefix in "${cache_prefixes[@]}"; do' in compact
    assert re.search(r'newest=\$\(gh cache list .*?--key "\$prefix" .*?--sort created_at --order desc --limit 1', compact)
    assert 'newest_ids="$newest_ids,${newest:-0}"' in compact
    guard = r'select(.id as \$id | [$newest_ids] | index(\$id) | not)'
    assert re.sub(r"\s+", "", guard) in re.sub(r"\s+", "", text)
    assert 'select(.createdAt < \\"$c\\")' in compact


def test_pages_dir_ignored():
    assert "pages/" in (PROJECT / ".gitignore").read_text().split()


@pytest.mark.parametrize("name", ["daily.yml", "daily_de.yml"])
def test_daily_workflows_are_valid_yaml(name):
    yaml = pytest.importorskip("yaml", reason="PyYAML is not installed in the existing venv; skipping workflow YAML validation")
    workflow = yaml.safe_load(_wf(name))
    assert isinstance(workflow, dict)
    assert {"run", "deploy-pages"} <= workflow["jobs"].keys()
