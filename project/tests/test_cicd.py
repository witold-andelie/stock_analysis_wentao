"""CI/CD contract (OPM SD7): workflows exist, keep data out of git, and use config.Retention."""

import re
from pathlib import Path

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
