"""First run on an empty warehouse: ingest steps that read derived tables must skip, not crash."""

from __future__ import annotations

import pandas as pd

from driftrev import config, filing_text


def test_filing_text_skips_on_first_run(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for name, path in (("DATA_DIR", data), ("WAREHOUSE", data / "warehouse.duckdb"), ("CACHE_DIR", data / "cache"),
                       ("SNAPSHOT_DIR", data / "snapshot"), ("ARTIFACTS_DIR", data / "artifacts")):
        monkeypatch.setattr(config, name, path)

    def boom(*args, **kwargs):
        raise AssertionError("must not fetch anything without events")

    monkeypatch.setattr(filing_text, "load_lm", boom)
    report = filing_text.ingest()
    assert report["skipped"] is True and report["targets"] == 0
    assert filing_text.select_targets(pd.DataFrame(), pd.DataFrame()).empty
