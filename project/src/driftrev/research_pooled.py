"""Compare US+DE training with German-only models, chosen on German VALIDATION.

Each market is prepared separately. TEST scores use train+validation refits and
are reported only; all warehouse access is explicitly read-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import duckdb
import pandas as pd

from . import config, dataset, models, research
from .dataset import TARGET

log = logging.getLogger(__name__)

VARIANTS = ("de_only", "pooled_all", "pooled_same_families", "us_transfer_same_families")
SAME_FAMILIES = ("shock_crash", "momentum_breakout")


def load_events(warehouse: Path) -> pd.DataFrame:
    """Read the events table directly from an explicit, read-only DuckDB file."""
    with duckdb.connect(str(warehouse), read_only=True) as con:
        return con.execute("select * from events").df()


def prepare_market(events: pd.DataFrame, market: str) -> pd.DataFrame:
    """Prepare within-market/date features, then add market identifiers."""
    ml, _ = dataset.prepare(events)
    return ml.assign(market=market, is_de=1.0 if market == "de" else 0.0)


def common_features(ml_us: pd.DataFrame, ml_de: pd.DataFrame, feature_set: str) -> list[str]:
    """Intersect the usable feature sets, preserving the German order."""
    us_features = set(dataset.feature_sets(ml_us)[feature_set])
    return [c for c in dataset.feature_sets(ml_de)[feature_set] if c in us_features]


def stack(frames: list[pd.DataFrame], features: list[str]) -> pd.DataFrame:
    """Stack frames after padding missing feature columns with dummy zeros."""
    padded = [frame.assign(**{c: 0.0 for c in features if c not in frame}) for frame in frames]
    return pd.concat(padded, ignore_index=True, sort=False)


def pooled_comparison(ml_us: pd.DataFrame, ml_de: pd.DataFrame, params: dict[str, dict],
                      learners: tuple[str, ...] = ("lgbm", "rf"), feature_set: str | None = None) -> pd.DataFrame:
    """Score four training variants on DE validation and DE test; choose on validation only."""
    if feature_set is None:
        feature_set = config.research_settings(config.MARKETS["de"]).feature_set
    common = common_features(ml_us, ml_de, feature_set)
    us = ml_us[ml_us[TARGET].notna()]
    de = ml_de[ml_de[TARGET].notna()]
    us_same = us[us["event_family"].isin(SAME_FAMILIES)]
    va = de[de["split"] == "validation"]
    te = de[de["split"] == "test"]
    sources = ([de], [us, de], [us_same, de], [us_same])
    rows = []
    for variant, frames in zip(VARIANTS, sources):
        feats = common + ["is_de"] if variant in ("pooled_all", "pooled_same_families") else common
        tr = stack([frame[frame["split"] == "train"] for frame in frames], feats)
        trva = stack([frame[frame["split"].isin(["train", "validation"])] for frame in frames], feats)
        for kind in learners:
            t0 = time.perf_counter()
            prm = params.get(kind, {})
            row = {"variant": variant, "learner": kind, "feature_set": feature_set, "n_features": len(feats),
                   "n_train_us": int(tr["market"].eq("us").sum()), "n_train_de": int(tr["market"].eq("de").sum())}
            row.update(research.score_block(va, research.fit_predict(tr, va, feats, kind, prm, "pooled"), "val"))
            row.update(research.score_block(te, research.fit_predict(trva, te, feats, kind, prm, "pooled"), "test"))
            row["seconds"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
    comparison = pd.DataFrame(rows)
    winner = research.pick_variant(comparison, eligible=("rf", "lgbm"))
    comparison["winner"] = comparison.index == winner.name
    for row in comparison.to_dict(orient="records"):
        log.info("pooled compare %s", row)
    return comparison


def run(us_warehouse: Path, de_warehouse: Path, out: Path, tuning: Path | None = None) -> pd.DataFrame:
    """Read explicit warehouses and optional tuning, then write only the comparison parquet."""
    if out.resolve() in {us_warehouse.resolve(), de_warehouse.resolve()}:
        raise ValueError("out must differ from the input warehouses")
    ml_us = prepare_market(load_events(us_warehouse), "us")
    ml_de = prepare_market(load_events(de_warehouse), "de")
    params = {}
    if tuning is not None and tuning.exists():
        table = pd.read_parquet(tuning)
        table["params"] = table["params"].map(json.loads)
        params = models.best_params(table)
    comparison = pooled_comparison(ml_us, ml_de, params)
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(out, index=False)
    return comparison


def main() -> None:
    """Compare explicit US and German warehouses without accessing configured storage."""
    ap = argparse.ArgumentParser(description="US+DE vs German-only models, chosen on German VALIDATION")
    ap.add_argument("--us-warehouse", type=Path, required=True)
    ap.add_argument("--de-warehouse", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tuning", type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.us_warehouse, args.de_warehouse, args.out, args.tuning)


if __name__ == "__main__":
    main()
