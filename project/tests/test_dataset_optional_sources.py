"""Optional market sources keep the ML schema but contribute no empty features."""

import logging

import numpy as np
import pandas as pd
import pytest
from test_de_pipeline import _build, _world

from driftrev import config, dataset


@pytest.fixture(scope="module")
def german_events():
    events, _, _ = _build(_world())
    assert not events.empty
    return events


def test_prepare_accepts_missing_us_sources(german_events, caplog):
    original = german_events.copy(deep=True)
    missing = [c for c in dataset.NUMERIC if c not in german_events]
    assert len(missing) == 50
    with caplog.at_level(logging.INFO, logger=dataset.__name__):
        ml, feats = dataset.prepare(german_events)
    assert set(dataset.NUMERIC) <= set(ml.columns)
    assert ml[missing].isna().all().all()
    assert ml[missing].dtypes.eq(np.dtype("float64")).all()
    assert feats and ml[feats].notna().any().all()
    assert not any(c.startswith(("att_", "item_", "form_", "ft_")) for c in feats)
    assert ml["bin_attention_nan"].eq(1).all()
    pd.testing.assert_frame_equal(german_events, original)
    records = [r for r in caplog.records if r.name == dataset.__name__]
    assert len(records) == 2
    assert records[0].levelno == records[1].levelno == logging.INFO
    assert str(len(missing)) in records[0].getMessage()
    assert all(c in records[0].getMessage() for c in missing)
    dropped = [c for c in dataset.numeric_features() if ml[c].notna().sum() == 0]
    assert str(len(dropped)) in records[1].getMessage()
    assert all(c in records[1].getMessage() for c in dropped)


def test_feature_sets_drop_only_all_nan(caplog):
    ml = pd.DataFrame({"ret_1d": [np.nan, np.nan], "abn_ret_1d": [np.nan, 0.0],
                       "sector_unknown": [1, 1], "month_wom_Jan_w1": [1, 1],
                       "cs_rank_ret_1d": [np.nan, np.nan], "cs_dm_abn_ret_1d": [np.nan, 0.0]})
    with caplog.at_level(logging.INFO, logger=dataset.__name__):
        sets = dataset.feature_sets(ml)
    assert set(sets) == set(dataset.FEATURE_SETS)
    for feats in sets.values():
        assert "ret_1d" not in feats and "cs_rank_ret_1d" not in feats
        assert "abn_ret_1d" in feats and "sector_unknown" in feats
        assert ml[feats].notna().any().all()
    assert "month_wom_Jan_w1" in sets["pooled_wom"]
    assert "cs_dm_abn_ret_1d" in sets["cs_keep_market"]
    assert "cs_dm_abn_ret_1d" in sets["cs_drop_market"]
    records = [r for r in caplog.records if r.name == dataset.__name__]
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert all(c in records[0].getMessage() for c in ("2", "ret_1d", "cs_rank_ret_1d"))


def test_us_frame_unchanged(german_events, caplog):
    events = german_events.reindex(columns=list(german_events) + [c for c in dataset.NUMERIC if c not in german_events])
    events[dataset.NUMERIC] = events[dataset.NUMERIC].astype("float64").fillna(0.0)
    assert events[dataset.NUMERIC].notna().any().all()
    # Rebuild pre-change prepare from its private pieces and original feature lists.
    # All NUMERIC inputs have values, so the added padding/filtering must be a no-op.
    d = dataset.add_engineered(events)
    dummies, dummy_cols = dataset.make_dummies(d)
    num = dataset.numeric_features()
    X = d[num].astype("float64").replace([np.inf, -np.inf], np.nan)
    keep = ["ticker", "company", "date", "sector", "cap_bucket", "event_family", "edgar_category", "edgar_items",
            "regime_state", "open", "close", "next_date", "sue_is_current", "atr_pct"] + [c for c in dataset.TO_PREDICT if c in d]
    expected = pd.concat([d[keep].reset_index(drop=True), X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    expected = expected.loc[:, ~expected.columns.duplicated()]
    expected["split"] = dataset.temporal_split(expected["date"])
    expected = dataset.add_cross_sectional(expected)
    wom = [c for c in dummy_cols if c.startswith("month_wom_")]
    base_dummies = [c for c in dummy_cols if c not in wom]
    cs = [c for c in expected.columns if c.startswith(("cs_rank_", "cs_dm_"))]
    pooled = num + base_dummies
    stock_num = [c for c in num if c not in dataset.MARKET_NUMERIC]
    stock_dummies = [c for c in base_dummies if not c.startswith(dataset.MARKET_DUMMY_PREFIXES)]
    expected_sets = {"pooled": pooled, "pooled_wom": pooled + wom, "cs_keep_market": pooled + cs,
                     "cs_drop_market": stock_num + stock_dummies + cs}
    with caplog.at_level(logging.INFO, logger=dataset.__name__):
        ml, feats = dataset.prepare(events)
        actual_sets = dataset.feature_sets(ml)
    assert actual_sets == expected_sets
    assert feats == expected_sets[config.research_settings().feature_set]
    pd.testing.assert_frame_equal(ml, expected)
    assert not [r for r in caplog.records if r.name == dataset.__name__]
