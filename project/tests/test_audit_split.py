"""Audit F: temporal split purge (train/validation labels must not be realised inside the next block)."""

import numpy as np
import pandas as pd

from driftrev import config, dataset


def _dates():
    return pd.Series(pd.bdate_range("2021-10-01", "2024-02-29"))


def test_train_labels_do_not_reach_into_validation():
    d = _dates()
    sp = dataset.temporal_split(d)
    s = config.SETTINGS.split
    gap = pd.Timedelta(days=s.purge_days)
    assert (d[sp == "train"] <= pd.Timestamp(s.train_end) - gap).all()
    assert (d[sp == "validation"] <= pd.Timestamp(s.valid_end) - gap).all()
    assert (sp == "purged").sum() > 20                      # both boundaries purged
    assert set(sp.unique()) == {"train", "validation", "test", "purged"}
    # test block untouched
    assert (d[sp == "test"] > pd.Timestamp(s.valid_end)).all() and (d[d > pd.Timestamp(s.valid_end)].index.isin(sp[sp == "test"].index)).all()


def test_real_10_session_label_closes_before_next_block():
    """With real trading days: the 10th session after any kept train/validation row falls inside its block."""
    cal = pd.bdate_range("2021-06-01", "2024-03-31")
    sp = dataset.temporal_split(pd.Series(cal))
    pos = {d: i for i, d in enumerate(cal)}
    s = config.SETTINGS.split
    for blk, end in (("train", s.train_end), ("validation", s.valid_end)):
        for d in cal[(sp.to_numpy() == blk)][-3:]:
            assert cal[pos[d] + config.SETTINGS.target.horizon] <= pd.Timestamp(end) + pd.Timedelta(days=0)


def test_purge_can_be_disabled(monkeypatch):
    import dataclasses
    off = dataclasses.replace(config.SETTINGS, split=dataclasses.replace(config.SETTINGS.split, purge=False))
    monkeypatch.setattr(dataset, "S", off)
    sp = dataset.temporal_split(_dates())
    assert "purged" not in set(sp.unique())
    # and without the purge the leak the audit found is there: last train row's label closes in validation
    cal = pd.bdate_range("2021-06-01", "2024-03-31")
    sp2 = dataset.temporal_split(pd.Series(cal))
    last_train = cal[(sp2.to_numpy() == "train")][-1]
    assert cal[list(cal).index(last_train) + config.SETTINGS.target.horizon] > pd.Timestamp(config.SETTINGS.split.train_end)


def test_walkforward_never_fits_on_prediction_year_or_unclosed_labels(monkeypatch):
    from driftrev import research, walkforward
    rng = np.random.default_rng(0)
    dates = np.sort(rng.choice(pd.bdate_range("2014-01-01", "2020-12-31"), 12000))
    ml = pd.DataFrame({"date": dates, "f": rng.normal(size=len(dates)), dataset.TARGET: rng.integers(0, 2, len(dates)).astype(float)})
    calls = []

    def fake_fit_predict(tr, pr, features, kind, params, layout="pooled"):
        calls.append((tr["date"].max(), pr["date"].min(), pr["date"].max()))
        return np.full(len(pr), 0.5)

    monkeypatch.setattr(research, "fit_predict", fake_fit_predict)
    walkforward.run(ml, ["f"], "lgbm", {}, first_year=2018)
    assert calls
    purge = pd.Timedelta(days=config.SETTINGS.research.wf_purge_days)
    for tr_max, pr_min, _ in calls:
        assert tr_max + purge <= pr_min + pd.Timedelta(days=1)          # label (<= 16 calendar days) closed before the block starts
    # pairs: (threshold model trained -> validation year) then (prediction model -> prediction year)
    for (_tr_a, _v_min, v_max), (tr_b, p_min, p_max) in zip(calls[0::2], calls[1::2]):
        assert v_max.year == p_min.year - 1 and p_min.year == p_max.year and tr_b.year < p_min.year
