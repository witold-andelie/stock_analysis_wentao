"""R4 Classifying Filing Text: lexicon, negation, exhibit choice, tone, point-in-time aggregation."""

import numpy as np
import pandas as pd

from driftrev import filing_text as F

INDEX_HTML = """
<html><body><table summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/1/000000000126000001/d8k.htm">d8k.htm</a></td><td>8-K</td><td>10</td></tr>
<tr><td>2</td><td>PR</td><td><a href="/ix?doc=/Archives/edgar/data/1/000000000126000001/ex99.htm">ex99.htm</a></td><td>EX-99.1</td><td>10</td></tr>
<tr><td>3</td><td>slides</td><td><a href="/Archives/edgar/data/1/000000000126000001/ex992.htm">ex992.htm</a></td><td>EX-99.2</td><td>10</td></tr>
</table></body></html>"""


def flags(text):
    f, _ = F.find_flags(F.normalise(text))
    return {k for k, v in f.items() if v}


def test_guidance_flags_and_negation():
    assert "guidance_cut" in flags("The Company lowered its full-year guidance to $1.00.")
    assert "guidance_cut" in flags("Management now expects revenue to be below the prior range.")
    assert "guidance_cut" in flags("The company withdraws its financial guidance.")
    assert "guidance_cut" not in flags("The company does not lower its guidance.")
    assert "guidance_raise" in flags("We are raising our full year outlook.")
    assert "guidance_raise" not in flags("We are not raising our outlook.")
    assert "guidance_reaffirm" in flags("The Company reaffirms its fiscal 2025 guidance.")


def test_distress_flags():
    assert "going_concern" in flags("There is substantial doubt about our ability to continue as a going concern.")
    assert "going_concern" not in flags("Management concluded there is no substantial doubt about going concern.")
    assert "material_weakness" in flags("We identified a material weakness in internal control.")
    assert "material_weakness" not in flags("There was no material weakness.")
    assert "clinical_failure" in flags("The Phase 3 trial did not meet its primary endpoint.")
    assert "clinical_failure" in flags("The FDA issued a Complete Response Letter.")
    assert "exec_departure" in flags("The Chief Financial Officer resigned effective immediately.")
    assert "covenant" in flags("The Company obtained a waiver of the leverage covenant.")
    assert "investigation" in flags("The company received a subpoena from the SEC.")
    assert "beat" in flags("Record quarterly revenue exceeded analyst expectations.")
    assert "miss" in flags("Results fell short of expectations.")
    assert flags("Quarterly dividend declared.") == set()


def test_offering_and_discount():
    txt = F.normalise("Announces pricing of $50 million public offering. The public offering price is $4.50 per share. "
                      "The last reported sale price of our common stock was $5.00 per share.")
    f, snips = F.find_flags(txt)
    assert f["offering_priced"] == 1 and "offering_priced" in snips
    off, last, disc = F.parse_offering_discount(txt)
    assert (off, last) == (4.5, 5.0) and abs(disc - 0.10) < 1e-9


def test_exhibit_selection():
    docs = F.parse_index(INDEX_HTML)
    assert [d["type"] for d in docs] == ["8-K", "EX-99.1", "EX-99.2"]
    typ, url = F.choose_exhibit(docs, "8-K")
    assert typ == "EX-99.1" and url == "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/ex99.htm"
    only_primary = [d for d in docs if d["type"] == "8-K"]
    assert F.choose_exhibit(only_primary, "8-K")[0] == "8-K"
    docs_ex99 = [d for d in docs if d["type"] != "EX-99.1"]
    assert F.choose_exhibit(docs_ex99, "8-K")[0] == "EX-99.2"
    assert F.choose_exhibit(docs, "424B5") is None or F.choose_exhibit(docs, "424B5")[0] != "EX-99.1"
    assert F.index_url(1, "0000000001-26-000001").endswith("/000000000126000001/0000000001-26-000001-index.htm")


def test_tone_curated_and_directional():
    lm = {"negative": {"loss", "decline"}, "positive": {"gain"}, "uncertainty": {"may"}, "source": "t"}
    t = F.lm_tone("Loss and decline may occur. A gain.", lm)
    assert abs(t["lm_negative_pct"] - 100 * 2 / 7) < 1e-9
    assert t["lm_net_tone"] == (1 - 2) / 3 and t["lm_uncertainty_pct"] > 0
    assert np.isnan(F.lm_tone("", lm)["lm_net_tone"])
    real = F.lm_tone("Record growth and strong gains.", {k: set(v) for k, v in F._CURATED.items()})
    assert real["lm_net_tone"] > 0


def _fixture():
    ev = pd.DataFrame({
        "ticker": ["A", "B", "C"], "cik": [1, 2, 3],
        "date": pd.to_datetime(["2023-05-02"] * 3),
        "prev_date": pd.to_datetime(["2023-05-01"] * 3), "next_date": pd.to_datetime(["2023-05-03"] * 3),
        "edgar_category": ["earnings", "earnings", "offering"],
    }, index=[10, 11, 12])
    fl = pd.DataFrame({
        "accession": ["a1", "a2", "b1", "c1", "c2"], "cik": [1, 1, 2, 3, 3],
        "form": ["8-K", "8-K", "8-K", "424B5", "NT 10-Q"], "items": ["2.02", "7.01", "2.02", "", ""],
        "accepted_et": pd.to_datetime(["2023-05-01 16:30", "2023-05-03 09:30",  # a2 is AFTER 09:00 next day: excluded
                                       "2023-05-01 15:59",                       # b1 before prev close 16:00: excluded
                                       "2023-05-02 07:00", "2023-05-02 08:00"]),
    })
    base = dict.fromkeys(F.FLAG_NAMES, 0)
    ft = pd.DataFrame([
        {"accession": "a1", "cik": 1, "status": "ok", "exhibit_type": "EX-99.1", "n_chars": 900, **base | {"guidance_cut": 1},
         "lm_negative_pct": 3.0, "lm_positive_pct": 1.0, "lm_uncertainty_pct": 1.0, "lm_net_tone": -0.5, "offering_discount": np.nan},
        {"accession": "a2", "cik": 1, "status": "ok", "exhibit_type": "EX-99.1", "n_chars": 900, **base | {"guidance_raise": 1},
         "lm_negative_pct": 0, "lm_positive_pct": 5, "lm_uncertainty_pct": 0, "lm_net_tone": 1.0, "offering_discount": np.nan},
        {"accession": "b1", "cik": 2, "status": "ok", "exhibit_type": "EX-99.1", "n_chars": 900, **base | {"going_concern": 1},
         "lm_negative_pct": 0, "lm_positive_pct": 0, "lm_uncertainty_pct": 0, "lm_net_tone": 0.0, "offering_discount": np.nan},
        {"accession": "c1", "cik": 3, "status": "ok", "exhibit_type": "424B5", "n_chars": 5000, **base | {"offering_priced": 1},
         "lm_negative_pct": 1, "lm_positive_pct": 1, "lm_uncertainty_pct": 1, "lm_net_tone": 0.0, "offering_discount": 0.12},
        {"accession": "c2", "cik": 3, "status": "no_doc", "exhibit_type": None, "n_chars": None, **dict.fromkeys(F.FLAG_NAMES)},
    ])
    return ev, ft, fl


def test_attach_point_in_time_and_category():
    ev, ft, fl = _fixture()
    out = F.attach_filing_text(ev, ft, fl)
    assert list(out.index) == [10, 11, 12]
    assert out.loc[10, "ft_guidance_cut"] == 1 and out.loc[10, "ft_guidance_raise"] == 0  # later 8-K is out of the window
    assert out.loc[10, "edgar_category_text"] == "earnings_guidance_cut"
    assert out.loc[10, "ft_lm_net_tone"] == -0.5
    assert out.loc[11, "ft_text_available"] == 0 and out.loc[11, "ft_going_concern"] == 0  # filed before the window
    assert out.loc[11, "edgar_category_text"] == "earnings_other"
    assert out.loc[12, "ft_offering_priced"] == 1 and out.loc[12, "edgar_category_text"] == "offering_priced"
    assert out.loc[12, "ft_offering_discount"] == 0.12 and out.loc[12, "ft_n_filings_window"] == 2 and out.loc[12, "ft_n_text"] == 1
    assert np.isnan(out.loc[11, "ft_lm_net_tone"])
    for c in F.FT_COLUMNS:
        assert c in out.columns


def test_going_concern_overrides_earnings_and_empty_input():
    ev, ft, fl = _fixture()
    fl2 = fl.copy()
    fl2.loc[fl2.accession == "b1", ["cik", "accepted_et"]] = [2, pd.Timestamp("2023-05-02 10:00")]
    ft2 = ft.copy()
    ft2.loc[ft2.accession == "b1", "cik"] = 2
    ev3 = ev.copy()
    ev3.loc[11, "cik"] = 2
    out = F.attach_filing_text(ev3, ft2, fl2)
    assert out.loc[11, "edgar_category_text"] == "going_concern"
    empty = F.attach_filing_text(ev, pd.DataFrame(), fl)
    assert (empty["ft_text_available"] == 0).all() and list(empty["edgar_category_text"]) == ["earnings_other", "earnings_other", "offering"]


def test_select_targets_scope():
    ev, _, fl = _fixture()
    ev["event_family"] = ["earnings_up", "shock_crash", "earnings_crash"]
    ev["date"] = pd.to_datetime(["2023-05-02"] * 3)
    t = F.select_targets(ev, fl, done={"c1"})
    assert set(t["accession"]) == {"c2", "a1"}  # crash: 8-K/424B/NT only; earnings: 8-K 2.02; done skipped
    assert F.select_targets(ev, fl, include_earnings=False, max_filings=1).shape[0] == 1


def test_refine_category_handles_missing_edgar_category():
    import pandas as pd

    from driftrev.filing_text import FLAG_NAMES, refine_category
    row = pd.Series({"edgar_category": None, **{f"ft_{f}": 0 for f in FLAG_NAMES}})
    assert refine_category(row) == "none"
