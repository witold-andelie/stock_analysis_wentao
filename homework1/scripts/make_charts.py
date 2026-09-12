"""
make_charts.py - Module 1 homework interactive dashboard (offline).

Reads the cached data (data/raw/*.parquet outputs) and the analysis CSVs
(data/outputs/*.csv) produced by homework/solution.py and renders:

  data/outputs/dashboard.html   - single self-contained dashboard (inline plotly.js)
  data/outputs/answers.json     - final answers as machine-readable summary
  data/outputs/charts/*.html    - one standalone html per chart

Usage:
    py scripts/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "outputs"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

pio.templates.default = "plotly_white"

PAL = {
    "blue": "#4C72B0",
    "orange": "#E4572E",
    "green": "#2E9E5B",
    "red": "#C0392B",
    "gold": "#F2A900",
    "gray": "#8A94A6",
}

GENERATED = "2026-09-13"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


# ----------------------------------------------------------------------------
# Q1 - additions per year
# ----------------------------------------------------------------------------
def fig_q1() -> go.Figure:
    y = read("q1_additions_by_year.csv")
    d = y[y.added_year.between(2020, 2026)].reset_index(drop=True)
    colors = [PAL["orange"] if yr == 2025 else PAL["blue"] for yr in d["added_year"]]

    fig = go.Figure(
        go.Bar(
            x=d["added_year"],
            y=d["count"],
            marker_color=colors,
            text=[f"{v}" for v in d["count"]],
            textposition="outside",
            hovertemplate="Year %{x}<br>Additions %{y}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=2025,
        y=18,
        yshift=22,
        showarrow=False,
        text="most since 2020",
        font=dict(color=PAL["orange"], size=13),
    )
    fig.update_layout(
        title=dict(
            text="Q1 · S&P 500 index additions per year (2020–2026, YTD)",
            x=0.02,
        ),
        height=430,
        margin=dict(l=60, r=40, t=90, b=50),
        yaxis=dict(title="Number of additions", range=[-1, 21]),
        xaxis=dict(title="Year added", dtick=1),
        showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------------
# Q2 - world indices YTD
# ----------------------------------------------------------------------------
def fig_q2() -> go.Figure:
    w = read("q2_world_indices_ytd.csv")
    us_ret = w.loc[w.ticker == "^GSPC", "ytd_return_pct"].iloc[0]
    w = w.sort_values("ytd_return_pct")

    colors = []
    for _, r in w.iterrows():
        if r.ticker == "^GSPC":
            colors.append(PAL["gold"])
        elif r.ytd_return_pct > us_ret:
            colors.append(PAL["green"])
        else:
            colors.append(PAL["gray"])

    fig = go.Figure(
        go.Bar(
            x=w.ytd_return_pct,
            y=w.country_index,
            orientation="h",
            marker_color=colors,
            text=w.ytd_return_pct.map("{:.2f}%".format),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>YTD %{x:.2f}%<br>"
                "%{customdata[0]} → %{customdata[1]}<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    w.first_close.map("{:,.2f}".format),
                    w.last_close.map("{:,.2f}".format),
                ]
            ),
        )
    )
    fig.add_annotation(
        x=7,
        y=10.4,
        showarrow=False,
        text=f"2 of {len(w)} markets beat the S&P 500 (+{us_ret:.2f}%)",
        font=dict(size=13, color="#1f2933"),
    )
    fig.update_layout(
        title=dict(
            text="Q2 · World indices YTD return vs S&P 500 (2026-01-01 → 2026-08-21)",
            x=0.02,
        ),
        height=480,
        margin=dict(l=140, r=60, t=70, b=50),
        yaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=list(reversed(w.country_index)),
        ),
        xaxis=dict(
            title="YTD return (%)",
            range=[w.ytd_return_pct.min() - 2, w.ytd_return_pct.max() + 4],
        ),
        showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------------
# Q3 - corrections timeline + distribution
# ----------------------------------------------------------------------------
def _load_q3() -> pd.DataFrame:
    c = read("q3_corrections_all.csv")
    c["peak_dt"] = pd.to_datetime(c.peak_date)
    c["trough_dt"] = pd.to_datetime(c.trough_date)
    return c


def fig_q3a() -> go.Figure:
    c = _load_q3()
    top = c.nlargest(5, "drawdown_pct")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=c.trough_dt,
            y=c.drawdown_pct,
            marker_color=PAL["blue"],
            opacity=0.75,
            name="corrections ≥ 5%",
            customdata=np.column_stack(
                [
                    c.peak_dt.dt.strftime("%Y-%m-%d"),
                    c.duration_days.astype(str),
                ]
            ),
            hovertemplate=(
                "Trough %{x|%Y-%m-%d}<br>From peak %{customdata[0]}<br>"
                "Drawdown %{y:.2f}%<br>Duration %{customdata[1]} days<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=top.trough_dt,
            y=top.drawdown_pct,
            mode="markers+text",
            text=top.drawdown_pct.map("{:.1f}%".format),
            textposition="top center",
            marker=dict(color=PAL["red"], size=10, line=dict(color="white", width=1)),
            name="largest 5",
            customdata=np.column_stack(
                [top.peak_dt.dt.strftime("%Y-%m-%d"), top.duration_days.astype(str)]
            ),
            hovertemplate=(
                "Trough %{x|%Y-%m-%d}<br>From peak %{customdata[0]}<br>"
                "Drawdown %{y:.2f}%<br>Duration %{customdata[1]} days<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=5, line_dash="dot", line_color=PAL["gray"], annotation_text="threshold 5%"
    )
    fig.update_layout(
        title=dict(
            text=f"Q3 · S&P 500 corrections ≥ 5% since 1950 ({len(c)} episodes)", x=0.02
        ),
        height=430,
        margin=dict(l=60, r=140, t=70, b=50),
        yaxis=dict(title="Drawdown from prior all-time high (%)", range=[0, 62]),
        xaxis=dict(title="Trough date"),
        legend=dict(orientation="h", y=1.08, x=0.4),
    )
    return fig


def fig_q3b() -> go.Figure:
    c = _load_q3()
    med_dd = float(c.drawdown_pct.median())
    med_dur = float(c.duration_days.median())

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=("Drawdown depth (%)", "Trough-to-peak duration (days)"),
    )
    fig.add_trace(
        go.Histogram(
            x=c.drawdown_pct,
            nbinsx=16,
            marker_color=PAL["blue"],
            opacity=0.85,
            name="depth",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=c.duration_days,
            nbinsx=16,
            marker_color=PAL["blue"],
            opacity=0.85,
            name="duration",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=med_dd, row=1, col=1, line_dash="dash", line_color=PAL["orange"])
    fig.add_vline(x=med_dur, row=1, col=2, line_dash="dash", line_color=PAL["orange"])
    fig.add_annotation(
        x=med_dd,
        y=17,
        xref="x",
        yref="y",
        showarrow=False,
        text=f"median {med_dd:.2f}%",
        font=dict(color=PAL["orange"], size=12),
    )
    fig.add_annotation(
        x=med_dur,
        y=17,
        xref="x2",
        yref="y2",
        showarrow=False,
        text=f"median {med_dur:.0f} days",
        font=dict(color=PAL["orange"], size=12),
    )
    fig.update_layout(
        title=dict(
            text=f"Q3 · Distribution of the {len(c)} significant corrections", x=0.02
        ),
        height=420,
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Episodes")
    return fig


# ----------------------------------------------------------------------------
# Q4 - AMZN earnings surprise vs 2-day return
# ----------------------------------------------------------------------------
def fig_q4a() -> go.Figure:
    q = read("q4_amzn_earnings_returns_full.csv")
    q["earnings_dt"] = pd.to_datetime(q.earnings_date, utc=True)
    xclip = np.clip(q.surprise_pct, -100, 100)
    pos = q.surprise_pct > 0
    colors = np.where(pos, PAL["green"], PAL["red"])

    fig = go.Figure(
        go.Scatter(
            x=xclip,
            y=q.return_2d_pct,
            mode="markers",
            marker=dict(color=colors, size=9, line=dict(color="white", width=1)),
            name="earnings events",
            customdata=np.column_stack(
                [
                    q.earnings_dt.dt.strftime("%Y-%m-%d"),
                    q.surprise_pct.map("{:.2f}%".format),
                    q.reported_eps.map("{:.2f}".format),
                ]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>surprise %{customdata[1]}<br>"
                "reported EPS %{customdata[2]}<br>2-day return %{y:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color=PAL["gray"], line_width=1)
    fig.add_hline(y=0, line_color=PAL["gray"], line_width=1)
    fig.add_annotation(
        x=60,
        y=11,
        showarrow=False,
        text="49 events · corr(surprise, 2d return) = 0.079 (weak)",
        font=dict(size=12, color="#374151"),
    )
    fig.update_layout(
        title=dict(
            text="Q4 · AMZN 2-day forward return vs earnings surprise (green = positive surprise)",
            x=0.02,
        ),
        height=440,
        margin=dict(l=60, r=40, t=70, b=50),
        xaxis=dict(
            title="Reported surprise (%) — one outlier (+215%) clipped to the axis"
        ),
        yaxis=dict(title="2-day forward return (%)", range=[-14, 14]),
        showlegend=False,
    )
    return fig


def fig_q4b() -> go.Figure:
    p = read("q4_amzn_positive_surprises.csv")
    med = float(p.return_2d_pct.median())
    mean = float(p.return_2d_pct.mean())

    fig = go.Figure(
        go.Histogram(
            x=p.return_2d_pct,
            nbinsx=14,
            marker_color=PAL["green"],
            opacity=0.85,
            hovertemplate="2-day return %{x:.2f}%<br>events %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=med, line_dash="dash", line_color=PAL["orange"])
    fig.add_vline(x=mean, line_dash="dot", line_color=PAL["blue"])
    fig.add_annotation(
        x=med,
        y=7.5,
        showarrow=False,
        text=f"median +{med:.2f}%",
        font=dict(color=PAL["orange"], size=13),
    )
    fig.add_annotation(
        x=mean,
        y=7.0,
        showarrow=False,
        text=f"mean +{mean:.2f}%",
        font=dict(color=PAL["blue"], size=12),
    )
    fig.update_layout(
        title=dict(
            text=f"Q4 · 2-day return after a positive surprise — {len(p)} events",
            x=0.02,
        ),
        height=420,
        margin=dict(l=60, r=40, t=70, b=50),
        xaxis=dict(title="2-day forward return (%)"),
        yaxis=dict(title="Events"),
        showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------------
# dashboard assembly + standalone files
# ----------------------------------------------------------------------------
FIGURES = [
    ("fig1_q1_additions", "Q1", fig_q1),
    ("fig2_q2_ytd", "Q2", fig_q2),
    ("fig3a_q3_timeline", "Q3", fig_q3a),
    ("fig3b_q3_hist", "Q3", fig_q3b),
    ("fig4a_q4_scatter", "Q4", fig_q4a),
    ("fig4b_q4_hist", "Q4", fig_q4b),
]

ANSWERS = {
    "q1": {
        "answer": "2025 - most additions since 2020 (18)",
        "bonus": "227 current members in index > 20 years",
    },
    "q2": {
        "answer": "2 of 11 world indices beat S&P 500 YTD",
        "owners": ["Japan - Nikkei 225 (+27.36%)", "Canada - S&P/TSX (+14.86%)"],
        "sp500_ytd": "+11.90%",
    },
    "q3": {"episodes": 74, "median_drawdown_pct": 7.99, "median_duration_days": 40.5},
    "q4": {
        "positive_events": 36,
        "median_2d_return_pct": 0.51,
        "corr_all": 0.0786,
        "corr_positive": 0.0452,
    },
}


def build_dashboard(divs: list[str]) -> str:
    answers_rows = "".join(
        f"<tr><td class='q'>{k}</td><td>{v}</td></tr>" for k, v in ANSWERS.items()
    )
    cards = []
    for name, qlabel, _ in FIGURES:
        div = divs[name]
        cards.append(f"<div class='card'>{div}</div>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Module 1 Homework - Dashboard</title>
<style>
  body {{ margin:0; font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         background:#f5f6fa; color:#1f2933; }}
  header {{ background:linear-gradient(135deg,#0f4c81,#163a62); color:#fff;
           padding:26px 34px; }}
  header h1 {{ margin:0 0 6px; font-size:24px; }}
  header p {{ margin:2px 0; opacity:.85; font-size:13px; }}
  .answers {{ border-collapse:collapse; margin-top:14px; width:100%; max-width:1000px;
              background:rgba(255,255,255,.08); border-radius:8px; overflow:hidden; }}
  .answers td {{ padding:7px 12px; font-size:13px; border-bottom:1px solid rgba(255,255,255,.12); }}
  .answers tr:last-child td {{ border-bottom:none; }}
  .answers td.q {{ font-weight:600; width:56px; color:#ffd166; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px;
           padding:18px 24px 30px; max-width:1500px; margin:0 auto; }}
  .card {{ background:#fff; border-radius:10px; box-shadow:0 2px 10px rgba(15,40,80,.10);
           padding:8px 10px; }}
  footer {{ text-align:center; color:#64748b; font-size:12px; padding:6px 0 22px; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Stock Market Analytics Zoomcamp — Module 1 (Week 1) Homework</h1>
  <p>Interactive dashboard · generated {GENERATED} · data via Wikipedia &amp; yfinance · scripts:
     <code>homework/solution.py</code> (data + answers), <code>scripts/make_charts.py</code> (charts)</p>
  <table class="answers">
    <tr><td class="q">Q1</td><td>Year with most index additions since 2020: <b>2025 (18) · bonus</b>: 227 current members in the index for more than 20 years</td></tr>
    <tr><td class="q">Q2</td><td><b>2 of 11</b> world indices beat the S&amp;P 500 YTD: Nikkei 225 <b>+27.36%</b>, TSX <b>+14.86%</b> vs S&amp;P 500 <b>+11.90%</b></td></tr>
    <tr><td class="q">Q3</td><td>Since 1950: <b>74</b> corrections ≥ 5% · median drawdown <b>7.99%</b> · median duration ≈ <b>40 days</b></td></tr>
    <tr><td class="q">Q4</td><td>Median 2-day return after a positive AMZN surprise: <b>+0.51%</b> (36 events)</td></tr>
  </table>
</header>
<div class="grid">
{"".join(cards)}
</div>
<footer>Interactive charts built with Plotly (offline). Individual charts also saved under data/outputs/charts/.</footer>
</body>
</html>"""
    return html


def main() -> None:
    divs: dict[str, str] = {}
    for i, (name, _, builder) in enumerate(FIGURES):
        fig = builder()
        include = "inline" if i == 0 else False
        divs[name] = pio.to_html(
            fig,
            include_plotlyjs=include,
            full_html=False,
            config={"displaylogo": False, "responsive": True},
        )
        fig.write_html(
            CH / f"{name}.html",
            include_plotlyjs=True,
            config={"displaylogo": False, "responsive": True},
        )

    (OUT / "dashboard.html").write_text(build_dashboard(divs), encoding="utf-8")
    (OUT / "answers.json").write_text(
        json.dumps(ANSWERS, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"charts -> {CH}")
    print(f"dashboard -> {OUT / 'dashboard.html'}")
    print(f"answers -> {OUT / 'answers.json'}")


if __name__ == "__main__":
    main()
