"""Static results report (OPM SD7 · Reporting): artifacts -> data/site/ for GitHub Pages.

Writes `index.html` (self-contained, no JS), `summary.md` (for the Actions job summary),
`signals.json`, `events_recent.json` and `metrics.json` (machine-readable, so a 24/7 agent can
read results from Pages even when the API is not deployed).

    python -m driftrev.report
"""

from __future__ import annotations

import html
import json
import shutil
from datetime import datetime, timezone

import pandas as pd

from . import artifacts, config

SIG_COLS = ["date", "ticker", "company", "event_family", "ret_1d", "prob", "tier", "expected_net_return",
            "entry_ref_price", "stop_loss_price", "take_profit_price", "exit_by_date", "edgar_category"]
PCT = {"ret_1d", "expected_net_return", "cagr", "max_drawdown", "win_rate", "avg_trade_ret", "volatility"}


def _fmt(col: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.1%}" if col in PCT else f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


def _table_html(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    if df.empty or not cols:
        return "<p class='muted'>No rows.</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(c, r[c]))}</td>" for c in cols) + "</tr>"
                   for _, r in df.iterrows())
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _table_md(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    if df.empty or not cols:
        return "_No rows._\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(c, r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


def build() -> dict:
    m = artifacts.load_metrics()
    sig = artifacts.load_frame("signals")
    ev = artifacts.load_frame("events")
    strat = pd.DataFrame((m.get("strategies") or []) + (m.get("benchmarks") or []))
    if not strat.empty:
        strat = strat.drop(columns=["report_card"], errors="ignore").sort_values("cagr", ascending=False)
    actionable = ev[ev["tier"].isin(["STRONG_BUY", "BUY"])].head(25) if not ev.empty else ev
    reg = m.get("regime") or {}
    model = m.get("model") or {}
    guide = m.get("guide") or {}
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scols = ["name", "cagr", "sharpe", "max_drawdown", "n_trades", "win_rate"]

    md = [f"# Drift-or-Reversal · results for {m.get('data_through', '?')}",
          f"Generated {when}. Regime **{reg.get('state', '?')}** ({reg.get('points', '?')} pts). "
          f"Model **{model.get('name', '?')}**, threshold {model.get('threshold', '?')}, "
          f"test AUC {model.get('test_auc', float('nan')):.3f}.\n",
          "## Signals for the latest session", _table_md(sig, SIG_COLS),
          "## Actionable signals, last 90 days (STRONG_BUY / BUY)", _table_md(actionable, SIG_COLS),
          "## Strategies vs benchmarks (TEST 2024 →)", _table_md(strat, scols),
          f"## Strategy-guide causal test\n{guide.get('verdict', '–')}\n",
          "_Research output, not investment advice. Paper trading only._\n"]
    summary = "\n".join(md)

    css = """:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--accent:#0a66c2}
@media (prefers-color-scheme:dark){:root{--bg:#111214;--fg:#eceef1;--muted:#9a9ca1;--line:#2a2c30;--accent:#6aa8ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)}.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:140px}.kpi b{display:block;font-size:20px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}
a{color:var(--accent)}"""
    best = strat.iloc[0] if not strat.empty else {}
    kpis = [("Data through", m.get("data_through", "?")), ("Regime", reg.get("state", "?")),
            ("Signals today", len(sig)), ("Universe", m.get("universe_size", "?")),
            ("Best CAGR (test)", f"{best.get('name', '?')}: {_fmt('cagr', best.get('cagr'))}" if len(strat) else "–")]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Drift or Reversal</title><style>{css}</style></head><body>
<h1>Drift or Reversal? Daily results</h1>
<p class="muted">Event-driven ML on the S&amp;P 1500 (earnings drift · crash reversal · momentum breakout). Generated {when} by GitHub Actions.
Machine-readable: <a href="signals.json">signals.json</a> · <a href="events_recent.json">events_recent.json</a> · <a href="metrics.json">metrics.json</a></p>
<div class="kpis">{''.join(f'<div class="kpi"><span class="muted">{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>' for k, v in kpis)}</div>
<h2>Signals for the latest session</h2>{_table_html(sig, SIG_COLS)}
<h2>Actionable signals, last 90 days</h2>{_table_html(actionable, SIG_COLS)}
<h2>Strategies vs benchmarks (out-of-sample test, 2024 →)</h2>{_table_html(strat, scols)}
<h2>Strategy-guide causal test</h2><p>{html.escape(guide.get('verdict', '–'))}</p>
<p class="muted">Research output, not investment advice. Orders are paper-only and disabled by default.</p>
</body></html>"""

    site = config.SITE_DIR
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    (site / "index.html").write_text(page)
    (site / "summary.md").write_text(summary)
    (site / "signals.json").write_text(json.dumps(artifacts.to_records(sig), default=str))
    (site / "events_recent.json").write_text(json.dumps(artifacts.to_records(ev), default=str))
    (site / "metrics.json").write_text(json.dumps(m, default=str, indent=1))
    (site / ".nojekyll").write_text("")
    return {"site": str(site), "signals": len(sig), "actionable_90d": len(actionable)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
