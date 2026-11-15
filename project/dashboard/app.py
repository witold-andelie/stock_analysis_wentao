"""Streamlit monitoring dashboard. Reads pipeline ARTIFACTS only (never the DuckDB warehouse).

  PYTHONPATH=src streamlit run dashboard/app.py
Single page with tabs on purpose (no pages/ dir: avoids multipage auto-discovery bugs).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from driftrev import artifacts

st.set_page_config(page_title="Drift or Reversal", layout="wide")


@st.cache_data(ttl=300)
def _frame(name: str) -> pd.DataFrame:
    return artifacts.load_frame(name)


@st.cache_data(ttl=300)
def _metrics() -> dict:
    return artifacts.load_metrics()


def _empty(what: str) -> None:
    st.info(f"No {what} yet. Run the pipeline: `python -m driftrev.pipeline --mode update`.")


m = _metrics()
with st.sidebar:
    st.header("Drift or Reversal")
    if st.button("Refresh"):
        st.cache_data.clear()
    st.caption("Make it live: deploy this repo on Streamlit Cloud (main file `dashboard/app.py`, "
               "artifacts via a synced data dir), or run `docker compose up -d` and open :8501.")

st.title("Drift or Reversal? Signal monitor")
t_over, t_sig, t_eq, t_tr, t_model, t_guide, t_runs = st.tabs(
    ["Overview", "Latest signals", "Equity curves", "Trades", "Model", "Rolling-window guide", "Pipeline runs"])

with t_over:
    if not m:
        _empty("metrics")
    else:
        sig = _frame("signals")
        reg = (m.get("regime") or {})
        c = st.columns(4)
        c[0].metric("Data through", str(m.get("data_through", "?")))
        c[1].metric("Regime", str(reg.get("state") or reg.get("regime_state") or "?"))
        c[2].metric("Signals", len(sig))
        c[3].metric("Universe", m.get("universe_size", "?"))
        strategies = m.get("strategies") or []
        if strategies:
            best = max(strategies, key=lambda s: s.get("sharpe") or -9)
            spy = next((b for b in m.get("benchmarks", []) if "SPY" in str(b.get("name", b.get("strategy", "")))), {})
            st.subheader(f"Best strategy: {best.get('name', best.get('strategy', '?'))}")
            k = st.columns(3)
            for col, key in zip(k, ("cagr", "sharpe", "max_drawdown")):
                v, b = best.get(key), spy.get(key)
                col.metric(key.upper(), f"{v:.2f}" if isinstance(v, (int, float)) else "n/a",
                           f"SPY {b:.2f}" if isinstance(b, (int, float)) else None, delta_color="off")
            table = pd.DataFrame(strategies + (m.get("benchmarks") or [])).drop(columns=["report_card"], errors="ignore")
            st.dataframe(table, use_container_width=True)

with t_sig:
    sig = _frame("signals")
    if sig.empty:
        _empty("signals")
    else:
        tiers = st.multiselect("Tier", sorted(sig["tier"].dropna().unique()), default=[t for t in ("STRONG_BUY", "BUY") if t in set(sig["tier"])])
        view = sig[sig["tier"].isin(tiers)] if tiers else sig
        st.dataframe(view.sort_values("prob", ascending=False), use_container_width=True)

with t_eq:
    eq = _frame("equity")
    if eq.empty:
        _empty("equity curves")
    else:
        fig = px.line(eq, x="date", y="equity", color="strategy",
                      log_y=st.checkbox("Log scale", value=False))
        st.plotly_chart(fig, use_container_width=True)

with t_tr:
    tr = _frame("trades")
    if tr.empty:
        _empty("trades")
    else:
        strat = st.selectbox("Strategy", ["(all)"] + sorted(tr["strategy"].unique()))
        view = tr if strat == "(all)" else tr[tr["strategy"] == strat]
        st.dataframe(view.sort_values("entry_date", ascending=False), use_container_width=True)
        if "ret_net" in view:
            st.plotly_chart(px.histogram(view, x="ret_net", nbins=50, title="Net return per trade"),
                            use_container_width=True)

with t_model:
    imp, br = _frame("importance"), _frame("breakdown")
    if imp.empty and br.empty:
        _empty("model artifacts")
    if not imp.empty:
        top = imp.sort_values("importance", ascending=False).head(20)
        st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h").update_yaxes(autorange="reversed"),
                        use_container_width=True)
    if not br.empty:
        dim = st.selectbox("Dimension", sorted(br["dimension"].unique()))
        st.dataframe(br[br["dimension"] == dim], use_container_width=True)

with t_guide:
    g = m.get("guide") or {}
    if not g:
        _empty("rolling-window analysis")
    else:
        st.markdown(f"**Key causal test:** {g.get('key_causal_test')}  \n**Verdict:** {g.get('verdict')}  \n"
                    f"**Arm D:** {g.get('arm_d')}")
        arms = _frame("guide_arms.parquet")
        if not arms.empty:
            st.dataframe(arms, use_container_width=True)
        cards = [{"name": s["name"], "cagr": s.get("cagr"), "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
                  **{k: v for k, v in (s.get("report_card") or {}).items() if not isinstance(v, (dict, list))}}
                 for s in (m.get("strategies") or []) if s["name"].startswith("guide_")]
        if cards:
            st.subheader("Report card (guide Section 9.3), identical execution & risk framework")
            st.dataframe(pd.DataFrame(cards), use_container_width=True)
        stress = _frame("cost_stress.parquet")
        if not stress.empty:
            st.subheader("Cost stress test")
            st.dataframe(stress.pivot_table(index="strategy", columns="cost_mult", values="cagr"), use_container_width=True)

with t_runs:
    log = artifacts.load_run_log(50)
    if not log:
        _empty("pipeline runs")
    else:
        st.dataframe(pd.DataFrame(log).iloc[::-1], use_container_width=True)
