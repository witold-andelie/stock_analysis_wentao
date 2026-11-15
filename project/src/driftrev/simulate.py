"""Trading simulations.

Two levels, as in Modules 4-5:

* Vector simulations (`vector_sim`) - one fixed $100 bet per positive prediction, fees per
  trade, approximate capital = $100 x holding days x Q75(daily bets) (course `sim1_`).
  `sim2_` adds a stop-loss / take-profit using the intraperiod extremes (MAE / MFE).
* Exact simulation (`exact_sim`) - a day-by-day loop with a cash account, reinvestment of
  gains, a maximum number of concurrent positions, probability-weighted sizing, stop-loss /
  take-profit / time exits on real daily OHLC paths, a regime pause and an optional short leg.

Benchmarks: SPY and IWM buy-and-hold, and the HW1 momentum rule (top quintile of the
blended 20/50/200-day momentum score, equal weight, rebalanced every 10 sessions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
S = config.SETTINGS
RT = S.costs.round_trip
ONE_WAY = S.costs.bps_one_way / 1e4


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def perf_metrics(equity: pd.Series, name: str) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {"name": name}
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = equity / equity.cummax() - 1
    downside = rets[rets < 0].std() * np.sqrt(252)
    roll = equity.pct_change(252).dropna()
    return {
        "name": name, "start": equity.index[0].date().isoformat(), "end": equity.index[-1].date().isoformat(),
        "final_equity": float(equity.iloc[-1]), "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": float(cagr), "volatility": float(rets.std() * np.sqrt(252)),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan,
        "sortino": float(rets.mean() * 252 / downside) if downside > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
        "rolling_1y_min": float(roll.min()) if len(roll) else np.nan,
        "rolling_1y_median": float(roll.median()) if len(roll) else np.nan,
    }


# --------------------------------------------------------------------------------------
# Vector simulations (course sim1_ / sim2_)
# --------------------------------------------------------------------------------------
def vector_sim(df: pd.DataFrame, pred_cols: list[str], bet: float = 100.0) -> pd.DataFrame:
    """df: TEST rows with growth_future_10d, mae_10d, mfe_10d. Returns one row per prediction."""
    hz = S.target.horizon
    st = S.strategy
    rows = []
    years = (df["date"].max() - df["date"].min()).days / 365.25
    for c in pred_cols:
        sel = df[(df[c] == 1) & df[f"growth_future_{hz}d"].notna()]
        if sel.empty:
            continue
        g = sel[f"growth_future_{hz}d"]
        # sim1: hold to the horizon
        gross1 = (bet * (g - 1)).sum()
        fees = bet * RT * len(sel)
        # sim2: stop-loss first (conservative), then take-profit, else horizon exit
        r2 = np.where(sel["mae_10d"] <= -st.stop_loss, -st.stop_loss,
                      np.where(sel["mfe_10d"] >= st.take_profit, st.take_profit, g - 1))
        gross2 = (bet * r2).sum()
        per_day = sel.groupby("date").size()
        q75 = float(per_day.quantile(0.75))
        capital = bet * hz * q75            # money needed to keep Q75 daily bets running for H days
        for tag, gross in (("sim1", gross1), ("sim2_sl_tp", gross2)):
            net = gross - fees
            cagr = ((capital + net) / capital) ** (1 / years) - 1 if years > 0 and capital > 0 else np.nan
            rows.append({"prediction": c, "sim": tag, "n_trades": len(sel), "avg_trades_per_day": float(per_day.mean()),
                         "q75_trades_per_day": q75, "gross": float(gross), "fees": float(fees), "net": float(net),
                         "capital": capital, "cagr": float(cagr), "hit_rate": float((g > 1).mean()),
                         "avg_net_return": float(net / len(sel) / bet)})
    return pd.DataFrame(rows).sort_values("cagr", ascending=False)


# --------------------------------------------------------------------------------------
# Exact simulation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimConfig:
    """One strategy. Two exit/sizing frameworks share the same loop:

    * fixed %:   stop_loss / take_profit as fractions of entry, equal (or prob-weighted) size.
    * guide R:   atr_stop_mult -> per-trade stop = k x ATR; take profit at r_target x R;
                 risk-based size = risk_per_trade x equity / stop distance, capped at
                 max_notional_pct of equity; total planned open risk <= max_open_risk;
                 skip gap-ups > gap_tolerance; no new entries the day after
                 >= circuit_breaker_stops stop-outs (guide Sections 6-7).
    """
    name: str
    prob_col: str = "proba_ensemble"
    min_prob: float = 0.55
    filter_col: str | None = None          # rule-based arm: trade rows where this column == 1
    max_positions: int = 5
    hold_days: int = 10
    stop_loss: float | None = None
    take_profit: float | None = None
    prob_weighted: bool = False
    regime_pause: bool = False
    families: tuple[str, ...] | None = None
    short_below: float | None = None       # open a short when prob <= this (long-short book)
    atr_stop_mult: float | None = None
    r_target: float | None = None
    risk_per_trade: float | None = None
    max_notional_pct: float = 1.0
    max_open_risk: float | None = None
    gap_tolerance: float | None = None
    circuit_breaker_stops: int | None = None
    cost_mult: float = 1.0                 # cost stress test: 1x / 2x / 3x
    initial_capital: float = S.strategy.initial_capital


def _paths(panel: pd.DataFrame, tickers: set[str], start, end) -> dict[str, pd.DataFrame]:
    p = panel[panel["ticker"].isin(tickers) & (panel["date"] >= start) & (panel["date"] <= end)]
    return {t: g.set_index("date")[["open", "high", "low", "close"]].astype(float) for t, g in p.groupby("ticker")}


def exact_sim(events: pd.DataFrame, panel: pd.DataFrame, cfg: SimConfig,
              start: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """events: rows with ticker, date, next_date, close, event_family, regime_state, prob_col
    (+ atr_pct for the R framework). Signals known at the close of `date` are executed at
    the open of `next_date`. Returns (daily equity, trades). Trades carry fill info."""
    one_way = ONE_WAY * cfg.cost_mult
    ev = events.dropna(subset=["next_date", cfg.prob_col]).copy()
    if start:
        ev = ev[ev["date"] >= pd.Timestamp(start)]
    if cfg.families:
        ev = ev[ev["event_family"].isin(cfg.families)]
    if cfg.filter_col:
        ev = ev[ev[cfg.filter_col] == 1]
    if cfg.regime_pause:
        ev = ev[ev["regime_state"] != "stressed"]
    longs = ev[ev[cfg.prob_col] >= cfg.min_prob].assign(side=1)
    shorts = ev[ev[cfg.prob_col] <= cfg.short_below].assign(side=-1) if cfg.short_below is not None else ev.iloc[0:0]
    cand = pd.concat([longs, shorts])
    cand["conviction"] = np.where(cand["side"] == 1, cand[cfg.prob_col], 1 - cand[cfg.prob_col])
    cand = cand.sort_values(["next_date", "conviction"], ascending=[True, False])
    by_day = {d: g for d, g in cand.groupby("next_date")}

    dates = np.sort(panel.loc[(panel["date"] >= (ev["date"].min() if len(ev) else pd.Timestamp(start or "2100"))), "date"].unique())
    if len(dates) == 0:
        return pd.Series(dtype=float), pd.DataFrame()
    paths = _paths(panel, set(cand["ticker"]), dates[0], dates[-1])

    cash = cfg.initial_capital
    positions: list[dict] = []
    trades, curve = [], []
    last_px: dict[str, float] = {}
    skipped = {"capacity": 0, "gap": 0, "risk_budget": 0, "circuit_breaker": 0, "no_data": 0}
    stops_yesterday = 0
    for day in pd.to_datetime(dates):
        stops_today = 0
        # 1) new entries at today's open (signals from yesterday's close)
        todays = by_day.get(day)
        if todays is not None:
            equity_prev = curve[-1][1] if curve else cash
            held = {p["ticker"] for p in positions}
            breaker = cfg.circuit_breaker_stops is not None and stops_yesterday >= cfg.circuit_breaker_stops
            for r in todays.itertuples(index=False):
                if breaker:
                    skipped["circuit_breaker"] += 1
                    continue
                if len(positions) >= cfg.max_positions:
                    skipped["capacity"] += 1
                    continue
                if r.ticker in held or r.ticker not in paths or day not in paths[r.ticker].index:
                    skipped["no_data"] += 1
                    continue
                bar = paths[r.ticker].loc[day]
                o = bar["open"]
                if not np.isfinite(o) or o <= 0:
                    skipped["no_data"] += 1
                    continue
                if cfg.gap_tolerance is not None and r.side == 1 and o > r.close * (1 + cfg.gap_tolerance):
                    skipped["gap"] += 1           # guide: skip large gap-ups beyond entry tolerance
                    continue
                if cfg.atr_stop_mult is not None:
                    stop_dist = float(np.clip(cfg.atr_stop_mult * (getattr(r, "atr_pct", 0.04) or 0.04), 0.02, 0.25))
                else:
                    stop_dist = cfg.stop_loss
                if cfg.risk_per_trade is not None and stop_dist:
                    size = cfg.risk_per_trade * equity_prev / stop_dist
                    size = min(size, cfg.max_notional_pct * equity_prev)
                    if cfg.max_open_risk is not None:
                        open_risk = sum(p["risk"] for p in positions)
                        room = cfg.max_open_risk * equity_prev - open_risk
                        if room <= 0:
                            skipped["risk_budget"] += 1
                            continue
                        size = min(size, room / stop_dist)
                else:
                    size = min(equity_prev / cfg.max_positions, cfg.max_notional_pct * equity_prev)
                    if cfg.prob_weighted:   # up to 1.5x for the most confident signals
                        size *= float(np.clip(0.5 + (r.conviction - cfg.min_prob) / max(1e-6, 1 - cfg.min_prob) * 2, 0.5, 1.5))
                if r.side == 1:
                    size = min(size, cash / (1 + one_way))
                if size < 50:
                    skipped["capacity"] += 1
                    continue
                shares = r.side * size / o
                cash -= shares * o + size * one_way
                tp = None
                if cfg.r_target is not None and stop_dist:
                    tp = cfg.r_target * stop_dist
                elif cfg.take_profit is not None:
                    tp = cfg.take_profit
                positions.append({"ticker": r.ticker, "side": r.side, "shares": shares, "entry": o, "entry_date": day,
                                  "days": 0, "prob": float(getattr(r, cfg.prob_col)), "event_family": r.event_family,
                                  "event_date": r.date, "cost_in": size * one_way, "stop": stop_dist, "tp": tp,
                                  "risk": size * (stop_dist or 0)})
                held.add(r.ticker)
        # 2) exits: stop / take-profit intraday, time exit at the close
        still = []
        for p in positions:
            path = paths.get(p["ticker"])
            if path is None or day not in path.index:
                still.append(p)
                continue
            bar = path.loc[day]
            p["days"] += 1
            last_px[p["ticker"]] = bar["close"]
            exit_px, reason = None, None
            e, sd = p["entry"], p["side"]
            if p["stop"]:
                stop = e * (1 - sd * p["stop"])
                if sd == 1 and bar["low"] <= stop:      # gap through the stop fills at the open
                    exit_px, reason = (min(bar["open"], stop) if p["days"] > 1 else stop), "stop_loss"
                elif sd == -1 and bar["high"] >= stop:
                    exit_px, reason = (max(bar["open"], stop) if p["days"] > 1 else stop), "stop_loss"
            if exit_px is None and p["tp"]:
                tp = e * (1 + sd * p["tp"])
                if (sd == 1 and bar["high"] >= tp) or (sd == -1 and bar["low"] <= tp):
                    exit_px, reason = (max(bar["open"], tp) if sd == 1 and p["days"] > 1 else tp), "take_profit"
            if exit_px is None and p["days"] >= cfg.hold_days:
                exit_px, reason = bar["close"], "time"
            if exit_px is None:
                still.append(p)
                continue
            stops_today += reason == "stop_loss"
            notional = abs(p["shares"]) * exit_px
            cash += p["shares"] * exit_px - notional * one_way
            pnl = p["shares"] * (exit_px - e) - p["cost_in"] - notional * one_way
            entry_notional = abs(p["shares"]) * e
            trades.append({"strategy": cfg.name, "ticker": p["ticker"], "side": "long" if sd == 1 else "short",
                           "event_family": p["event_family"], "event_date": p["event_date"], "prob": p["prob"],
                           "entry_date": p["entry_date"], "entry_price": e, "exit_date": day, "exit_price": exit_px,
                           "ret_net": pnl / entry_notional, "pnl": pnl, "exit_reason": reason, "days_held": p["days"],
                           "notional": entry_notional,
                           "r_multiple": (pnl / entry_notional) / p["stop"] if p["stop"] else np.nan})
        positions = still
        stops_yesterday = stops_today
        # 3) mark to market
        mtm = sum(p["shares"] * last_px.get(p["ticker"], p["entry"]) for p in positions)
        curve.append((day, cash + mtm))
    equity = pd.Series([v for _, v in curve], index=[d for d, _ in curve], name=cfg.name)
    tr = pd.DataFrame(trades)
    tr.attrs["signals"] = int(len(cand))
    tr.attrs["skipped"] = skipped
    return equity, tr


def guide_risk(cfg: SimConfig) -> SimConfig:
    """Apply the strategy guide's execution & risk template (Sections 6-7)."""
    return replace(cfg, atr_stop_mult=2.0, r_target=3.0, hold_days=8, risk_per_trade=0.005, max_notional_pct=0.10,
                   max_open_risk=0.02, gap_tolerance=0.03, circuit_breaker_stops=3, max_positions=10,
                   stop_loss=None, take_profit=None)


def standard_strategies(prob_col: str, thr: float) -> list[SimConfig]:
    st = S.strategy
    base = SimConfig(name="", prob_col=prob_col, min_prob=thr, max_positions=st.max_positions, hold_days=st.hold_days)
    rule = replace(base, prob_col="rw_prior_strength", min_prob=-1.0)   # arms rank by prior strength
    breakout = ("momentum_breakout",)
    return [
        replace(base, name="all_events_unconditional", min_prob=-1.0),
        replace(base, name="model_top5"),
        replace(base, name="model_top3_concentrated", max_positions=3),
        replace(base, name="model_top5_risk_managed", stop_loss=st.stop_loss, take_profit=st.take_profit,
                prob_weighted=True, regime_pause=True),
        replace(base, name="model_pead_only", families=("earnings_up", "earnings_down")),
        replace(base, name="model_crash_only", families=("earnings_crash", "shock_crash")),
        replace(base, name="model_breakout_only", families=breakout),
        replace(base, name="model_long_short", stop_loss=st.stop_loss, take_profit=st.take_profit,
                regime_pause=True, short_below=0.30, max_positions=6),
        # strategy guide: four-arm test, identical execution & risk framework for every arm
        guide_risk(replace(rule, name="guide_armA_momentum_breakout", filter_col="arm_a", families=breakout)),
        guide_risk(replace(rule, name="guide_armB_plus_consolidation", filter_col="arm_b", families=breakout)),
        guide_risk(replace(rule, name="guide_armC_plus_window_release", filter_col="arm_c", families=breakout)),
        guide_risk(replace(base, name="guide_model_breakout_R", families=breakout)),
        guide_risk(replace(base, name="guide_model_all_events_R")),
    ]


# --------------------------------------------------------------------------------------
# Report card (strategy guide Section 9.3)
# --------------------------------------------------------------------------------------
def report_card(trades: pd.DataFrame, equity: pd.Series, spy: pd.Series | None = None,
                n_boot: int = 2000, seed: int = 42) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    rng = np.random.default_rng(seed)
    r = trades["ret_net"].to_numpy()
    boots = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    top5 = pnl.sort_values(ascending=False).head(5).sum()
    signals = trades.attrs.get("signals", np.nan)
    out = {
        "n_signals": signals, "n_trades": int(len(trades)),
        "fill_rate": float(len(trades) / signals) if signals else np.nan,
        "skipped": trades.attrs.get("skipped", {}),
        "avg_trade_ret": float(r.mean()), "median_trade_ret": float(np.median(r)),
        "mean_ret_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "avg_r_multiple": float(trades["r_multiple"].mean()) if "r_multiple" in trades else np.nan,
        "win_rate": float((pnl > 0).mean()), "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
        "top5_profit_share": float(top5 / pnl.sum()) if pnl.sum() > 0 else np.nan,
        "turnover_per_year": float(trades["notional"].sum() / equity.mean() /
                                   max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)) if "notional" in trades else np.nan,
    }
    if spy is not None and len(equity) > 30:
        er = equity.pct_change()
        sr = spy.reindex(equity.index).pct_change()
        ok = er.notna() & sr.notna()
        out["beta_vs_spy"] = float(np.cov(er[ok], sr[ok])[0, 1] / np.var(sr[ok])) if ok.sum() > 30 else np.nan
    return out


# --------------------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------------------
def buy_and_hold(market: pd.DataFrame, ticker: str, start, end, capital: float) -> pd.Series:
    m = market[(market["ticker"] == ticker) & (market["date"] >= start) & (market["date"] <= end)].sort_values("date")
    tr = (m["close"] + m["dividends"].fillna(0)) / m["close"].shift(1)
    tr.iloc[0] = 1.0
    return pd.Series(capital * tr.cumprod().to_numpy(), index=pd.to_datetime(m["date"]).to_numpy(), name=f"{ticker}_buy_hold")


def momentum_baseline(panel: pd.DataFrame, start, end, capital: float, rebalance: int = 10,
                      quantile: float = 0.8) -> pd.Series:
    """HW1 rule: rank liquid stocks by the blended momentum score (price vs 20/50/200-day MAs
    plus 12-1 momentum), hold the top quintile equally weighted, rebalance every 10 sessions,
    pay one-way cost on turnover."""
    p = panel[(panel["date"] >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (panel["date"] <= end)].copy()
    p["score"] = (p[["close_sma20", "close_sma50", "close_sma200"]].sub(1).mean(axis=1) + p["mom_12_1"]) / 2
    rets = p.pivot_table(index="date", columns="ticker", values="ret_1d").sort_index()
    score = p[p["is_liquid"]].pivot_table(index="date", columns="ticker", values="score").reindex(rets.index)
    dates = rets.index[rets.index >= pd.Timestamp(start)]
    equity, weights, values = capital, pd.Series(dtype=float), []
    for i, d in enumerate(dates):
        if not weights.empty:
            r = rets.loc[d, weights.index].fillna(0)
            equity *= 1 + float((weights * r).sum())
            weights = weights * (1 + r)
            weights = weights / weights.sum()
        if i % rebalance == 0:
            s = score.loc[d].dropna()
            if len(s) > 50:
                top = s[s >= s.quantile(quantile)].index
                new = pd.Series(1 / len(top), index=top)
                turnover = new.subtract(weights, fill_value=0).abs().sum() if not weights.empty else 1.0
                equity *= 1 - turnover * ONE_WAY
                weights = new
        values.append(equity)
    return pd.Series(values, index=dates, name="hw1_momentum_top_quintile")
