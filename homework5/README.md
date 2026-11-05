# Stock Market Analytics Zoomcamp — Homework 5 (Module 5, 2026 cohort)

Solutions for Homework 5 (2026 cohort) of the
[Stock Market Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp).
Spec: [`cohorts/2026/homework5.md`](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework5.md).

This module is **exploratory**: three research ideas in free text, no code, no
trained model, no backtest required. Each answer below follows the spec's
six-part structure (market & scope → thesis → edge → data → research approach →
validation & risks) and can be copied straight into the
[submission form](https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05).
Where I state something as *checked*, I verified it myself (SEC filings, gov.hk,
company releases); everything marked *to verify* is exactly that, and separating
the two is deliberate.

---

## Answers (summary)

| Question | Idea | One-line thesis |
| --- | --- | --- |
| **Q1** | Cross-exchange relative value: NYSE ADSs vs Hong Kong primary listings (BABA/9988.HK pilot) | Same shares, two time zones, fully fungible — trade only the fat-tail dislocations that survive conversion frictions; target round-trip ≤ 10 days, market-neutral. |
| **Q2** | Weekly demand proxies to anticipate US advertising-platform revenue (META, GOOGL, TTD, APP) | Ad budgets react to observable consumer demand weekly but are reported quarterly; free demand-side and campaign-side proxies, lagged 4–10 weeks, should predict *relative* revenue surprises of ad platforms. |
| **Q3** | Panic vs permanent damage: post-crash reversal in US small/mid caps, filtered by an SEC event taxonomy | After a ≥ 10% single-day fall, whether the bounce is tradeable depends on what was filed, not how the price moved — classify the 8-K/EDGAR trail before sizing anything. |

Machine-readable one-line versions are in
[`data/outputs/answers.json`](data/outputs/answers.json).

### How the three ideas differ

The spec asks for differences in economic mechanism, market, data source, and
horizon — not three tickers and three indicators. Cross-checking the ideas
against each other:

| Axis | Q1 Pairs | Q2 Advertising | Q3 Crash reversal |
| --- | --- | --- | --- |
| Market | NYSE/Nasdaq ↔ HKEX dual listings, mega caps | US ad-platform sector, mega caps | US small/mid-cap cross-section |
| Mechanism | Fungibility frictions between two listings of one firm | Demand → budget → booked revenue, lagged by weeks | Overreaction vs permanent impairment after a shock |
| Bet type | Market-neutral relative value | Sector-relative long/short | Conditional directional (long / skip) |
| Data | Both legs' prices + FX + depositary/borrow cost records | Free demand & campaign proxies (no price data at the core) | Prices + free SEC filings |
| Horizon | ≤ 10 trading days | 4–10 weeks | 1–20 days |
| Kill switch | Net expectancy ≤ 0 after full costs, half-life > 15 sessions | Lagged signals only echo last quarter's headline | Filter can't beat unconditional dip-buying |

---

## Q1 — Cross-exchange relative value: NYSE ADSs vs Hong Kong primary listings (BABA/9988.HK pilot)

### 1. Market and scope

US-listed ADSs of Chinese companies that are also primary/dual-primary listed in
Hong Kong, and the Hong Kong shares of the same firms. Pilot pair: **Alibaba —
BABA (NYSE) vs 9988.HK**, where I verified that one ADS = 8 ordinary shares and
that the two legs are fully fungible, converting in both directions in about
two business days via the depositary (Citibank).
Extension set, only after the pilot works: BIDU/9888.HK, JD/9618.HK (1 ADS =
2 Class A shares, per JD's own Q1-2026 release), NTES/9999.HK (1 ADS = 5
ordinary since the Oct-2020 25:1 → 5:1 ratio change — verified via NetEase's
announcement), plus ZTO/2057.HK, XPEV/9868.HK, LI/2015.HK, BEKE/2423.HK
(ratios to confirm from each depositary agreement).
Horizon: **days** (round-trip target ≤ 10 trading sessions). Structure:
market-neutral, dollar-neutral legs, no directional China beta.

### 2. Investment thesis

These are literally the same shares trading in two time zones, so law-of-one-price
holds up to frictions: the properly converted spread (HK price × HKD/USD,
divided by shares-per-ADS, against the ADR price) should be stationary around a
near-zero anchor bound by conversion arbitrage. Realized divergences therefore
come from non-fundamental frictions: sessions where only one leg trades (typhoon
closures and HK holidays in one direction, US holidays in the other, weekend
policy news landing in exactly one session), one-way flows from desks that hold
and hedge only one leg, HKEX index rebalances and derivatives expiries, borrow
squeezes on either leg, and regime scares — the 2021 US-delisting episode bent
the ADR discount by multiple percent for weeks before reverting. Each fat-tail
dislocation is a bet that **fungibility mechanics, not fundamentals, re-assert
within days to weeks**. This is a narrow, testable claim about a mechanism, not
a growth story about Chinese tech.

### 3. Potential edge

The same frictions that keep institutions from closing every gap keep the tails
open to a disciplined operator: conversion takes ~2 business days, so
convergence cannot be arbitraged instantaneously and spreads overshoot; the HK
leg is shortable only for designated securities at variable borrow fees (to
verify per name); and Hong Kong's round-trip **stamp duty is 0.2%** (0.1% per
side since 17 Nov 2023 — checked on gov.hk), which together with commissions and
levies means the normal-sized spread stays untradeable — realized profit
concentrates in rare, wide dislocations, so whoever can act there, cheaply, has
the field mostly to themselves. Most China-focused funds express views on only
one leg; few retail accounts can even short the HK line.

### 4. Data and sources

- **Checked:** daily prices for both legs free via yfinance (the course toolkit
  from Modules 1–4 already covers it); ADS ratios and fungibility for BABA, JD,
  NTES from SEC filings and company releases; HK stamp duty rates from gov.hk /
  KPMG; HK/US calendars incl. typhoon closures from HKEX notices; BABA's
  additional RMB counter **89988** as a third leg (Alibaba investor FAQ).
- **To verify in implementation:** the HKEX designated short-selling list and
  borrow fee per name; how yfinance handles HK-side corporate actions
  (dividend-adjusted vs raw closes and which is right for a spread); which
  session's close to pair with which on both-open days; depositary conversion
  eligibility windows and any daily capacity; whether the RMB counter trades at
  a persistent discount that itself forms a second spread.

### 5. Research approach

For the pilot pair, build the daily log-spread
`z_t = ln((8 · P_HK · USD_per_HKD) / P_ADR)`, scaled by a rolling 60-day mean and
σ. Enter at |z| ≥ 2 with estimated round-trip costs < half the historical
mean-reversion amplitude; time-stop at 10 sessions; and hard-exit/re-examine if
|z| > 3 persists 5+ sessions, which usually means a structural break rather than
a trade (e.g., delisting risk repriced the ADR alone). Size per leg by inverse
volatility, no FX hedge initially (the HKD peg band 7.75–7.85 makes the FX leg
noise, not signal). The rule informs all three decisions: enter, size, exit.

### 6. Validation and risks

Benchmarks: (a) the naive version — a fixed 2.0% band without z-scoring or a
cost model; (b) doing nothing (the pair should earn ≈ 0 net by construction).
Kill criteria: net expectancy after **all** costs ≤ 0; median convergence
half-life > 15 sessions (capital efficiency gone); or spread variance dominated
by structural regimes — i.e., the strategy would have been run over in 2021.
Risks: borrow availability and fees spike exactly at the dislocated moments;
execution slippage around stress sessions (the HK gap against the US close);
information that legitimately belongs to one leg (then "buy every gap" loses the
whole book); and honest small-sample risk — tradeable episodes over ~6 years
are low double digits, so I would rather under-size and keep the parameter set
tiny than tune on a handful of tails.

---

## Q2 — Weekly demand proxies to anticipate US advertising-platform revenue (META, GOOGL, TTD, APP)

### 1. Market and scope

The US digital-advertising ecosystem: META (social feed ads), GOOGL
(search & YouTube), TTD (programmatic open internet), APP (in-app performance
ads), with PINS/SNAP as higher-risk satellites. Two trade types: (i) intra-sector
relative long/short between the highest- and lowest-ranked signal each week;
(ii) small directional tilts into earnings. Horizon: **4–10 weeks** — the gap
between when ad budgets react to demand and when platforms report the result —
rebalanced weekly.

### 2. Investment thesis

Advertising is the fastest-moving demand-sensitive line in these income
statements: budgets expand or pause within days–weeks of merchants observing
their own demand, platforms book spend on delivery during the quarter — yet
investors see the outcome only ~4 times a year. The bet is *not* that the ad
market is growing (that is public and priced). It is narrower: **(1)** budget
decisions follow observed consumer demand, so consumer-behavior proxies that
tick weekly — search/browsing interest, app-install and in-app activity, counts
of newly launched advertiser campaigns — should lead reported platform revenue
growth; **(2)** mix shifts between platforms (retail media vs open-internet
performance vs search) move at the sell side's forecast speed, so relative
4–10-week *signal deltas* should predict relative revenue surprises vs
consensus. If true, part of next quarter's beat/miss becomes tradeable from
data that refresh weekly.

### 3. Potential edge

The information is public but fragmented: no free, single view aggregates
cross-platform ad-demand signals weekly, and the expensive panels
(Sensor Tower / Pathmatics / eMarketer) are out of most individuals' budgets, so
the crowd largely trades price/volume or reacts to the print itself. My proposed
blend sits two steps *earlier* in the causal chain than price data — closer to
the budget decision — while still using only free or cheap sources. Because the
trade is relative-value within one sector, the big shared macro-ad-cycle factor
largely cancels, and what remains is exactly the part the aggregators ignore:
*which* platform the next demand dollar lands on.

### 4. Data and sources

- **Checked:** Meta Ad Library and Google Ads Transparency Center are free and
  searchable (checked; API stability and historical depth *to verify*);
  Google Trends provides free daily, geo/category search interest — the
  demand-side lens; the two app stores' front pages are free front-page rank
  signals (download *estimates* are paid — visibility, not download counts, is
  the free layer); earnings dates and consensus direction free via IR pages /
  yfinance (as used in Module 1).
- **To verify:** whether transparency-library aggregates are comparable across
  years (political-ad rules load up around US election cycles — needs election
  dummies or seasonal exclusion); a stable way to get weekly *new advertiser*
  counts per platform; how to debias the search proxies from platform-specific
  promo noise; optional paid add-ons only if the free layer fails.

### 5. Research approach

Weekly panel. Target: quarterly revenue actual-minus-consensus, refreshed at
each report. Features per platform-week: (i) Google-Trends consumer-interest
index for that platform's spend-heavy categories/geos; (ii) weekly count of new
campaigns/advertisers in its transparency library (down-weighted in political
seasons); (iii) app-store rank/download-curve momentum for its owned apps;
(iv) controls: the platform's last reported growth rate (to prove the signal
isn't just echoing the previous quarter). Model with a purged time-series CV
(leave-one-quarter-out); output is a ranked expected relative surprise, and the
trade is top-vs-bottom of the rank, exited at the earlier of the two names'
earnings prints. The rule informs which platform to overweight vs underweight
over the next 1–2 months — nothing longer.

### 6. Validation and risks

Benchmarks: (a) equal-weight sector basket; (b) always-long last year's share
gainer; (c) fade-the-loser after its last print. Kill criteria: the lagged
signal coefficients fail |t| ≥ 2 on a 3-year panel of ~28 platform-quarters
(then the proxies merely echo last quarter's headline), or the signal moves
*consensus* but not *actuals* (the market already eats it). Risks: reverse
causality (platforms' own campaigns distort the proxies); free proxies ≈ partial
spend, not true spend; political-ad contamination; crowding near prints; and
the honest possibility that platform *margins*, not revenue, drive the stocks —
in which case I extend to cost-side signals or stop, rather than trade a signal
that predicts the wrong thing.

---

## Q3 — Panic vs permanent damage: post-crash reversal in US small/mid caps, filtered by an SEC event taxonomy

### 1. Market and scope

US common stocks, market cap ≈ $500M–$15B (liquid enough to trade both ways,
small enough to overshoot), excluding sub-$500M micro-caps. Event universe: any
single-session fall ≥ 10% on ≥ 3× its 30-day average volume, with an
identifiable catalyst filed around the drop on SEC EDGAR (8-K Item 2.02 results
with lowered outlook, Item 1.01 material agreement/financing, Item 8.01 other
events, 424B/S-3 offerings, clinical-readout releases). Horizon: **1–20 days**;
position ≤ 2.5% of book per event.

### 2. Investment thesis

After extreme single-day drops, prices overshoot whenever the shock changes
*near-term* numbers rather than the business: stop-loss cascades and
short-horizon risk-control signals transact mechanically at the low, while
fundamentals move far more slowly than a one-day quote. The exploitable
asymmetry is that crashes are **not one base rate** — they are several: a
one-quarter guidance cut on an intact balance sheet, a contract loss, a smaller
clinical miss, versus dilution at a discount, going-concern language,
restatements. I am not claiming crashes bounce back; I am claiming
**a repeatable, rule-based separation of "temporary overreaction" from
"permanent impairment" exists, is checkable in the free filing trail, and is
where the return is concentrated** — buy the first class, skip the second,
and let the data tell me which is which.

### 3. Potential edge

The bounce/no-bounce answer lives in unglamorous, free, per-incident filings —
*what was filed, not how the tape moved*. Sell-side notes in the first 48 hours
mostly re-transmit the 8-K; social feeds are noise. Two concrete edges: (i) an
explicit event taxonomy whose branches have historically different bounce base
rates; (ii) **exclusion discipline** — offerings-at-a-discount, going-concern,
covenant/auditor and restatement items predict *continued* downside, and
dropping them plausibly changes the expectancy's sign (to be proven, not
assumed). This intentionally *inverts* my Module 1 capstone idea — which bets on
post-earnings **drift** in mega caps — because the mechanism is the opposite
(reversal after extreme shocks in smaller names), not the same signal with a
different ticker.

### 4. Data and sources

- **Checked:** daily OHLCV/volume via yfinance (already used in Modules 1–2);
  SEC EDGAR full-text search, filing index and 8-K item codes, S-3/424B
  postings, 10-K/10-Q excerpts for funding/covenant/going-concern language —
  all free.
- **To verify:** clean historical mapping of *which* filing triggered each
  crash (full-text search in a ±24h window, time-aligned to session boundaries);
  tagging "guidance cut" from press-release exhibits ("now expects…" phrasing) —
  feasible, but I plan to hand-label a few hundred events first to build a
  trustworthy tagged set; FINRA biweekly short interest (free, delayed) as a
  crowding feature; next-day-open execution data only if the first
  close-to-close version survives.

### 5. Research approach

Cheapest falsifier first. **Stage 1 — event study:** industry- and size-matched
abnormal returns D+1…D+20 after each event, split by taxonomy; unstable base
rates across 2019–2026 kill a branch. Output: which event types clear the
post-cost hurdle. **Stage 2 — rules, not ML-first:** a small transparent
classifier (depth-limited tree, Module-3 style) on structural features by hand —
event type, offering present?, short interest, cash runway in months for
clinical names, distance from the 52-week high — defines the allowed subset;
buy qualifying events with 1%–2.5% sizing, a 10-day time stop, and a mandatory
exit if any *second* systemic item lands mid-trade (restatement, auditor
resignation, delisting notice). A small hedged short book on the "permanent"
class is an optional extension, not the core. The decision it informs per
incident: buy / skip / hedge.

### 6. Validation and risks

Benchmarks: (a) act on **every** qualifying event with no filter — the taxonomy
must beat the unconditional dip-buying base rate, net of costs; (b) a
matched passive basket for the same window. Kill criteria: no event type's net
expectancy > 0 after borrow and fees; expectancy positive in only one regime
(e.g., 2020–2021) and negative elsewhere; or results surviving only because
delisted names are missing (survivorship). Risks: further −30% tails after
entry (knife-catching) — mitigations are sizing, time stops and the
second-event veto; borrow blowups; look-ahead bias in event tagging; and the
non-linearity that in liquidity crises (2008, Mar-2020) **everything kept
falling** — so the book caps total exposure to qualifying events and pauses
after an index-level stress trigger. The durable part of the edge, if any, is
the *exclusion rule* — what not to buy.

---

## Notes and judgment calls

- **Why these three.** Each lives on a different mechanism (fungibility
  arbitrage / demand-to-revenue lag / overreaction classification), a different
  bet type (market-neutral, sector-relative, conditional directional), a
  different data core (two-market prices + cost records / third-party demand
  proxies / filings text), and a different horizon (days / weeks–quarter /
  1–20 days).
- **A fourth idea was considered and set aside.** A bioinformatics angle —
  NIH RePORTER + USAspending procurement flows mapped to life-science tools
  names (TMO, ILMN, CRL, IQV) — shares its signal-construction pattern with Q2
  ("external spend data → company revenue"), so submitting both risks reading
  as one idea in two costumes. It is a reasonable swap if one of the three
  fails early, since the data there is genuinely free and queryable.
- **Consistency with earlier modules.** Q3's taxonomic rules follow Module 3's
  "simple model first" lesson (hand rules → tree → forest), Q1 reuses Module 4's
  cost-aware, out-of-sample discipline (model the fees before believing any
  spread), and the labeled-set caution in Q2 comes straight from the Module 1
  event-study experience.
- None of the three claims to be profitable. Per the spec's framing, the goal is
  three worthwhile investigations with a clear first test and a stated reason
  each could fail — the "Validation and risks" section of each idea serves as
  the pre-submission checklist.

## Repository layout

```
homework5/
  README.md           This file — the three submitted ideas (Q1–Q3)
  data/
    outputs/
      answers.json    Machine-readable summary (titles + one-line theses)
```

No code, dataset, or pipeline is included: this homework explicitly requires
none, and every claim that would need data has its source and verification
status marked inline.

### Submission

Form: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05
(leaderboard: https://courses.datatalks.club/sma-zoomcamp-2026/leaderboard).
