# Mean-Reversion Prototype V1 — Phase 1 Research Report

**Status: RESEARCH_CANDIDATE_NOT_OFFICIAL.** Read-only research + an
isolated, non-live prototype (`src/research/mean_reversion_prototype/`).
Never touched the frozen daily engine, `CONTROLLED_TICKERS`,
`LIVE_CONTROLLED_TICKERS`, or any live path. No live integration.

## Why this exists

The frozen daily EMA20/EMA50/RSI14 system trades sparsely by design
(~4 trades/year/ticker on equities, ~10/year/ticker on crypto — both
measured for the first time in a separate same-day research task).
That doesn't match a goal of seeing a more actively-trading paper bot.
This is Phase 1 of a separate, crypto-only, hourly-bar research track
exploring whether a different (mean-reversion) signal family can trade
more often without collapsing quality — proof-of-concept only.

## Data source

Alpaca's own hourly historical-bars endpoint
(`src/data/alpaca_market_data.get_crypto_bars`), not yfinance — the
same day's separate crypto-universe research found yfinance returning
wrong/dead-instrument data for 6 of 33 tickers. Alpaca's data quality
here was clean: 0 invalid-OHLC rows across all 5 tickers, and <0.2%
missing bars for 4 of 5. SOL/USD is the one real exception: 20.5% of
its expected hourly bars are missing over its full history — a
genuine gap in Alpaca's own crypto data coverage for that pair, not a
fetch bug, and SOL's results should be read with that caveat.

## The three signal components

See `src/research/mean_reversion_prototype/signals.py` for the exact
implementation and inline parameter rationale (summarized here):

1. **Core: RSI(2) + SMA(200) trend filter.** RSI period 2 (Connors'
   defining choice — deliberately noisy/reactive). Entry <10 (Connors'
   own daily-bar threshold, kept on hourly bars — fires far more often
   here since hourly bars are noisier than daily ones; not tightened,
   to avoid a second uncontrolled variable). Exit at RSI≥70, or 24-bar
   (~1 day) time-out backstop. SMA(200) on **hourly** bars is ~8.3 days
   of trend context, not the ~10-month window Connors' daily SMA(200)
   represents — a genuine scale change, kept as specified rather than
   rescaled, and flagged here rather than assumed equivalent.
2. **Parallel OR signal: VWAP deviation.** A 24-bar rolling VWAP (built
   from Alpaca's own real intra-bar VWAP field, volume-weighted, not a
   cruder typical-price proxy) and its 24-bar rolling standard
   deviation. Entry when price ≤ −2.0 std devs below that rolling VWAP
   — the standard Bollinger-style threshold, deliberately left untuned.
   Independent of RSI(2); either can trigger entry (both still gated by
   the same SMA(200) trend filter — a design choice: the filter's
   stated purpose, refusing dip-buys in a downtrend, is a general
   risk rule that should apply no matter which trigger fired).
3. **Volume quality weighting (sizing, never a gate).** 20-bar volume
   moving average. At or above it → full size; below it → half size,
   never rejected outright.

## Isolated backtest

Not `portfolio_backtest_engine.py` (frozen, built for the daily
system's own event ordering) — a small standalone loop
(`backtest.py`): long-only, one position at a time per ticker,
next-bar-Open entry (no lookahead), exit at whichever of {RSI≥70,
price back at/above rolling VWAP, 24-bar time-out, 3% stop-loss} comes
first. Fixed $1,000 base notional, 0.10% round-trip cost assumption —
both simple and deliberately untuned. Full results:
`data/research/mean_reversion_prototype_v1_20260814_110449.json`.

## Component comparison (the core question)

| Config | Total trades (5 tickers) | Mean PF | Mean win-rate |
|---|---|---|---|
| RSI(2)+SMA200 only | 4,442 | 0.879 | 59.8% |
| VWAP deviation only | 1,293 | 0.771 | 57.1% |
| Combined (OR) | 4,761 | 0.874 | 59.8% |
| Combined, no volume weight | 4,761 | 0.881 | 59.8% |

**VWAP is mostly redundant with RSI, not a genuinely new opportunity
source.** Of VWAP's own signals (when run alone), a mean of **~64%**
across the 5 tickers coincide with a bar RSI(2) would already have
flagged. Adding VWAP as an OR-condition to RSI only lifts total trade
count by **~5–10% per ticker** (mean ~7.4%) — a real but modest
frequency gain, not a second independent edge. It doesn't move PF in a
consistent direction either (helped BTC/ETH slightly, hurt SOL/DOGE
slightly, flat on XRP) — it adds a handful of extra, similarly-mediocre
trades, not better ones.

**Volume weighting's effect on PF is small and inconsistent** (0.874 vs
0.881 aggregate — weighting was very slightly *worse* for PF in 4 of 5
tickers). Its real effect is on **net dollar outcome**, not the PF
ratio: total net P&L was less negative with weighting on for 4 of 5
tickers (e.g. BTC: −$68.09 weighted vs −$84.16 unweighted) because it
shrinks losing low-conviction trades' size, not because it improves
signal quality. Framed accurately: it's a risk-sizing tool, not a
quality filter, and it does that job modestly — it doesn't rescue an
unprofitable edge.

## Frequency — the central motivating question

| Ticker | Trades/day (Combined) | Trades/week |
|---|---|---|
| BTC/USD | 0.56 | 3.90 |
| ETH/USD | 0.55 | 3.86 |
| SOL/USD | 0.47 | 3.32 |
| DOGE/USD | 0.50 | 3.52 |
| XRP/USD | 0.51 | 3.54 |

**Per ticker, this does NOT hit a 2–3-trades/day target** — it's
roughly 1 trade every two days per ticker, honestly far short.
**Summed across all 5 tickers as one basket, it's ~2.59 trades/day** —
right in the stated 2–3/day range. Whether that counts as "hitting the
goal" depends entirely on whether the goal was ever meant per-ticker or
for the whole crypto sleeve — this report states both rather than
picking the flattering one. Versus the daily system's own ~0.027
trades/day/ticker (≈10/year), this is a real ~18× frequency increase
per ticker, whichever way it's counted.

## Honest quality verdict

**The frequency gain comes at a severe, not modest, quality cost.**
Every configuration on every ticker has profit factor **at or below
~1.0** — barely breakeven to net losing, despite respectable-looking
58–61% win rates (small frequent wins, evidently offset by larger or
more frequent losses). Before the 0.10% round-trip cost assumption,
gross PF hovers right at ~1.0 too (0.97–1.18, mean ~1.07) — so this
isn't primarily a transaction-cost problem masking a real edge; the
raw signal is close to a coin-flip in dollar terms even before costs.
Stop-losses fired on **~14.9%** of all combined trades — a real,
non-trivial share of trades resolving as losses via the stop rather
than a clean reversion. This is a stark contrast with the daily
system's own real crypto PF (BTC 2.468, ETH 2.359, measured the same
day) — the daily system is 2.4–2.6× more profitable per trade despite
trading ~18× less often.

## Phase 2 addendum — disciplined in-sample/out-of-sample parameter search

Full report: see the chat response for this phase, summarized here for
consistency with this doc's structure. Grid: 432 combinations (RSI
entry {5,10,15,20} x RSI exit {60,70,80} x timeout {12,24,48} bars x
VWAP deviation {1.5,2.0,2.5,3.0} std x SMA period {50,100,200}) x 5
tickers, split chronologically at **2024-12-07** (one shared calendar
date across all tickers, ~70% of BTC/USD's own full range) into
in-sample (search) and out-of-sample (final check, touched once).
Full grid: `data/research/mean_reversion_parameter_search_v1_20260814_123841.json`.

**Best in-sample PF across the entire grid: 1.102** — even the single
best-fitting combination out of 432 barely clears breakeven
in-sample, before any out-of-sample check. Top-8 in-sample combos all
share `rsi_entry=5` and `trend_sma_period=50` — a consistent region,
not an isolated spike (marginal mean PF is monotonic across both
dimensions: rsi_entry 5→20 gives 0.961→0.891, sma 50→200 gives
0.929→0.888 — smooth gradients, not noise). Top-5 in-sample combos'
mean out-of-sample PF: **~1.036** — real but modest, degrading from
in-sample as expected, and nowhere near the 1.3+ target.

**Frequency-quality tradeoff, re-checked fairly on the identical OOS
window:** V1's untuned defaults do ~2.46 trades/day (basket) at OOS
PF 0.724; the best-tuned combo reaches OOS PF ~1.07 but frequency
collapses to ~0.80 trades/day (basket) — a ~3.1x frequency loss,
taking it well out of the 2-3/day target range. Tuning trades
frequency for quality; it cannot have significant amounts of both with
this signal family.

## Overall assessment and recommended next step

**RSI(2)+SMA(200) is the one component doing real (if weak) work; VWAP
and volume-weighting are both minor, non-transformative additions on
top of it** — VWAP for a modest frequency bump at redundant-signal
quality, volume-weighting for modestly smaller losses on weak trades,
neither for a better edge. As specified (default, untuned parameters),
this combined system is **not ready to move past Phase 1** — it
achieves the frequency goal only at the basket level and only by
accepting a PF regime (~0.87) that would be a real net loser after any
further real-world friction (spread, funding, slippage beyond the flat
0.10% assumed here). Recommended next step, if this track continues:
before any live-adjacent step, a parameter-sensitivity pass (tighter
RSI threshold, wider VWAP deviation, different stop/exit rules) to see
whether PF can be pulled meaningfully above 1.0 without giving up most
of the frequency gain — this report deliberately did not tune
parameters, so that question is still open, not answered "no" by this
result.

**That question is answered by Phase 2 (see the addendum above): no.**
A disciplined 432-combination in-sample/out-of-sample search found the
best achievable out-of-sample PF is ~1.0-1.07, not 1.3+, and getting
even that far required giving up ~68% of the trade frequency. This
signal family's core weakness is not a missing parameter tweak.
