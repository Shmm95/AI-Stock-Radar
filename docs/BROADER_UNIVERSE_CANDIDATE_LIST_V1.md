# Broader-Universe Candidate List V1

## Status

Research-only candidate list and data-quality/correlation pre-screen. This
is **not** a decision to change `CONTROLLED_TICKERS`, not a backtest, and not
an approval to trade any new ticker. No production file was modified. Nothing
here is committed pending owner review.

This is a lighter-weight, informal companion to the separate, formal
`docs/BROADER_UNIVERSE_ROBUSTNESS_V1_SPEC.md` cohort-registration protocol
(which targets a 150-name, point-in-time, survivorship-free equity cohort).
The owner's stated preference for this task is a **modest expansion to
roughly 20-30 total assets**, prioritizing sector diversity and genuinely low
correlation over raw count — not the 150-name cohort.

The controlled development basket remains unchanged: AAPL, AMZN, GOOGL, META,
MSFT, NVDA, TSLA, BTC-USD, ETH-USD (9 assets). Nothing in this document alters
that.

## Method

- Real historical daily bars fetched live via `src.data.alpaca_market_data`
  (the same read-only, IEX-feed, free-tier client the rest of the project
  uses), ~400 calendar days back from 2026-08-11.
- Data quality: row count vs. a 150-row minimum (enough for `EMA50` to
  meaningfully converge), missing-day fraction vs. the expected business-day
  (equity) or calendar-day (crypto) calendar, and largest single-day absolute
  move as an anomaly smell test.
- Correlation: daily `Close`-to-`Close` percent returns, aligned on common
  trading dates (275 common days, 2025-07-09 to 2026-08-11), Pearson
  correlation of each candidate against each of the 7 existing equity
  tickers. Crypto vs. equity correlation was intentionally not computed —
  different trading calendar, and out of scope for the sector-diversification
  question this task asks.
- For context, the **existing 7 tech tickers' own pairwise correlation**
  (mean 0.264, range 0.088-0.488) is the baseline every candidate is judged
  against: a candidate whose correlation to the existing basket is below that
  range is, by definition, less correlated with tech than tech already is
  with itself.
- No backtest, no strategy signal, no `EMA20`/`RSI14`/regime-gate logic was
  run against any candidate. This is a raw-price screen only.

## 1. Sector gap analysis

Mapping the existing 9 assets to GICS-style sectors:

| Ticker | Sector |
|---|---|
| AAPL, MSFT, NVDA | Information Technology |
| GOOGL, META | Communication Services |
| AMZN, TSLA | Consumer Discretionary |
| BTC-USD, ETH-USD | Crypto (separate asset class) |

**Sectors with zero representation:** Health Care, Financials, Energy,
Consumer Staples, Industrials, Utilities, Materials, Real Estate — 8 of 11
GICS sectors. The basket is concentrated in 3 correlated growth/tech-adjacent
sectors plus crypto.

## 2. Candidate pool (pre-screen)

19 large-cap, S&P-500-level, highly liquid names, chosen to cover all 8
missing sectors with 1-4 names each (favoring recognizable mega-caps over
smaller/less liquid names per the project's existing liquid-names-only
principle):

| Sector | Candidates | Rationale |
|---|---|---|
| Health Care | UNH, JNJ, LLY, ABBV | Managed care, diversified pharma/consumer health, high-growth pharma (GLP-1), pharma/biotech — 4 genuinely different sub-industries |
| Financials | JPM, V, MA | Largest US bank, and the two dominant payment networks |
| Energy | XOM, CVX | The two largest US integrated oil majors |
| Consumer Staples | PG, KO, WMT | Household products, beverages, discount/grocery retail |
| Industrials | CAT, HON, UNP | Heavy machinery, diversified aerospace/industrial conglomerate, freight rail |
| Utilities | NEE | Largest US utility by market cap |
| Materials | LIN | Largest US materials company (industrial gases) |
| Real Estate | PLD, AMT | Largest industrial/logistics REIT, largest telecom-tower REIT |

## 3. Data quality results

All 19 candidates (and both existing crypto tickers, re-checked for
completeness) passed:

| Check | Result |
|---|---|
| Minimum 150 rows | All 19/19 candidates: 276 rows (equities) or 400 rows (crypto) — pass |
| Missing-day fraction | 0.035 for every equity (holiday calendar, expected and normal), 0.000 for crypto |
| Anomalous single-day moves | Largest was UNH at 19.6% (a real, publicly reported guidance-cut event in this window, not a data artifact); all others under 16%. No split-adjustment artifacts or gaps detected. |

No candidate was excluded for data quality. Full per-ticker table available
in the underlying research run (row count, date range, missing fraction, max
daily move per ticker).

## 4. Correlation results

### Candidate vs. existing 7 tech equities (mean correlation, ascending —
lower is a better diversifier)

| Candidate | Mean corr. vs. existing 7 | Max corr. (vs. which ticker) |
|---|---|---|
| XOM | -0.168 | -0.098 (MSFT) |
| CVX | -0.148 | -0.061 (MSFT) |
| KO | -0.137 | 0.092 (AAPL) |
| JNJ | -0.130 | 0.034 (AAPL) |
| ABBV | -0.110 | 0.027 (AAPL) |
| AMT | -0.061 | 0.040 (AAPL) |
| WMT | -0.059 | 0.083 (AAPL) |
| LIN | -0.045 | 0.148 (AAPL) |
| PG | -0.032 | 0.159 (AAPL) |
| NEE | -0.017 | 0.078 (GOOGL) |
| LLY | -0.004 | 0.105 (GOOGL) |
| UNP | 0.005 | 0.150 (AAPL) |
| UNH | 0.041 | 0.121 (TSLA) |
| PLD | 0.077 | 0.262 (AAPL) |
| V | 0.101 | 0.236 (AAPL) |
| HON | 0.115 | 0.188 (META) |
| MA | 0.117 | 0.246 (MSFT) |
| CAT | 0.164 | 0.355 (NVDA) |
| JPM | 0.173 | 0.202 (NVDA) |

**Every single candidate's mean correlation to the existing basket is below
the existing basket's own internal tech-vs-tech baseline (0.264).** Several
(Energy, Consumer Staples, Health Care ex-UNH) are outright negatively
correlated over this window — a real hedge, not just "less correlated." None
were excluded on this basis.

### Within-candidate-pool redundancy (the actual finding worth acting on)

Correlation against the existing basket was uniformly low, but two pairs
inside the candidate pool are highly correlated **with each other**, meaning
including both adds size, not independence:

| Pair | Correlation | Read |
|---|---|---|
| V vs. MA | 0.843 | Same factor (payments network) twice |
| XOM vs. CVX | 0.819 | Same factor (integrated oil major) twice |

All other same-sector pairs checked (PG/KO 0.530, JNJ/ABBV 0.480, CAT/HON
0.337, PLD/AMT 0.246, and others) stayed moderate enough that both members
still represent meaningfully different sub-industries (e.g. UNH's managed-care
model correlates with JNJ at only 0.069 — nothing like a duplicate).

## 5. Final recommendation

**Include (17 new tickers; recommended primary list):**
UNH, JNJ, LLY, ABBV, JPM, V, XOM, PG, KO, WMT, CAT, HON, UNP, NEE, LIN, PLD,
AMT.

Rationale: passed data quality cleanly, and each is either negatively or only
weakly correlated with the existing basket (below the basket's own internal
baseline in every case) while remaining a distinct sub-industry bet from
every other included name.

**Low priority / optional secondary pick (2 tickers, not recommended for the
initial expansion):**
- **MA** — redundant with V (0.843 correlation); would double the payments
  bet, not diversify it. Keep as a bench candidate only if V is later dropped
  for an unrelated reason (e.g. a future data or liquidity issue).
- **CVX** — redundant with XOM (0.819 correlation); same reasoning for the
  energy-major bet.

**Excluded: none.** No candidate failed data quality or showed correlation
high enough (against the *existing* basket) to warrant exclusion outright —
the only real risk found was intra-pool duplication, handled above by
demoting one of each pair rather than dropping the sector.

**Resulting total universe size (existing + primary list):**
9 + 17 = **26 assets** — inside the owner's requested ~20-30 range, with 2
further optional names available (28 max) if a future review wants slightly
broader within-sector redundancy instead of the leaner 26.

## Explicitly out of scope for this document

- `CONTROLLED_TICKERS` and every file that reads it were not touched.
- No backtest, walk-forward, or strategy signal was run against any
  candidate — whether the frozen TREND_RSI edge actually holds on these
  names is a separate, later research stage, not addressed here.
- No cron, order-submission, or live-trading code was touched.
- Nothing in this document is committed; it is a candidate list awaiting
  owner review before any further step (e.g. an actual backtest on the
  approved subset) is authorized.
