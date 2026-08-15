"""The live-trading ticker universe -- independent of the frozen research universe.

`src.backtest.run_portfolio_entry_statistics.CONTROLLED_TICKERS` (9 tickers:
7 equity + BTC-USD/ETH-USD) is the frozen, 9-asset research universe every
locked research artifact was built against. `run_portfolio_execution_slippage_stress.py`
imports that constant directly and asserts the frozen snapshot's own market
files equal exactly that set (`set(manifest["market_files"]) != set(CONTROLLED_TICKERS)`).
Widening it in place would break that assertion against the immutable
9-ticker snapshot -- confirmed by reading that check, not assumed.
`run_portfolio_mfe_mae_holding_path_attribution.py` and
`run_portfolio_research_baseline_lock.py` each hold their own independent,
hardcoded 9-ticker copy (not an import of this constant), so they were
already unaffected either way -- confirmed by reading their source, not
assumed either.

`LIVE_CONTROLLED_TICKERS` below is therefore a deliberately separate list
for the live-trading path only (`src/live/data_preparer.py`,
`scripts/run_daily_decision.py`). It starts as the original 9 plus four new
equities (UNH, JPM, XOM, PG) -- see docs/BROADER_UNIVERSE_CANDIDATE_LIST_V1.md
for the sector-diversification research behind that choice. Nothing in
src/backtest/ imports from this module, and this module imports nothing from
src/backtest/ -- the two universes are intentionally decoupled, not just
accidentally different.

Phase 4, stage 1 (13 -> 30): 17 more equities added, selected from the
S&P 500 candidate research pipeline (data/research/sp500_phase1a_*
correlation report + sp500_phase2b_isolated_sanity_report -- automated
per-ticker data-quality pass, isolated-backtest signal sanity check with
no anomaly, and a real-broker tradability check via a live Alpaca
get_all_assets() call, not just the general S&P 500 membership check).
Selection method: for each GICS sector the existing 13 either has zero
equities in (Industrials, Materials, Real Estate, Utilities) or only one
(Consumer Staples, Energy, Financials, Health Care), ranked that sector's
eligible candidates by mean correlation to the existing 13 (ascending --
lowest first) and picked from the top, favoring a different sub-industry
than any name already picked in this batch (e.g. WM/waste-management
alongside NOC/aerospace-defense in Industrials, not two names from the
same sub-industry) to avoid within-selection redundancy of the kind found
in the original 17-candidate research (V/MA, XOM/CVX). Confirmed none of
the 17 pairs with each other above the same 0.75 redundancy threshold.
7 candidates with a low-win-rate anomaly in the isolated sanity check
(SRE, CTVA, TECH, PSA, UPS, VZ, EG) were deliberately deferred, not
included here. ROL and ZTS (solid-sample, high-win-rate anomalies --
57.1%/35 trades and 56.0%/25 trades respectively) were included in place
of two lower-signal, correlation-only picks for the same sector slots
(Industrials, Health Care) -- their isolated backtest quality is
directly demonstrated, not just inferred from correlation.

Phase 4, stage 2 (30 -> 50): a real dollar-P&L/profit-factor pass
(data/research/sp500_phase5_profit_factor_report_*) on stage 1's own 30
tickers found 4 (COR, O, AMT, AWK) net LOSING despite passing the
win-rate-only screen -- all from defensive/low-volatility sectors
(healthcare distribution, REITs, water utility). Stage 2 therefore adds
profit factor as a second selection axis, not just correlation/sector
gaps: for Utilities, Real Estate, Communication Services, Health Care,
and Consumer Staples specifically, a candidate needed an isolated-backtest
profit factor >= 1.2 to be picked at all (data/research/
sp500_phase6_stage2_pf_candidates_* -- same isolated method as Phase 5,
include_benchmark=False to route around the engine's confirmed
_run_equal_weight_benchmark infinite-loop bug on a single fractional-
allowed ticker at 100% cash allocation; engine itself never touched).
Non-defensive sectors used the same PF-ranked, redundancy-checked
approach without that floor (all happened to clear it anyway). SRE was
explicitly added back to the candidate pool per this same Phase 5 result
(20.7% win rate but profit factor 1.398, net +$629) -- one of the 7
originally-deferred low-win-rate tickers turning out to be a real, if
easily-missed, profitable signal; the other 6 (CTVA, PSA, UPS, TECH, EG,
VZ) stay deferred, their poor profit factor (0.32-0.84) confirming the
original deferral was correct. All 20 confirmed to add no within-batch
or vs-existing-30 redundancy above the same 0.75 correlation threshold.

Phase 4, stage 3 (50 -> 82) -- APPROVED, universe widening only. The
fair 79-ticker backtest (VLTO/SNDK/Q excluded, see below) showed the
wider universe diluting net P&L under the unchanged 6-slot/6.0% cap
(-5.6% vs stage 2's own 50-ticker result). Raising the cap to 8/8.0%
alongside this widening was tried and then explicitly abandoned: the
re-run backtest showed the higher cap made results WORSE, not better
-- both in isolation on the unchanged 50-ticker universe (-8.0%) and
combined with the 79-ticker universe (-43.7%), because
risk_per_trade_percent stayed at 1% while more concurrent positions
fragmented the same fixed initial_cash, driving a sharp rise in
ZERO_QUANTITY rejections. scripts/run_daily_decision.py's
LIVE_MAXIMUM_OPEN_POSITIONS/LIVE_MAXIMUM_TOTAL_OPEN_RISK_PERCENT
therefore stay at 6/6.0%, deliberately decoupled from this widening --
the ~5.6% dilution is accepted as-is rather than "fixed" by a change
that backtest evidence showed makes it worse. Same PF-priority +
sector-diversity method as stage 2: candidate pool
is Phase 6's 222 (data/research/sp500_phase6_stage2_pf_candidates_*)
minus stage 2's own 20 additions (202 remaining), plus two Phase 2b
small-sample positive win-rate anomalies that stage 2 had left out
pending separate review -- VLTO (83.3% win rate, 6 trades) and Q
(55.6%, 9 trades, also flagged SHORT_HISTORY in Phase 1: listed
2025-11-14, ~9 months of history) -- for 204 total. Same PF >= 1.2
floor for Utilities, Real Estate, Communication Services, Health Care,
and Consumer Staples; other sectors ranked by profit factor without
that floor, same as stage 2. Took the top 3 candidates per GICS sector
by profit factor -- a natural cutoff given this stage's larger
candidate pool, not a forced quota -- rejecting any candidate that
would pair above the same 0.75 correlation threshold with an
already-live or already-selected-this-batch ticker
(sp500_phase1a_quality_correlation_report_20260812_204205.json's
redundant_pairs; zero rejections this batch). Real Estate had only 2
eligible candidates after the PF floor (AVB, CSGP); both included.
VLTO and Q are both small-sample results (6 and 9 trades) -- their
high profit factor (6.965 and 2.591) is a positive signal, not a
confident estimate. VLTO, SNDK, and Q are all recently-listed and
short-history (~2.8 years, ~17 months, and ~9 months respectively);
Q's history in particular means it will truncate any shared-timeline
aggregate backtest the same way CEG did in stage 2 (confirmed: the
82-ticker as-is backtest's shared window collapses to 2025-11-14;
all three must be excluded from that comparison, not from the live
list itself, to get a fair read -- see the stage-3 backtest report for
the excluded-vs-included comparison). CVX (Energy, PF 1.094 -- no
defensive floor applies there) is
the one pick below the 1.2 threshold used elsewhere; kept per the
stated non-defensive-sector rule but is this batch's weakest signal.
All 32 confirmed tradable via a live Alpaca get_all_assets() call, and
confirmed to add no within-batch or vs-existing-50 redundancy above
the 0.75 threshold.

Crypto quality exception (82 -> 83) -- PREPARED, NOT YET APPROVED.
UNI-USD is prepared under the same explicit quality-not-diversity
exception framework used for ROL and ZTS above. The Alpaca-only refresh
(data/research/uni_alpaca_quality_exception_candidate_20260815_070239.json,
RESEARCH_CANDIDATE_NOT_OFFICIAL) measured isolated PF 1.7771 and daily-
return correlations of 0.631448 vs BTC and 0.716905 vs ETH, reasonably
consistent with the prior session's PF 1.514 / 0.60 / 0.71. UNI still
does not clear the strict < 0.60 correlation gate. Its inclusion therefore
makes no claim that UNI adds independent diversification; the sole rationale
is demonstrated standalone signal quality. This tuple entry is preparation
for review only and must not be staged, committed, or deployed without
separate approval.
"""

from __future__ import annotations

LIVE_CONTROLLED_TICKERS: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
    "BTC-USD",
    "ETH-USD",
    # PREPARED, NOT YET APPROVED -- crypto quality exception, not a
    # diversification claim (Alpaca PF 1.7771; BTC/ETH corr 0.631448/0.716905;
    # candidate artifact: uni_alpaca_quality_exception_candidate_20260815_070239.json).
    "UNI-USD",
    "UNH",
    "JPM",
    "XOM",
    "PG",
    # Phase 4, stage 1 (13 -> 30) -- see module docstring for selection method.
    "NOC",   # Industrials -- aerospace/defense
    "WM",    # Industrials -- waste management
    "ROL",   # Industrials -- pest control services (high-win-rate anomaly, included on demonstrated quality)
    "CF",    # Materials -- fertilizer/chemicals
    "NEM",   # Materials -- gold mining
    "AMT",   # Real Estate -- telecom-tower REIT
    "O",     # Real Estate -- net-lease retail REIT
    "AWK",   # Utilities -- water utility
    "SO",    # Utilities -- electric utility
    "KR",    # Consumer Staples -- grocery retail
    "MO",    # Consumer Staples -- tobacco
    "WMB",   # Energy -- midstream/pipeline (deliberately not another integrated major like XOM)
    "CBOE",  # Financials -- exchange operator
    "PGR",   # Financials -- insurance
    "COR",   # Health Care -- pharma distribution
    "JNJ",   # Health Care -- diversified pharma
    "ZTS",   # Health Care -- animal health (high-win-rate anomaly, included on demonstrated quality)
    # Phase 4, stage 2 (30 -> 50) -- see module docstring for the PF-priority selection method.
    "TKO",   # Communication Services -- sports entertainment (PF 2.836)
    "NFLX",  # Communication Services -- streaming (PF 2.241)
    "MCD",   # Consumer Discretionary -- quick-service restaurants (PF 1.924)
    "AZO",   # Consumer Discretionary -- auto-parts retail (PF 1.850)
    "COST",  # Consumer Staples -- warehouse club retail (PF 2.846)
    "WMT",   # Consumer Staples -- discount retail (PF 2.209)
    "TPL",   # Energy -- land/royalty, not another integrated major (PF 2.034)
    "ERIE",  # Financials -- insurance (PF 3.541)
    "WRB",   # Financials -- specialty insurance (PF 2.498)
    "IDXX",  # Health Care -- veterinary diagnostics (PF 2.321)
    "ABBV",  # Health Care -- biopharma (PF 2.163)
    "CPRT",  # Industrials -- vehicle auction/salvage (PF 3.083)
    "AXON",  # Industrials -- public-safety technology (PF 2.855)
    "FICO",  # Information Technology -- credit-scoring analytics (PF 2.992)
    "MSI",   # Information Technology -- public-safety communications (PF 2.838)
    "SHW",   # Materials -- paints/coatings (PF 2.099)
    "IRM",   # Real Estate -- data-center/records-storage REIT, not another
             # tower/retail REIT (PF 1.790) -- Real Estate's stage-1 picks
             # (AMT, O) were both net losing; this and WELL below required
             # clearing the 1.2 defensive-sector PF floor to be added at all.
    "WELL",  # Real Estate -- healthcare REIT (PF 1.663)
    "CEG",   # Utilities -- nuclear/power generation (PF 2.073)
    "SRE",   # Utilities -- added back per the Phase 5 finding above (PF 1.398)
    # Phase 4, stage 3 (50 -> 82) -- see module docstring: universe widening
    # only, cap/risk% stays 6/6.0% (raising it was tried and abandoned).
    "CMCSA",  # Communication Services -- cable & satellite (PF 1.697)
    "TTWO",   # Communication Services -- interactive entertainment (PF 1.518)
    "TMUS",   # Communication Services -- wireless carrier (PF 1.513)
    "DPZ",    # Consumer Discretionary -- restaurants (PF 1.888)
    "CMG",    # Consumer Discretionary -- restaurants (PF 1.711)
    "HD",     # Consumer Discretionary -- home improvement retail (PF 1.646)
    "ADM",    # Consumer Staples -- agricultural products (PF 1.602)
    "KO",     # Consumer Staples -- soft drinks (PF 1.466)
    "CHD",    # Consumer Staples -- household products (PF 1.460)
    "VLO",    # Energy -- refining/marketing (PF 1.582)
    "OKE",    # Energy -- gas storage/transportation (PF 1.491)
    "CVX",    # Energy -- integrated major (PF 1.094 -- weakest pick this batch, no defensive floor applies)
    "MSCI",   # Financials -- financial data/exchange (PF 2.005)
    "CME",    # Financials -- financial exchange (PF 1.982)
    "ALL",    # Financials -- property & casualty insurance (PF 1.898)
    "WST",    # Health Care -- health care supplies (PF 2.234)
    "HCA",    # Health Care -- hospital operator (PF 2.006)
    "REGN",   # Health Care -- biotechnology (PF 1.930)
    "VLTO",   # Industrials -- environmental/facilities services (PF 6.965 -- small sample, 6 trades)
    "CTAS",   # Industrials -- diversified support services (PF 2.872)
    "BR",     # Industrials -- data processing/outsourcing (PF 2.783)
    "SNDK",   # Information Technology -- storage hardware (PF 5.684 -- also short-history: listed 2025-03, ~17 months)
    "Q",      # Information Technology -- semiconductor materials (PF 2.591 -- small sample, 9 trades, SHORT_HISTORY)
    "TER",    # Information Technology -- semiconductor equipment (PF 2.325)
    "AVY",    # Materials -- paper/plastic packaging (PF 1.495)
    "BALL",   # Materials -- metal/glass/plastic containers (PF 1.438)
    "LIN",    # Materials -- industrial gases (PF 1.420)
    "AVB",    # Real Estate -- multi-family residential REIT (PF 1.337)
    "CSGP",   # Real Estate -- real estate data/services (PF 1.302)
    "FE",     # Utilities -- electric utility (PF 1.987)
    "AES",    # Utilities -- independent power producer (PF 1.744)
    "NEE",    # Utilities -- multi-utility (PF 1.568)
)
