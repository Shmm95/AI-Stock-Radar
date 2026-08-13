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
)
