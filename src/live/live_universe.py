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
)
