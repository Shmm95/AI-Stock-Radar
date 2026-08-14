"""Phase-1 proof-of-concept research: a crypto-only, hourly-bar, combined
mean-reversion prototype, kept fully isolated from the frozen daily
EMA20/EMA50/RSI14 system and its protected engine
(`src/backtest/portfolio_backtest_engine.py`). Nothing here is imported
by, or imports from, `src/live/` or the live universe files. See
`docs/MEAN_REVERSION_PROTOTYPE_V1_REPORT.md` for the research report
this package's scripts were used to produce.
"""
