"""Pre-registered, blind-selected S&P MidCap 400 control-arm universe --
completely separate from `live_universe.LIVE_CONTROLLED_TICKERS`. Never
imported by `run_daily_decision.py`'s live path or by anything under
`src/backtest/`. This exists so the same frozen EMA20/EMA50/RSI14 signal
logic can be run, in complete isolation, against a ticker set that was
selected WITHOUT looking at any performance data -- a genuine blind
control arm for the PF-selected 83-ticker live universe.

Source: workspace-c commit bfa2763 ("Pre-registered MidCap400 control
universe — frozen before backtest, seed=40020260815, artifact
SHA-256=746729141d43413f34150926bd9bdc57e4b9f927dfb9904dc5b2623113e975f2"),
snapshot file data/research/sp_midcap400_control_universe_snapshot_20260815_165011.json.

Selection method (verified from that snapshot, not re-derived):
for each of the 11 GICS sectors (lexicographically sorted), rank the
eligible S&P MidCap 400 tickers in that sector by
SHA-256(seed|sector|ticker) and take the top 4 -- a fully deterministic,
performance-blind hash selection. `performance_input_fields` in the
snapshot is an empty list -- confirmed no backtest/PF data was consulted.
Seed: 40020260815. 44 tickers total (4 x 11 sectors).

Non-overlap confirmed in the same snapshot
(`current_live_universe_identity_check`): checked against 80 live
equity tickers, 0 overlap tickers excluded (because there were none --
the pool was already disjoint).

Tradability confirmed in the same snapshot
(`alpaca_tradability_verification`): real `TradingClient(paper=True).get_all_assets()`
call (`mock_used: false, real_api_call: true`), all 400 Wikipedia-sourced
MidCap constituents matched and tradable at snapshot time
(2026-08-15T16:50:04Z). This module does NOT re-verify tradability --
see the control-arm setup report for a fresh, real re-check done today.

Status: RESEARCH_CANDIDATE_NOT_OFFICIAL, PREPARED — NOT YET DEPLOYED.
No live integration exists yet. This file is data only.
"""

from __future__ import annotations

CONTROL_UNIVERSE_SEED = 40020260815
CONTROL_UNIVERSE_SOURCE_COMMIT = "bfa2763bf3e26a240657bfc090232c4b5ad16ac3"
CONTROL_UNIVERSE_SOURCE_SNAPSHOT = "data/research/sp_midcap400_control_universe_snapshot_20260815_165011.json"

CONTROL_UNIVERSE_TICKERS: tuple[str, ...] = (
    "AA",     # Materials -- Alcoa
    "ASB",    # Financials -- Associated Bank
    "AVNT",   # Materials -- Avient
    "BJ",     # Consumer Staples -- BJ's Wholesale Club
    "CDE",    # Materials -- Coeur Mining
    "CNO",    # Financials -- CNO Financial Group
    "CSL",    # Industrials -- Carlisle Companies
    "DTM",    # Energy -- DT Midstream
    "EXEL",   # Health Care -- Exelixis
    "FR",     # Real Estate -- First Industrial Realty Trust
    "GXO",    # Industrials -- GXO Logistics
    "HGV",    # Consumer Discretionary -- Hilton Grand Vacations
    "HR",     # Real Estate -- Healthcare Realty Trust
    "INGR",   # Consumer Staples -- Ingredion
    "JAZZ",   # Health Care -- Jazz Pharmaceuticals
    "JLL",    # Real Estate -- Jones Lang LaSalle
    "MANH",   # Information Technology -- Manhattan Associates
    "MTDR",   # Energy -- Matador Resources
    "NJR",    # Utilities -- New Jersey Resources
    "NVST",   # Health Care -- Envista Holdings
    "NXST",   # Communication Services -- Nexstar Media Group
    "NYT",    # Communication Services -- New York Times Company
    "OGE",    # Utilities -- OGE Energy
    "ORA",    # Utilities -- Ormat Technologies
    "OVV",    # Energy -- Ovintiv
    "PEGA",   # Information Technology -- Pegasystems
    "PINS",   # Communication Services -- Pinterest
    "POST",   # Consumer Staples -- Post Holdings
    "PVH",    # Consumer Discretionary -- PVH Corp.
    "REXR",   # Real Estate -- Rexford Industrial Realty
    "RGEN",   # Health Care -- Repligen
    "RGLD",   # Materials -- Royal Gold
    "ROKU",   # Communication Services -- Roku, Inc.
    "SAM",    # Consumer Staples -- Boston Beer Company
    "SWX",    # Utilities -- Southwest Gas Corp
    "TCBI",   # Financials -- Texas Capital Bancshares
    "TREX",   # Industrials -- Trex
    "TTMI",   # Information Technology -- TTM Technologies
    "UNM",    # Financials -- Unum
    "VC",     # Consumer Discretionary -- Visteon
    "VIAV",   # Information Technology -- Viavi Solutions
    "VNOM",   # Energy -- Viper Energy
    "WHR",    # Consumer Discretionary -- Whirlpool Corporation
    "WSO",    # Industrials -- Watsco
)
