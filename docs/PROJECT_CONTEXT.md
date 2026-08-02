# Project Context

## Objective

AI-Stock-Radar is intended to become a deterministic Python trading platform that progresses from historical research through portfolio validation, broader-universe robustness, paper validation, carefully controlled live trading, and eventually controlled automation.

The platform must remain causal, reproducible, auditable, and governed by explicit rules. AI may assist research and review, but it is not the primary trading or execution brain.

## Architecture

The repository currently contains three distinct paths. They share some market-data and indicator utilities, but they must not be treated as one interchangeable source of trading authority.

### Radar and main path

`src/main.py` and the scanner, scoring, decision, news, fundamental, universe, pipeline, and reporting modules form the radar path. This path downloads and analyzes assets, creates rankings and recommendations, and writes radar reports. It is not the authoritative portfolio research engine.

### Portfolio research path

The portfolio research path lives mainly under `src/backtest/`. It prepares historical market data, applies causal indicators and regime masks, and runs deterministic single-asset and shared-cash portfolio simulations. It also contains benchmark, ablation, walk-forward, replay/provenance, attribution, and entry-diagnostic research.

The shared-cash portfolio engine is the authoritative implementation for the current research baseline. It owns the active entry logic, execution ordering, position sizing constraints, portfolio limits, stops, trailing exits, fees, slippage, and portfolio accounting used by current portfolio studies.

### Paper path

`src/paper/`, `src/execution/`, the trade-plan and standalone risk modules, and paper performance analytics form a pre-existing local paper-trading path. Their presence does not authorize paper-trading work, does not advance the current stage, and does not change the approved roadmap.

Existing alternate, legacy, backup, experimental, and historical code paths are not automatically authoritative. Their rules and defaults may differ from the active portfolio research baseline.

## Controlled research basket

The current controlled development and research basket is:

- AAPL
- AMZN
- GOOGL
- META
- MSFT
- NVDA
- TSLA
- BTC-USD
- ETH-USD

These nine assets remain only the controlled development basket and are not the final universe.

The planned V1 universe target is 150 liquid equities and 20 liquid cryptocurrencies. Broader-universe work must occur only at its approved roadmap stage.
