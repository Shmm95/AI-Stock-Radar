# Approved Portfolio Research Baseline

This document records the active, approved portfolio research baseline. Alternate, legacy, paper, backup, and experimental code paths do not supersede it.

## Strategy

- Direction: long-only.
- Strategy: `TREND_RSI`.
- `EMA20 > EMA50`.
- `Close > EMA20`.
- `RSI14` is between 45 and 70, inclusive.
- The entry condition must be newly valid.
- MACD is not a required entry condition.
- Signals generated at a bar's Close execute at that ticker's next available Open.

## Exits

- Initial stop loss: 5%.
- Highest-Close trailing exit: 7.5%.
- Close-based exit signals retain next-available-Open execution.

## Portfolio risk and allocation

- Maximum open positions: 4.
- Risk per trade: 1%.
- Maximum total open risk: 4%.
- Maximum position allocation: 25%.
- Maximum crypto allocation: 25%.
- Fractional stocks: disabled.
- Fractional crypto: enabled.
- Cash is allowed; the portfolio is not required to remain fully invested.

## Regime handling

- Equities have no regime filter.
- Crypto uses the BTC bull-regime filter. Crypto entries are allowed only when both of these conditions are true:
  - `BTC Close > BTC EMA200`.
  - `BTC EMA50 > BTC EMA200`.
- Both conditions must be true for the crypto bull-regime filter to allow crypto entries.

## Transaction assumptions

- Commission: 0.05% (`0.0005`).
- Minimum fee: 1 USD.
- Slippage: 5 basis points.

## Protected portfolio event ordering

1. Establish the current Opens.
2. Execute pending exits at Open.
3. Process gap stops at the actual Open.
4. Execute accepted pending buys at Open.
5. Process intrabar stops.
6. Mark positions and portfolio at Close.
7. Evaluate and queue close-based exits.
8. Evaluate, rank, and queue newly valid entries.
9. Record portfolio equity.

This ordering must not change without explicit approval.

## Authoritative implementation

The authoritative implementation currently lives mainly in:

- `src/backtest/portfolio_backtest_engine.py`
- `src/backtest/portfolio_backtest_models.py`
- `src/backtest/run_portfolio_backtest.py`

The empty interface files `src/strategy/entry_rules.py`, `src/strategy/exit_rules.py`, and `src/risk/portfolio_limits.py` must not be treated as the active baseline.

No strategy rule, parameter, execution assumption, or portfolio event-ordering behavior may be changed without explicit approval.
