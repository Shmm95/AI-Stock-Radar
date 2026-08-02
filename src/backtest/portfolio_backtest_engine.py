"""Deterministic shared-cash multi-asset portfolio backtest engine.

Execution rules:
- Entry and exit signals are created only after a ticker bar closes.
- Pending signals execute at that ticker's next available Open.
- Same-day BUY candidates are ranked deterministically before queuing.
- Pending EXIT orders execute before pending BUY orders.
- Gap stop exits use the actual Open.
- Intrabar stop exits use the configured stop price.
- Commission and adverse slippage are included.
- The engine is long-only and supports one position per ticker.
- Stocks and crypto can use separate stop, trailing, and fractional rules.
- Crypto regime filtering is supplied causally through RegimeAllowed.

This module performs historical simulation only. It cannot place broker orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, inf, isfinite
from typing import Iterable

import pandas as pd

from src.backtest.portfolio_backtest_models import (
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    PortfolioBenchmarkResult,
    PortfolioEquityPoint,
    PortfolioPosition,
    PortfolioRejection,
    PortfolioSignal,
    PortfolioTrade,
    utc_now,
)


_REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "EMA20",
    "EMA50",
    "RSI14",
)


@dataclass(slots=True)
class _PendingOrder:
    signal: PortfolioSignal
    submitted_portfolio_bar_index: int


@dataclass(slots=True)
class _MutablePosition:
    ticker: str
    asset_class: str
    entry_timestamp: str
    entry_portfolio_bar_index: int
    quantity: float
    entry_price: float
    entry_fee: float
    stop_loss_price: float
    highest_close: float
    trailing_close_percent: float
    initial_risk_amount: float
    signal_score: float
    signal_reason: str


@dataclass(slots=True)
class _PortfolioState:
    cash: float
    positions: dict[str, _MutablePosition]
    pending_buys: dict[str, _PendingOrder]
    pending_exits: dict[str, _PendingOrder]
    last_prices: dict[str, float]
    trades: list[PortfolioTrade]
    rejections: list[PortfolioRejection]
    equity_curve: list[PortfolioEquityPoint]
    realized_pnl: float
    peak_equity: float
    maximum_drawdown_amount: float
    maximum_drawdown_percent: float
    skipped_signals: int
    maximum_open_positions_observed: int


def _round_money(value: float) -> float:
    return round(float(value), 2)


def _round_price(value: float) -> float:
    return round(float(value), 8)


def _timestamp_to_string(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _is_crypto_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(("-USD", "-EUR", "-GBP"))


def _asset_class(ticker: str) -> str:
    return "CRYPTO" if _is_crypto_ticker(ticker) else "EQUITY"


def _stop_percent(ticker: str, config: PortfolioBacktestConfig) -> float:
    return (
        config.crypto_stop_loss_percent
        if _is_crypto_ticker(ticker)
        else config.stock_stop_loss_percent
    )


def _trailing_percent(
    ticker: str,
    config: PortfolioBacktestConfig,
) -> float:
    return (
        config.crypto_trailing_close_percent
        if _is_crypto_ticker(ticker)
        else config.stock_trailing_close_percent
    )


def _allow_fractional(
    ticker: str,
    config: PortfolioBacktestConfig,
) -> bool:
    return (
        config.allow_fractional_crypto
        if _is_crypto_ticker(ticker)
        else config.allow_fractional_stocks
    )


def _calculate_fee(notional: float, config: PortfolioBacktestConfig) -> float:
    if notional <= 0:
        return 0.0
    return _round_money(
        max(notional * config.commission_rate, config.minimum_fee)
    )


def _apply_buy_slippage(price: float, config: PortfolioBacktestConfig) -> float:
    return _round_price(price * (1 + config.slippage_bps / 10_000))


def _apply_sell_slippage(price: float, config: PortfolioBacktestConfig) -> float:
    return _round_price(price * (1 - config.slippage_bps / 10_000))


def _normalize_quantity(value: float, allow_fractional: bool) -> float:
    if allow_fractional:
        return round(max(float(value), 0.0), 6)
    return float(max(floor(float(value)), 0))


def _normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)
    normalized = normalized.sort_index()
    return normalized.loc[~normalized.index.duplicated(keep="last")]


def _validate_market_data(
    data_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not data_by_ticker:
        raise ValueError("data_by_ticker cannot be empty.")

    validated: dict[str, pd.DataFrame] = {}

    for raw_ticker, raw_data in data_by_ticker.items():
        ticker = raw_ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker keys cannot be empty.")
        if raw_data.empty:
            raise ValueError(f"Market data is empty for {ticker}.")

        data = raw_data.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                column[0] if isinstance(column, tuple) else column
                for column in data.columns
            ]

        missing = [
            column for column in _REQUIRED_COLUMNS if column not in data.columns
        ]
        if missing:
            raise ValueError(
                f"{ticker} is missing columns: {', '.join(missing)}"
            )

        data = _normalize_index(data)
        for column in _REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        if "RegimeAllowed" not in data.columns:
            data["RegimeAllowed"] = True
        else:
            data["RegimeAllowed"] = (
                data["RegimeAllowed"].fillna(False).astype(bool)
            )

        data = data.dropna(subset=list(_REQUIRED_COLUMNS))
        data = data[
            (data["Open"] > 0)
            & (data["High"] > 0)
            & (data["Low"] > 0)
            & (data["Close"] > 0)
        ]

        invalid = data[
            (data["High"] < data[["Open", "Close", "Low"]].max(axis=1))
            | (data["Low"] > data[["Open", "Close", "High"]].min(axis=1))
        ]
        if not invalid.empty:
            raise ValueError(f"Invalid OHLC rows detected for {ticker}.")
        if len(data) < 2:
            raise ValueError(f"At least two valid rows are required for {ticker}.")

        validated[ticker] = data

    common_start = max(data.index.min() for data in validated.values())
    common_end = min(data.index.max() for data in validated.values())
    if common_start > common_end:
        raise ValueError("Ticker histories have no overlapping date range.")

    trimmed: dict[str, pd.DataFrame] = {}
    for ticker, data in validated.items():
        selected = data.loc[(data.index >= common_start) & (data.index <= common_end)]
        if len(selected) < 2:
            raise ValueError(
                f"Insufficient common-period data remains for {ticker}."
            )
        trimmed[ticker] = selected

    return trimmed


def _current_prices(
    state: _PortfolioState,
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    prices = dict(state.last_prices)
    if fallback:
        prices.update(fallback)
    return prices


def _positions_market_value(
    state: _PortfolioState,
    prices: dict[str, float],
) -> float:
    return sum(
        position.quantity * prices.get(ticker, position.entry_price)
        for ticker, position in state.positions.items()
    )


def _crypto_market_value(
    state: _PortfolioState,
    prices: dict[str, float],
) -> float:
    return sum(
        position.quantity * prices.get(ticker, position.entry_price)
        for ticker, position in state.positions.items()
        if position.asset_class == "CRYPTO"
    )


def _open_risk_amount(state: _PortfolioState) -> float:
    return sum(
        max(position.entry_price - position.stop_loss_price, 0.0)
        * position.quantity
        for position in state.positions.values()
    )


def _current_equity(
    state: _PortfolioState,
    prices: dict[str, float],
) -> float:
    return state.cash + _positions_market_value(state, prices)


def _is_entry_setup(row: pd.Series) -> bool:
    return (
        float(row["EMA20"]) > float(row["EMA50"])
        and float(row["Close"]) > float(row["EMA20"])
        and 45 <= float(row["RSI14"]) <= 70
        and bool(row.get("RegimeAllowed", True))
    )


def _entry_score(row: pd.Series) -> float:
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    close = float(row["Close"])
    rsi = float(row["RSI14"])

    trend_spread = max((ema20 / ema50 - 1) * 100, 0.0)
    price_extension = max((close / ema20 - 1) * 100, 0.0)
    rsi_quality = max(0.0, 15.0 - abs(rsi - 57.5) * 1.2)

    score = (
        50.0
        + min(trend_spread * 5.0, 20.0)
        + min(price_extension * 3.0, 15.0)
        + rsi_quality
    )
    return round(min(score, 100.0), 4)


def _build_entry_signal(
    ticker: str,
    data: pd.DataFrame,
    local_bar_index: int,
) -> PortfolioSignal | None:
    if local_bar_index < 1:
        return None

    current = data.iloc[local_bar_index]
    previous = data.iloc[local_bar_index - 1]
    if not _is_entry_setup(current) or _is_entry_setup(previous):
        return None

    score = _entry_score(current)
    return PortfolioSignal(
        timestamp=_timestamp_to_string(data.index[local_bar_index]),
        ticker=ticker,
        action="BUY",
        reference_price=float(current["Close"]),
        score=score,
        technical_score=score,
        confidence=score,
        reason=(
            "TREND_RSI newly valid: EMA20 above EMA50, Close above EMA20, "
            "RSI14 between 45 and 70, regime allowed."
        ),
    )


def _record_rejection(
    state: _PortfolioState,
    *,
    timestamp: str,
    signal: PortfolioSignal,
    reason_code: str,
    reason: str,
    requested_value: float = 0.0,
    prices: dict[str, float] | None = None,
) -> None:
    active_prices = prices or state.last_prices
    equity = _current_equity(state, active_prices)
    open_risk = _open_risk_amount(state)
    crypto_value = _crypto_market_value(state, active_prices)

    state.rejections.append(
        PortfolioRejection(
            timestamp=timestamp,
            ticker=signal.ticker,
            action=signal.action,
            reason_code=reason_code,
            reason=reason,
            score=signal.score,
            requested_value=_round_money(requested_value),
            available_cash=_round_money(state.cash),
            open_position_count=len(state.positions),
            open_risk_percent=round(
                open_risk / equity * 100 if equity > 0 else 0.0,
                4,
            ),
            crypto_allocation_percent=round(
                crypto_value / equity * 100 if equity > 0 else 0.0,
                4,
            ),
        )
    )


def _public_position(position: _MutablePosition) -> PortfolioPosition:
    return PortfolioPosition(
        ticker=position.ticker,
        asset_class=position.asset_class,
        entry_timestamp=position.entry_timestamp,
        entry_portfolio_bar_index=position.entry_portfolio_bar_index,
        quantity=position.quantity,
        entry_price=position.entry_price,
        entry_fee=position.entry_fee,
        stop_loss_price=position.stop_loss_price,
        highest_close=position.highest_close,
        trailing_close_percent=position.trailing_close_percent,
        initial_risk_amount=position.initial_risk_amount,
        signal_score=position.signal_score,
        signal_reason=position.signal_reason,
    )


def _attempt_open_position(
    state: _PortfolioState,
    *,
    pending: _PendingOrder,
    timestamp: str,
    portfolio_bar_index: int,
    raw_open_price: float,
    config: PortfolioBacktestConfig,
    prices: dict[str, float],
) -> None:
    signal = pending.signal
    ticker = signal.ticker

    if ticker in state.positions:
        _record_rejection(
            state,
            timestamp=timestamp,
            signal=signal,
            reason_code="DUPLICATE_TICKER",
            reason=f"An open position already exists for {ticker}.",
            prices=prices,
        )
        return

    if len(state.positions) >= config.maximum_open_positions:
        _record_rejection(
            state,
            timestamp=timestamp,
            signal=signal,
            reason_code="MAX_OPEN_POSITIONS",
            reason="Maximum open-position limit reached.",
            prices=prices,
        )
        return

    entry_price = _apply_buy_slippage(raw_open_price, config)
    stop_percent = _stop_percent(ticker, config)
    stop_loss_price = _round_price(entry_price * (1 - stop_percent / 100))
    stop_distance = entry_price - stop_loss_price

    equity = _current_equity(state, prices)
    maximum_risk_amount = equity * config.risk_per_trade_percent / 100
    maximum_position_value = equity * config.maximum_position_percent / 100

    risk_quantity = maximum_risk_amount / stop_distance
    cash_quantity = max(state.cash, 0.0) / entry_price
    position_quantity = maximum_position_value / entry_price

    quantity = _normalize_quantity(
        min(risk_quantity, cash_quantity, position_quantity),
        _allow_fractional(ticker, config),
    )

    if quantity <= 0:
        _record_rejection(
            state,
            timestamp=timestamp,
            signal=signal,
            reason_code="ZERO_QUANTITY",
            reason="Risk and cash limits produced zero quantity.",
            prices=prices,
        )
        return

    position_value = quantity * entry_price
    entry_fee = _calculate_fee(position_value, config)

    if quantity > 0 and position_value + entry_fee > state.cash:
        commission_multiplier = 1.0 + max(config.commission_rate, 0.0)
        affordable_after_minimum_fee = (
            max(state.cash - config.minimum_fee, 0.0) / entry_price
        )
        affordable_after_commission = (
            max(state.cash, 0.0) / (entry_price * commission_multiplier)
        )
        affordable = min(
            quantity,
            affordable_after_minimum_fee,
            affordable_after_commission,
        )

        fractional = _allow_fractional(ticker, config)
        quantity = _normalize_quantity(affordable, fractional)

        if fractional and quantity > affordable:
            quantity = (
                floor(max(affordable, 0.0) * 1_000_000)
                / 1_000_000
            )

        position_value = quantity * entry_price
        entry_fee = _calculate_fee(position_value, config)

        if quantity > 0 and position_value + entry_fee > state.cash:
            excess = position_value + entry_fee - state.cash

            if fractional:
                quantity_step = 0.000001
                reduction_steps = max(
                    1,
                    ceil(excess / (entry_price * quantity_step)),
                )
                quantity = max(
                    quantity - reduction_steps * quantity_step,
                    0.0,
                )
                quantity = (
                    floor(quantity * 1_000_000)
                    / 1_000_000
                )
            else:
                quantity = max(
                    quantity - max(1, ceil(excess / entry_price)),
                    0.0,
                )

            position_value = quantity * entry_price
            entry_fee = _calculate_fee(position_value, config)

    if quantity <= 0 or position_value + entry_fee > state.cash:
        _record_rejection(
            state,
            timestamp=timestamp,
            signal=signal,
            reason_code="INSUFFICIENT_CASH",
            reason="Available cash is insufficient after fees.",
            requested_value=position_value + entry_fee,
            prices=prices,
        )
        return

    new_risk_amount = quantity * stop_distance
    projected_open_risk = _open_risk_amount(state) + new_risk_amount
    maximum_total_risk = (
        equity * config.maximum_total_open_risk_percent / 100
    )

    if projected_open_risk > maximum_total_risk + 1e-9:
        _record_rejection(
            state,
            timestamp=timestamp,
            signal=signal,
            reason_code="MAX_TOTAL_OPEN_RISK",
            reason="Projected portfolio open risk exceeds the configured limit.",
            requested_value=position_value,
            prices=prices,
        )
        return

    if _is_crypto_ticker(ticker):
        projected_crypto_value = (
            _crypto_market_value(state, prices) + position_value
        )
        maximum_crypto_value = (
            equity * config.maximum_crypto_allocation_percent / 100
        )
        if projected_crypto_value > maximum_crypto_value + 1e-9:
            _record_rejection(
                state,
                timestamp=timestamp,
                signal=signal,
                reason_code="MAX_CRYPTO_ALLOCATION",
                reason="Projected crypto allocation exceeds the configured limit.",
                requested_value=position_value,
                prices=prices,
            )
            return

    state.cash = _round_money(state.cash - position_value - entry_fee)
    state.positions[ticker] = _MutablePosition(
        ticker=ticker,
        asset_class=_asset_class(ticker),
        entry_timestamp=timestamp,
        entry_portfolio_bar_index=portfolio_bar_index,
        quantity=quantity,
        entry_price=entry_price,
        entry_fee=entry_fee,
        stop_loss_price=stop_loss_price,
        highest_close=entry_price,
        trailing_close_percent=_trailing_percent(ticker, config),
        initial_risk_amount=_round_money(new_risk_amount),
        signal_score=signal.score,
        signal_reason=signal.reason,
    )
    state.maximum_open_positions_observed = max(
        state.maximum_open_positions_observed,
        len(state.positions),
    )


def _close_position(
    state: _PortfolioState,
    *,
    ticker: str,
    timestamp: str,
    portfolio_bar_index: int,
    raw_exit_price: float,
    exit_reason: str,
    config: PortfolioBacktestConfig,
) -> None:
    position = state.positions.get(ticker)
    if position is None:
        state.skipped_signals += 1
        state.pending_exits.pop(ticker, None)
        return

    exit_price = _apply_sell_slippage(raw_exit_price, config)
    exit_value = position.quantity * exit_price
    exit_fee = _calculate_fee(exit_value, config)

    gross_pnl = (exit_price - position.entry_price) * position.quantity
    total_fees = position.entry_fee + exit_fee
    net_pnl = gross_pnl - total_fees
    invested_value = position.entry_price * position.quantity
    return_percent = (
        net_pnl / invested_value * 100 if invested_value > 0 else 0.0
    )

    state.cash = _round_money(state.cash + exit_value - exit_fee)
    state.realized_pnl = _round_money(state.realized_pnl + net_pnl)

    state.trades.append(
        PortfolioTrade(
            ticker=ticker,
            asset_class=position.asset_class,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=timestamp,
            entry_portfolio_bar_index=position.entry_portfolio_bar_index,
            exit_portfolio_bar_index=portfolio_bar_index,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            total_fees=_round_money(total_fees),
            gross_pnl=_round_money(gross_pnl),
            net_pnl=_round_money(net_pnl),
            return_percent=round(return_percent, 4),
            holding_period_bars=max(
                portfolio_bar_index - position.entry_portfolio_bar_index,
                0,
            ),
            exit_reason=exit_reason,
            signal_score=position.signal_score,
            signal_reason=position.signal_reason,
        )
    )

    del state.positions[ticker]
    state.pending_exits.pop(ticker, None)
    state.pending_buys.pop(ticker, None)


def _execute_pending_exits_at_open(
    state: _PortfolioState,
    *,
    bars_today: dict[str, pd.Series],
    timestamp: str,
    portfolio_bar_index: int,
    config: PortfolioBacktestConfig,
) -> None:
    executable = sorted(
        (
            (ticker, pending)
            for ticker, pending in state.pending_exits.items()
            if ticker in bars_today
            and pending.submitted_portfolio_bar_index < portfolio_bar_index
        ),
        key=lambda item: (
            item[1].submitted_portfolio_bar_index,
            item[0],
        ),
    )

    for ticker, _pending in executable:
        if ticker in state.positions:
            _close_position(
                state,
                ticker=ticker,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_exit_price=float(bars_today[ticker]["Open"]),
                exit_reason="EXIT_SIGNAL_NEXT_OPEN",
                config=config,
            )
        else:
            state.pending_exits.pop(ticker, None)
            state.skipped_signals += 1


def _check_gap_stops(
    state: _PortfolioState,
    *,
    bars_today: dict[str, pd.Series],
    timestamp: str,
    portfolio_bar_index: int,
    config: PortfolioBacktestConfig,
) -> None:
    for ticker in sorted(list(state.positions)):
        if ticker not in bars_today:
            continue
        position = state.positions.get(ticker)
        if position is None:
            continue
        open_price = float(bars_today[ticker]["Open"])
        if open_price <= position.stop_loss_price:
            _close_position(
                state,
                ticker=ticker,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_exit_price=open_price,
                exit_reason="GAP_STOP_LOSS",
                config=config,
            )


def _execute_pending_buys_at_open(
    state: _PortfolioState,
    *,
    bars_today: dict[str, pd.Series],
    timestamp: str,
    portfolio_bar_index: int,
    config: PortfolioBacktestConfig,
) -> None:
    executable = sorted(
        (
            (ticker, pending)
            for ticker, pending in state.pending_buys.items()
            if ticker in bars_today
            and pending.submitted_portfolio_bar_index < portfolio_bar_index
        ),
        key=lambda item: (
            item[1].submitted_portfolio_bar_index,
            item[1].signal.priority_rank,
            -item[1].signal.score,
            item[0],
        ),
    )

    for ticker, pending in executable:
        state.pending_buys.pop(ticker, None)
        open_price = float(bars_today[ticker]["Open"])
        execution_prices = _current_prices(state, {ticker: open_price})
        _attempt_open_position(
            state,
            pending=pending,
            timestamp=timestamp,
            portfolio_bar_index=portfolio_bar_index,
            raw_open_price=open_price,
            config=config,
            prices=execution_prices,
        )


def _check_intrabar_stops(
    state: _PortfolioState,
    *,
    bars_today: dict[str, pd.Series],
    timestamp: str,
    portfolio_bar_index: int,
    config: PortfolioBacktestConfig,
) -> None:
    for ticker in sorted(list(state.positions)):
        if ticker not in bars_today:
            continue
        position = state.positions.get(ticker)
        if position is None:
            continue
        low_price = float(bars_today[ticker]["Low"])
        if low_price <= position.stop_loss_price:
            _close_position(
                state,
                ticker=ticker,
                timestamp=timestamp,
                portfolio_bar_index=portfolio_bar_index,
                raw_exit_price=position.stop_loss_price,
                exit_reason="STOP_LOSS",
                config=config,
            )


def _queue_close_based_exits(
    state: _PortfolioState,
    *,
    bars_today: dict[str, pd.Series],
    timestamp: str,
    portfolio_bar_index: int,
) -> None:
    for ticker in sorted(state.positions):
        if ticker not in bars_today or ticker in state.pending_exits:
            continue

        position = state.positions[ticker]
        row = bars_today[ticker]
        close = float(row["Close"])
        position.highest_close = max(position.highest_close, close)

        trend_exit = float(row["EMA20"]) < float(row["EMA50"])
        trailing_level = position.highest_close * (
            1 - position.trailing_close_percent / 100
        )
        trailing_exit = close <= trailing_level

        if not trend_exit and not trailing_exit:
            continue

        reason_parts: list[str] = []
        if trend_exit:
            reason_parts.append("EMA20 below EMA50")
        if trailing_exit:
            reason_parts.append(
                f"Close below {position.trailing_close_percent:g}% "
                "highest-Close trailing level"
            )

        signal = PortfolioSignal(
            timestamp=timestamp,
            ticker=ticker,
            action="EXIT",
            reference_price=close,
            score=position.signal_score,
            technical_score=position.signal_score,
            confidence=position.signal_score,
            reason="; ".join(reason_parts),
        )
        state.pending_exits[ticker] = _PendingOrder(
            signal=signal,
            submitted_portfolio_bar_index=portfolio_bar_index,
        )


def _queue_ranked_entry_signals(
    state: _PortfolioState,
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    local_indices: dict[str, int],
    timestamp: str,
    portfolio_bar_index: int,
) -> None:
    candidates: list[PortfolioSignal] = []

    for ticker in sorted(local_indices):
        if (
            ticker in state.positions
            or ticker in state.pending_buys
            or ticker in state.pending_exits
        ):
            continue

        signal = _build_entry_signal(
            ticker,
            data_by_ticker[ticker],
            local_indices[ticker],
        )
        if signal is not None:
            candidates.append(signal)

    candidates.sort(key=lambda item: (-item.score, item.ticker))

    for rank, signal in enumerate(candidates, start=1):
        ranked_signal = PortfolioSignal(
            timestamp=signal.timestamp,
            ticker=signal.ticker,
            action=signal.action,
            reference_price=signal.reference_price,
            score=signal.score,
            technical_score=signal.technical_score,
            confidence=signal.confidence,
            reason=signal.reason,
            priority_rank=rank,
        )
        state.pending_buys[signal.ticker] = _PendingOrder(
            signal=ranked_signal,
            submitted_portfolio_bar_index=portfolio_bar_index,
        )


def _record_equity(
    state: _PortfolioState,
    *,
    timestamp: str,
    replace_last: bool = False,
) -> None:
    prices = state.last_prices
    position_value = _positions_market_value(state, prices)
    crypto_value = _crypto_market_value(state, prices)
    total_equity = state.cash + position_value
    open_risk = _open_risk_amount(state)

    unrealized_pnl = sum(
        (
            prices.get(ticker, position.entry_price) - position.entry_price
        )
        * position.quantity
        - position.entry_fee
        for ticker, position in state.positions.items()
    )

    state.peak_equity = max(state.peak_equity, total_equity)
    drawdown_amount = max(state.peak_equity - total_equity, 0.0)
    drawdown_percent = (
        drawdown_amount / state.peak_equity * 100
        if state.peak_equity > 0
        else 0.0
    )

    state.maximum_drawdown_amount = max(
        state.maximum_drawdown_amount,
        drawdown_amount,
    )
    state.maximum_drawdown_percent = max(
        state.maximum_drawdown_percent,
        drawdown_percent,
    )

    point = PortfolioEquityPoint(
        timestamp=timestamp,
        cash=_round_money(state.cash),
        positions_market_value=_round_money(position_value),
        total_equity=_round_money(total_equity),
        realized_pnl=_round_money(state.realized_pnl),
        unrealized_pnl=_round_money(unrealized_pnl),
        open_position_count=len(state.positions),
        open_risk_amount=_round_money(open_risk),
        open_risk_percent=round(
            open_risk / total_equity * 100 if total_equity > 0 else 0.0,
            4,
        ),
        crypto_market_value=_round_money(crypto_value),
        crypto_allocation_percent=round(
            crypto_value / total_equity * 100 if total_equity > 0 else 0.0,
            4,
        ),
        peak_equity=_round_money(state.peak_equity),
        drawdown_amount=_round_money(drawdown_amount),
        drawdown_percent=round(drawdown_percent, 4),
    )

    if replace_last and state.equity_curve:
        state.equity_curve[-1] = point
    else:
        state.equity_curve.append(point)


def _calculate_statistics(
    trades: list[PortfolioTrade],
) -> dict[str, float | int]:
    winners = [trade for trade in trades if trade.net_pnl > 0]
    losers = [trade for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(trade.net_pnl for trade in winners)
    gross_loss = abs(sum(trade.net_pnl for trade in losers))

    if gross_loss == 0:
        profit_factor = inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    completed = len(trades)
    return {
        "completed_trades": completed,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_percent": round(
            len(winners) / completed * 100 if completed else 0.0,
            4,
        ),
        "gross_profit": _round_money(gross_profit),
        "gross_loss": _round_money(gross_loss),
        "profit_factor": (
            round(profit_factor, 4) if isfinite(profit_factor) else inf
        ),
        "total_fees": _round_money(
            sum(trade.total_fees for trade in trades)
        ),
    }


def _find_local_bar_index(data: pd.DataFrame, timestamp: pd.Timestamp) -> int:
    location = data.index.get_loc(timestamp)
    if isinstance(location, slice):
        return int(location.start)
    if isinstance(location, Iterable) and not isinstance(location, (str, bytes)):
        return int(list(location)[0])
    return int(location)


def _run_equal_weight_benchmark(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    timeline: pd.DatetimeIndex,
    config: PortfolioBacktestConfig,
) -> PortfolioBenchmarkResult:
    tickers = sorted(data_by_ticker)
    allocation = config.initial_cash / len(tickers)
    cash = config.initial_cash
    holdings: dict[str, tuple[float, float]] = {}
    total_fees = 0.0

    for ticker in tickers:
        data = data_by_ticker[ticker]
        first_timestamp = data.index[data.index >= timeline[0]][0]
        raw_price = float(data.loc[first_timestamp, "Open"])
        entry_price = _apply_buy_slippage(raw_price, config)
        quantity = _normalize_quantity(
            allocation / entry_price,
            _allow_fractional(ticker, config),
        )
        value = quantity * entry_price
        fee = _calculate_fee(value, config)
        while quantity > 0 and value + fee > cash:
            if _allow_fractional(ticker, config):
                quantity = _normalize_quantity(
                    max(cash - config.minimum_fee, 0.0) / entry_price,
                    True,
                )
            else:
                quantity = max(quantity - 1, 0)
            value = quantity * entry_price
            fee = _calculate_fee(value, config)

        if quantity <= 0:
            continue
        cash -= value + fee
        total_fees += fee
        holdings[ticker] = (quantity, entry_price)

    peak = config.initial_cash
    maximum_drawdown = 0.0
    maximum_drawdown_percent = 0.0
    last_prices: dict[str, float] = {}

    for timestamp in timeline:
        for ticker, data in data_by_ticker.items():
            if timestamp in data.index:
                last_prices[ticker] = float(data.loc[timestamp, "Close"])
        equity = cash + sum(
            quantity * last_prices.get(ticker, entry_price)
            for ticker, (quantity, entry_price) in holdings.items()
        )
        peak = max(peak, equity)
        drawdown = max(peak - equity, 0.0)
        drawdown_percent = drawdown / peak * 100 if peak > 0 else 0.0
        maximum_drawdown = max(maximum_drawdown, drawdown)
        maximum_drawdown_percent = max(
            maximum_drawdown_percent,
            drawdown_percent,
        )

    ending_equity = cash
    for ticker, (quantity, entry_price) in holdings.items():
        data = data_by_ticker[ticker]
        final_timestamp = data.index[data.index <= timeline[-1]][-1]
        raw_exit = float(data.loc[final_timestamp, "Close"])
        exit_price = _apply_sell_slippage(raw_exit, config)
        value = quantity * exit_price
        fee = _calculate_fee(value, config)
        ending_equity += value - fee
        total_fees += fee

    total_return = ending_equity - config.initial_cash
    return PortfolioBenchmarkResult(
        initial_cash=_round_money(config.initial_cash),
        ending_equity=_round_money(ending_equity),
        total_return_amount=_round_money(total_return),
        total_return_percent=round(
            total_return / config.initial_cash * 100,
            4,
        ),
        maximum_drawdown_amount=_round_money(maximum_drawdown),
        maximum_drawdown_percent=round(maximum_drawdown_percent, 4),
        total_fees=_round_money(total_fees),
        invested_tickers=len(holdings),
        uninvested_cash=_round_money(cash),
    )


def run_portfolio_backtest(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    config: PortfolioBacktestConfig | None = None,
    include_benchmark: bool = True,
) -> PortfolioBacktestResult:
    """Run a deterministic shared-cash long-only portfolio backtest."""

    active_config = config or PortfolioBacktestConfig()
    active_config.validate()
    market_data = _validate_market_data(data_by_ticker)
    started_at = utc_now()

    timeline = pd.DatetimeIndex(
        sorted(
            set().union(*(set(data.index) for data in market_data.values()))
        )
    )
    if len(timeline) < 2:
        raise ValueError("Portfolio timeline requires at least two dates.")

    state = _PortfolioState(
        cash=_round_money(active_config.initial_cash),
        positions={},
        pending_buys={},
        pending_exits={},
        last_prices={},
        trades=[],
        rejections=[],
        equity_curve=[],
        realized_pnl=0.0,
        peak_equity=float(active_config.initial_cash),
        maximum_drawdown_amount=0.0,
        maximum_drawdown_percent=0.0,
        skipped_signals=0,
        maximum_open_positions_observed=0,
    )

    for portfolio_bar_index, timestamp_value in enumerate(timeline):
        timestamp = pd.Timestamp(timestamp_value)
        timestamp_string = _timestamp_to_string(timestamp)

        bars_today: dict[str, pd.Series] = {}
        local_indices: dict[str, int] = {}

        for ticker, data in market_data.items():
            if timestamp in data.index:
                bars_today[ticker] = data.loc[timestamp]
                local_indices[ticker] = _find_local_bar_index(data, timestamp)
                state.last_prices[ticker] = float(data.loc[timestamp, "Open"])

        _execute_pending_exits_at_open(
            state,
            bars_today=bars_today,
            timestamp=timestamp_string,
            portfolio_bar_index=portfolio_bar_index,
            config=active_config,
        )
        _check_gap_stops(
            state,
            bars_today=bars_today,
            timestamp=timestamp_string,
            portfolio_bar_index=portfolio_bar_index,
            config=active_config,
        )
        _execute_pending_buys_at_open(
            state,
            bars_today=bars_today,
            timestamp=timestamp_string,
            portfolio_bar_index=portfolio_bar_index,
            config=active_config,
        )
        _check_intrabar_stops(
            state,
            bars_today=bars_today,
            timestamp=timestamp_string,
            portfolio_bar_index=portfolio_bar_index,
            config=active_config,
        )

        for ticker, row in bars_today.items():
            state.last_prices[ticker] = float(row["Close"])

        _queue_close_based_exits(
            state,
            bars_today=bars_today,
            timestamp=timestamp_string,
            portfolio_bar_index=portfolio_bar_index,
        )
        _queue_ranked_entry_signals(
            state,
            data_by_ticker=market_data,
            local_indices=local_indices,
            timestamp=timestamp_string,
            portfolio_bar_index=portfolio_bar_index,
        )
        _record_equity(state, timestamp=timestamp_string)

    if active_config.force_close_at_end and state.positions:
        final_timestamp = pd.Timestamp(timeline[-1])
        final_string = _timestamp_to_string(final_timestamp)
        for ticker in sorted(list(state.positions)):
            data = market_data[ticker]
            final_ticker_timestamp = data.index[data.index <= final_timestamp][-1]
            raw_exit = float(data.loc[final_ticker_timestamp, "Close"])
            _close_position(
                state,
                ticker=ticker,
                timestamp=final_string,
                portfolio_bar_index=len(timeline) - 1,
                raw_exit_price=raw_exit,
                exit_reason="FORCE_CLOSE_END",
                config=active_config,
            )
            state.last_prices[ticker] = raw_exit
        _record_equity(state, timestamp=final_string, replace_last=True)

    statistics = _calculate_statistics(state.trades)
    ending_equity = (
        state.equity_curve[-1].total_equity
        if state.equity_curve
        else state.cash
    )
    total_return = ending_equity - active_config.initial_cash

    average_exposure = (
        sum(
            point.positions_market_value / point.total_equity * 100
            if point.total_equity > 0
            else 0.0
            for point in state.equity_curve
        )
        / len(state.equity_curve)
        if state.equity_curve
        else 0.0
    )

    benchmark = (
        _run_equal_weight_benchmark(
            data_by_ticker=market_data,
            timeline=timeline,
            config=active_config,
        )
        if include_benchmark
        else None
    )

    return PortfolioBacktestResult(
        config=active_config,
        tickers=tuple(sorted(market_data)),
        start_date=_timestamp_to_string(timeline[0]),
        end_date=_timestamp_to_string(timeline[-1]),
        started_at=started_at,
        finished_at=utc_now(),
        initial_cash=_round_money(active_config.initial_cash),
        ending_cash=_round_money(state.cash),
        ending_equity=_round_money(ending_equity),
        total_return_amount=_round_money(total_return),
        total_return_percent=round(
            total_return / active_config.initial_cash * 100,
            4,
        ),
        maximum_drawdown_amount=_round_money(
            state.maximum_drawdown_amount
        ),
        maximum_drawdown_percent=round(
            state.maximum_drawdown_percent,
            4,
        ),
        completed_trades=int(statistics["completed_trades"]),
        winning_trades=int(statistics["winning_trades"]),
        losing_trades=int(statistics["losing_trades"]),
        win_rate_percent=float(statistics["win_rate_percent"]),
        gross_profit=float(statistics["gross_profit"]),
        gross_loss=float(statistics["gross_loss"]),
        profit_factor=float(statistics["profit_factor"]),
        total_fees=float(statistics["total_fees"]),
        average_exposure_percent=round(average_exposure, 4),
        maximum_open_positions_observed=(
            state.maximum_open_positions_observed
        ),
        rejected_signals=len(state.rejections),
        skipped_signals=state.skipped_signals,
        benchmark=benchmark,
        open_positions=tuple(
            _public_position(position)
            for position in sorted(
                state.positions.values(),
                key=lambda item: item.ticker,
            )
        ),
        trades=state.trades,
        rejections=state.rejections,
        equity_curve=state.equity_curve,
    )


def print_portfolio_backtest_result(result: PortfolioBacktestResult) -> None:
    """Print a readable portfolio summary."""

    print()
    print("=" * 96)
    print("AI STOCK RADAR — PORTFOLIO BACKTEST")
    print("=" * 96)
    print(f"Period:                         {result.start_date[:10]} to {result.end_date[:10]}")
    print(f"Tickers:                        {len(result.tickers)}")
    print(f"Initial cash:                   {result.initial_cash:,.2f}")
    print(f"Ending equity:                  {result.ending_equity:,.2f}")
    print(f"Total return:                   {result.total_return_amount:+,.2f} ({result.total_return_percent:+.4f}%)")
    print(f"Maximum drawdown:               {result.maximum_drawdown_amount:,.2f} ({result.maximum_drawdown_percent:.4f}%)")
    print(f"Completed trades:               {result.completed_trades}")
    print(f"Win rate:                       {result.win_rate_percent:.4f}%")
    print(f"Profit factor:                  {result.profit_factor}")
    print(f"Total fees:                     {result.total_fees:,.2f}")
    print(f"Average exposure:               {result.average_exposure_percent:.4f}%")
    print(f"Maximum positions observed:     {result.maximum_open_positions_observed}")
    print(f"Rejected signals:               {result.rejected_signals}")
    print(f"Skipped signals:                {result.skipped_signals}")

    if result.benchmark is not None:
        print()
        print("EQUAL-WEIGHT BUY-AND-HOLD BENCHMARK")
        print(f"Benchmark ending equity:        {result.benchmark.ending_equity:,.2f}")
        print(f"Benchmark return:               {result.benchmark.total_return_percent:+.4f}%")
        print(f"Benchmark maximum drawdown:     {result.benchmark.maximum_drawdown_percent:.4f}%")
        print(f"Strategy excess return:         {result.total_return_percent - result.benchmark.total_return_percent:+.4f}%")

    print("=" * 96)
