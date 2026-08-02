"""Deterministic single-asset historical backtest engine.

Execution rules:
- Signals are created after a historical bar closes.
- BUY and EXIT signals execute at the next bar's Open.
- Stop-loss and take-profit remain active overnight.
- A gap through stop or target exits at the actual Open.
- Intrabar exits use the current bar's High and Low.
- If stop and target are both touched in one bar, stop is selected
  conservatively because the intrabar path is unknown.
- Commission and adverse slippage are included.
- V1 supports one long position at a time.

This module performs historical simulation only.
It cannot place paper or real-money broker orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, inf, isfinite
from typing import Callable

import pandas as pd

from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestEquityPoint,
    BacktestPosition,
    BacktestResult,
    BacktestSignal,
    BacktestTrade,
    utc_now,
)


SignalProvider = Callable[
    [pd.DataFrame, int, str],
    BacktestSignal | None,
]


@dataclass(slots=True)
class _PendingOrder:
    """A signal waiting for execution at the next Open."""

    signal: BacktestSignal
    submitted_bar_index: int


@dataclass(slots=True)
class _MutablePosition:
    """Internal mutable position state."""

    ticker: str
    entry_timestamp: str
    entry_bar_index: int

    quantity: float
    entry_price: float
    entry_fee: float

    stop_loss_price: float
    take_profit_price: float
    signal_reason: str


@dataclass(slots=True)
class _BacktestState:
    """Internal mutable account state."""

    cash: float
    position: _MutablePosition | None

    pending_buy: _PendingOrder | None
    pending_exit: _PendingOrder | None

    trades: list[BacktestTrade]
    equity_curve: list[BacktestEquityPoint]

    realized_pnl: float
    peak_equity: float
    maximum_drawdown_amount: float
    maximum_drawdown_percent: float

    rejected_signals: int
    skipped_signals: int


def _round_money(
    value: float,
) -> float:
    """Round monetary values to two decimals."""

    return round(
        float(value),
        2,
    )


def _round_price(
    value: float,
) -> float:
    """Round prices while retaining crypto precision."""

    return round(
        float(value),
        8,
    )


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert an index value into a stable timestamp string."""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(
        value
    )


def _normalize_quantity(
    value: float,
    *,
    allow_fractional: bool,
) -> float:
    """Normalize equity-share or crypto quantity."""

    if allow_fractional:
        return round(
            max(
                float(value),
                0.0,
            ),
            6,
        )

    return float(
        max(
            floor(
                float(value)
            ),
            0,
        )
    )


def _validate_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize OHLC market data."""

    if data.empty:
        raise ValueError(
            "Backtest market data cannot be empty."
        )

    normalized = data.copy()

    if isinstance(
        normalized.columns,
        pd.MultiIndex,
    ):
        normalized.columns = [
            (
                column[0]
                if isinstance(
                    column,
                    tuple,
                )
                else column
            )
            for column in normalized.columns
        ]

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing_columns = (
        required_columns
        - set(
            normalized.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Backtest data is missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()

    normalized = normalized.loc[
        ~normalized.index.duplicated(
            keep="last"
        )
    ]

    for column in required_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized = normalized.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    normalized = normalized[
        (
            normalized["Open"] > 0
        )
        & (
            normalized["High"] > 0
        )
        & (
            normalized["Low"] > 0
        )
        & (
            normalized["Close"] > 0
        )
    ]

    invalid_rows = normalized[
        (
            normalized["High"]
            < normalized[
                [
                    "Open",
                    "Close",
                    "Low",
                ]
            ].max(
                axis=1
            )
        )
        | (
            normalized["Low"]
            > normalized[
                [
                    "Open",
                    "Close",
                    "High",
                ]
            ].min(
                axis=1
            )
        )
    ]

    if not invalid_rows.empty:
        raise ValueError(
            "Backtest data contains invalid OHLC relationships."
        )

    if len(
        normalized
    ) < 2:
        raise ValueError(
            "Backtest requires at least two valid bars."
        )

    return normalized


def _calculate_fee(
    *,
    transaction_value: float,
    config: BacktestConfig,
) -> float:
    """Calculate simulated commission."""

    proportional_fee = (
        abs(
            transaction_value
        )
        * config.commission_rate
    )

    return _round_money(
        max(
            proportional_fee,
            config.minimum_fee,
        )
    )


def _apply_buy_slippage(
    *,
    raw_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse BUY slippage."""

    return _round_price(
        raw_price
        * (
            1
            + config.slippage_bps
            / 10_000
        )
    )


def _apply_sell_slippage(
    *,
    raw_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse SELL slippage."""

    return _round_price(
        raw_price
        * (
            1
            - config.slippage_bps
            / 10_000
        )
    )


def _position_market_value(
    position: _MutablePosition | None,
    mark_price: float,
) -> float:
    """Calculate marked-to-market position value."""

    if position is None:
        return 0.0

    return (
        position.quantity
        * mark_price
    )


def _current_equity(
    *,
    state: _BacktestState,
    mark_price: float,
) -> float:
    """Calculate current marked-to-market equity."""

    return (
        state.cash
        + _position_market_value(
            state.position,
            mark_price,
        )
    )


def _calculate_entry_quantity(
    *,
    state: _BacktestState,
    entry_price: float,
    config: BacktestConfig,
) -> tuple[float, float]:
    """Calculate risk-, allocation-, and cash-constrained quantity."""

    current_equity = _current_equity(
        state=state,
        mark_price=entry_price,
    )

    stop_distance = (
        entry_price
        * config.stop_loss_percent
        / 100
    )

    if stop_distance <= 0:
        return (
            0.0,
            0.0,
        )

    risk_budget = (
        current_equity
        * config.risk_per_trade_percent
        / 100
    )

    allocation_budget = (
        current_equity
        * config.maximum_position_percent
        / 100
    )

    quantity_by_risk = (
        risk_budget
        / stop_distance
    )

    quantity_by_allocation = (
        allocation_budget
        / entry_price
    )

    quantity_by_cash = (
        state.cash
        / entry_price
    )

    quantity = _normalize_quantity(
        min(
            quantity_by_risk,
            quantity_by_allocation,
            quantity_by_cash,
        ),
        allow_fractional=(
            config.allow_fractional
        ),
    )

    for _ in range(
        12
    ):
        if quantity <= 0:
            return (
                0.0,
                0.0,
            )

        entry_value = (
            quantity
            * entry_price
        )

        entry_fee = _calculate_fee(
            transaction_value=entry_value,
            config=config,
        )

        total_cost = (
            entry_value
            + entry_fee
        )

        if total_cost <= state.cash + 1e-9:
            return (
                quantity,
                entry_fee,
            )

        affordable_quantity = max(
            (
                state.cash
                - entry_fee
            )
            / entry_price,
            0.0,
        )

        new_quantity = _normalize_quantity(
            min(
                quantity,
                affordable_quantity,
            ),
            allow_fractional=(
                config.allow_fractional
            ),
        )

        if new_quantity >= quantity:
            decrement = (
                0.000001
                if config.allow_fractional
                else 1.0
            )

            new_quantity = (
                _normalize_quantity(
                    quantity
                    - decrement,
                    allow_fractional=(
                        config.allow_fractional
                    ),
                )
            )

        quantity = new_quantity

    return (
        0.0,
        0.0,
    )


def _public_position(
    position: _MutablePosition | None,
) -> BacktestPosition | None:
    """Convert an internal position into the public model."""

    if position is None:
        return None

    return BacktestPosition(
        ticker=position.ticker,
        entry_timestamp=(
            position.entry_timestamp
        ),
        entry_bar_index=(
            position.entry_bar_index
        ),
        quantity=position.quantity,
        entry_price=position.entry_price,
        entry_fee=position.entry_fee,
        stop_loss_price=(
            position.stop_loss_price
        ),
        take_profit_price=(
            position.take_profit_price
        ),
        signal_reason=(
            position.signal_reason
        ),
    )


def _open_position(
    *,
    state: _BacktestState,
    ticker: str,
    timestamp: str,
    bar_index: int,
    raw_open_price: float,
    signal: BacktestSignal,
    config: BacktestConfig,
) -> bool:
    """Open one long position at the current Open."""

    if state.position is not None:
        state.skipped_signals += 1
        return False

    entry_price = _apply_buy_slippage(
        raw_price=raw_open_price,
        config=config,
    )

    quantity, entry_fee = (
        _calculate_entry_quantity(
            state=state,
            entry_price=entry_price,
            config=config,
        )
    )

    if quantity <= 0:
        state.rejected_signals += 1
        return False

    entry_value = (
        quantity
        * entry_price
    )

    total_cost = (
        entry_value
        + entry_fee
    )

    if total_cost > state.cash + 1e-9:
        state.rejected_signals += 1
        return False

    state.cash = _round_money(
        state.cash
        - total_cost
    )

    state.position = _MutablePosition(
        ticker=ticker,
        entry_timestamp=timestamp,
        entry_bar_index=bar_index,
        quantity=quantity,
        entry_price=entry_price,
        entry_fee=entry_fee,
        stop_loss_price=_round_price(
            entry_price
            * (
                1
                - config.stop_loss_percent
                / 100
            )
        ),
        take_profit_price=_round_price(
            entry_price
            * (
                1
                + config.take_profit_percent
                / 100
            )
        ),
        signal_reason=str(
            signal.reason
        ),
    )

    return True


def _close_position(
    *,
    state: _BacktestState,
    timestamp: str,
    bar_index: int,
    raw_exit_price: float,
    exit_reason: str,
    config: BacktestConfig,
) -> BacktestTrade:
    """Close the current long position."""

    position = state.position

    if position is None:
        raise RuntimeError(
            "No open position to close."
        )

    exit_price = _apply_sell_slippage(
        raw_price=raw_exit_price,
        config=config,
    )

    exit_value = (
        position.quantity
        * exit_price
    )

    exit_fee = _calculate_fee(
        transaction_value=exit_value,
        config=config,
    )

    state.cash = _round_money(
        state.cash
        + exit_value
        - exit_fee
    )

    gross_pnl = (
        exit_price
        - position.entry_price
    ) * position.quantity

    total_fees = (
        position.entry_fee
        + exit_fee
    )

    net_pnl = (
        gross_pnl
        - total_fees
    )

    entry_value = (
        position.entry_price
        * position.quantity
    )

    return_percent = (
        net_pnl
        / entry_value
        * 100
        if entry_value > 0
        else 0.0
    )

    trade = BacktestTrade(
        ticker=position.ticker,
        entry_timestamp=(
            position.entry_timestamp
        ),
        exit_timestamp=timestamp,
        entry_bar_index=(
            position.entry_bar_index
        ),
        exit_bar_index=bar_index,
        quantity=position.quantity,
        entry_price=position.entry_price,
        exit_price=exit_price,
        entry_fee=position.entry_fee,
        exit_fee=exit_fee,
        total_fees=_round_money(
            total_fees
        ),
        gross_pnl=_round_money(
            gross_pnl
        ),
        net_pnl=_round_money(
            net_pnl
        ),
        return_percent=round(
            return_percent,
            4,
        ),
        holding_period_bars=max(
            bar_index
            - position.entry_bar_index,
            0,
        ),
        exit_reason=exit_reason,
        signal_reason=(
            position.signal_reason
        ),
    )

    state.trades.append(
        trade
    )

    state.realized_pnl = (
        _round_money(
            state.realized_pnl
            + trade.net_pnl
        )
    )

    state.position = None
    state.pending_exit = None

    return trade


def _check_gap_exit(
    *,
    state: _BacktestState,
    timestamp: str,
    bar_index: int,
    open_price: float,
    config: BacktestConfig,
) -> bool:
    """Check whether the market opened beyond stop or target."""

    position = state.position

    if position is None:
        return False

    if open_price <= position.stop_loss_price:
        _close_position(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            raw_exit_price=open_price,
            exit_reason="GAP_STOP_LOSS",
            config=config,
        )

        return True

    if open_price >= position.take_profit_price:
        _close_position(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            raw_exit_price=open_price,
            exit_reason="GAP_TAKE_PROFIT",
            config=config,
        )

        return True

    return False


def _check_intrabar_exit(
    *,
    state: _BacktestState,
    timestamp: str,
    bar_index: int,
    high_price: float,
    low_price: float,
    config: BacktestConfig,
) -> bool:
    """Check stop and target against the current High and Low."""

    position = state.position

    if position is None:
        return False

    stop_touched = (
        low_price
        <= position.stop_loss_price
    )

    target_touched = (
        high_price
        >= position.take_profit_price
    )

    if (
        stop_touched
        and target_touched
    ):
        _close_position(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            raw_exit_price=(
                position.stop_loss_price
            ),
            exit_reason=(
                "STOP_AND_TARGET_SAME_BAR"
            ),
            config=config,
        )

        return True

    if stop_touched:
        _close_position(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            raw_exit_price=(
                position.stop_loss_price
            ),
            exit_reason="STOP_LOSS",
            config=config,
        )

        return True

    if target_touched:
        _close_position(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            raw_exit_price=(
                position.take_profit_price
            ),
            exit_reason="TAKE_PROFIT",
            config=config,
        )

        return True

    return False


def _queue_signal(
    *,
    state: _BacktestState,
    signal: BacktestSignal,
    bar_index: int,
    ticker: str,
) -> None:
    """Queue BUY or EXIT for next-bar Open execution."""

    signal_ticker = (
        str(
            signal.ticker
        )
        .strip()
        .upper()
    )

    if (
        signal_ticker
        and signal_ticker != ticker
    ):
        state.rejected_signals += 1
        return

    action = (
        str(
            signal.action
        )
        .strip()
        .upper()
    )

    if action == "BUY":
        if (
            state.position is not None
            or state.pending_buy is not None
        ):
            state.skipped_signals += 1
            return

        state.pending_buy = (
            _PendingOrder(
                signal=signal,
                submitted_bar_index=(
                    bar_index
                ),
            )
        )

        return

    if action == "EXIT":
        if (
            state.position is None
            or state.pending_exit is not None
        ):
            state.skipped_signals += 1
            return

        state.pending_exit = (
            _PendingOrder(
                signal=signal,
                submitted_bar_index=(
                    bar_index
                ),
            )
        )

        return

    if action not in {
        "HOLD",
        "NONE",
        "",
    }:
        state.rejected_signals += 1


def _execute_pending_orders_at_open(
    *,
    state: _BacktestState,
    ticker: str,
    timestamp: str,
    bar_index: int,
    open_price: float,
    config: BacktestConfig,
) -> None:
    """Execute signals created after the previous bar closed."""

    if state.pending_exit is not None:
        if (
            state.pending_exit
            .submitted_bar_index
            < bar_index
        ):
            if state.position is not None:
                _close_position(
                    state=state,
                    timestamp=timestamp,
                    bar_index=bar_index,
                    raw_exit_price=open_price,
                    exit_reason=(
                        "EXIT_SIGNAL_NEXT_OPEN"
                    ),
                    config=config,
                )

            else:
                state.skipped_signals += 1
                state.pending_exit = None

    if state.pending_buy is not None:
        if (
            state.pending_buy
            .submitted_bar_index
            < bar_index
        ):
            pending_buy = (
                state.pending_buy
            )

            state.pending_buy = None

            if state.position is None:
                _open_position(
                    state=state,
                    ticker=ticker,
                    timestamp=timestamp,
                    bar_index=bar_index,
                    raw_open_price=open_price,
                    signal=pending_buy.signal,
                    config=config,
                )

            else:
                state.skipped_signals += 1


def _record_equity(
    *,
    state: _BacktestState,
    timestamp: str,
    close_price: float,
    replace_last: bool = False,
) -> None:
    """Record marked-to-market equity and drawdown."""

    positions_market_value = (
        _position_market_value(
            state.position,
            close_price,
        )
    )

    unrealized_pnl = 0.0
    open_position_count = 0

    if state.position is not None:
        unrealized_pnl = (
            (
                close_price
                - state.position.entry_price
            )
            * state.position.quantity
            - state.position.entry_fee
        )

        open_position_count = 1

    total_equity = (
        state.cash
        + positions_market_value
    )

    state.peak_equity = max(
        state.peak_equity,
        total_equity,
    )

    drawdown_amount = max(
        state.peak_equity
        - total_equity,
        0.0,
    )

    drawdown_percent = (
        drawdown_amount
        / state.peak_equity
        * 100
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

    point = BacktestEquityPoint(
        timestamp=timestamp,
        cash=_round_money(
            state.cash
        ),
        positions_market_value=_round_money(
            positions_market_value
        ),
        total_equity=_round_money(
            total_equity
        ),
        realized_pnl=_round_money(
            state.realized_pnl
        ),
        unrealized_pnl=_round_money(
            unrealized_pnl
        ),
        open_position_count=(
            open_position_count
        ),
        peak_equity=_round_money(
            state.peak_equity
        ),
        drawdown_amount=_round_money(
            drawdown_amount
        ),
        drawdown_percent=round(
            drawdown_percent,
            4,
        ),
    )

    if (
        replace_last
        and state.equity_curve
    ):
        state.equity_curve[-1] = point

    else:
        state.equity_curve.append(
            point
        )


def _calculate_trade_statistics(
    trades: list[BacktestTrade],
) -> dict[str, float | int]:
    """Calculate completed-trade statistics."""

    winning_trades = [
        trade
        for trade in trades
        if trade.net_pnl > 0
    ]

    losing_trades = [
        trade
        for trade in trades
        if trade.net_pnl < 0
    ]

    gross_profit = sum(
        trade.net_pnl
        for trade in winning_trades
    )

    gross_loss = abs(
        sum(
            trade.net_pnl
            for trade in losing_trades
        )
    )

    if gross_loss == 0:
        profit_factor = (
            inf
            if gross_profit > 0
            else 0.0
        )

    else:
        profit_factor = (
            gross_profit
            / gross_loss
        )

    completed_trades = len(
        trades
    )

    win_rate_percent = (
        len(
            winning_trades
        )
        / completed_trades
        * 100
        if completed_trades
        else 0.0
    )

    total_fees = sum(
        trade.total_fees
        for trade in trades
    )

    return {
        "completed_trades": (
            completed_trades
        ),
        "winning_trades": len(
            winning_trades
        ),
        "losing_trades": len(
            losing_trades
        ),
        "win_rate_percent": round(
            win_rate_percent,
            4,
        ),
        "gross_profit": _round_money(
            gross_profit
        ),
        "gross_loss": _round_money(
            gross_loss
        ),
        "profit_factor": (
            round(
                profit_factor,
                4,
            )
            if isfinite(
                profit_factor
            )
            else inf
        ),
        "total_fees": _round_money(
            total_fees
        ),
    }


def run_backtest(
    *,
    ticker: str,
    data: pd.DataFrame,
    signal_provider: SignalProvider,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a causal deterministic long-only backtest."""

    normalized_ticker = (
        ticker.strip()
        .upper()
    )

    if not normalized_ticker:
        raise ValueError(
            "ticker cannot be empty."
        )

    if not callable(
        signal_provider
    ):
        raise TypeError(
            "signal_provider must be callable."
        )

    active_config = (
        config
        if config is not None
        else BacktestConfig()
    )

    active_config.validate()

    if (
        active_config
        .maximum_open_positions
        != 1
    ):
        raise ValueError(
            "Backtest Engine V1 supports "
            "exactly one open position."
        )

    market_data = (
        _validate_market_data(
            data
        )
    )

    started_at = utc_now()

    state = _BacktestState(
        cash=_round_money(
            active_config.initial_cash
        ),
        position=None,
        pending_buy=None,
        pending_exit=None,
        trades=[],
        equity_curve=[],
        realized_pnl=0.0,
        peak_equity=float(
            active_config.initial_cash
        ),
        maximum_drawdown_amount=0.0,
        maximum_drawdown_percent=0.0,
        rejected_signals=0,
        skipped_signals=0,
    )

    for bar_index in range(
        len(
            market_data
        )
    ):
        row = market_data.iloc[
            bar_index
        ]

        timestamp = (
            _timestamp_to_string(
                market_data.index[
                    bar_index
                ]
            )
        )

        open_price = float(
            row["Open"]
        )

        high_price = float(
            row["High"]
        )

        low_price = float(
            row["Low"]
        )

        close_price = float(
            row["Close"]
        )

        # Protective stop and target remain active overnight.
        _check_gap_exit(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            open_price=open_price,
            config=active_config,
        )

        # Previous close signals execute at today's Open.
        _execute_pending_orders_at_open(
            state=state,
            ticker=normalized_ticker,
            timestamp=timestamp,
            bar_index=bar_index,
            open_price=open_price,
            config=active_config,
        )

        # Position is exposed to today's High and Low.
        _check_intrabar_exit(
            state=state,
            timestamp=timestamp,
            bar_index=bar_index,
            high_price=high_price,
            low_price=low_price,
            config=active_config,
        )

        # No future bars are visible.
        visible_data = (
            market_data.iloc[
                : bar_index + 1
            ]
        )

        signal = signal_provider(
            visible_data,
            bar_index,
            normalized_ticker,
        )

        if signal is not None:
            _queue_signal(
                state=state,
                signal=signal,
                bar_index=bar_index,
                ticker=normalized_ticker,
            )

        _record_equity(
            state=state,
            timestamp=timestamp,
            close_price=close_price,
        )

    final_bar_index = (
        len(
            market_data
        )
        - 1
    )

    final_timestamp = (
        _timestamp_to_string(
            market_data.index[-1]
        )
    )

    final_close = float(
        market_data.iloc[-1][
            "Close"
        ]
    )

    # A final-bar signal has no next Open.
    if state.pending_buy is not None:
        state.skipped_signals += 1
        state.pending_buy = None

    if state.pending_exit is not None:
        state.skipped_signals += 1
        state.pending_exit = None

    if (
        state.position is not None
        and active_config.force_close_at_end
    ):
        _close_position(
            state=state,
            timestamp=final_timestamp,
            bar_index=final_bar_index,
            raw_exit_price=final_close,
            exit_reason="END_OF_DATA",
            config=active_config,
        )

        _record_equity(
            state=state,
            timestamp=final_timestamp,
            close_price=final_close,
            replace_last=True,
        )

    final_equity = _current_equity(
        state=state,
        mark_price=final_close,
    )

    total_return_amount = (
        final_equity
        - active_config.initial_cash
    )

    total_return_percent = (
        total_return_amount
        / active_config.initial_cash
        * 100
    )

    statistics = (
        _calculate_trade_statistics(
            state.trades
        )
    )

    return BacktestResult(
        ticker=normalized_ticker,
        config=active_config,
        start_date=_timestamp_to_string(
            market_data.index[0]
        ),
        end_date=final_timestamp,
        started_at=started_at,
        finished_at=utc_now(),
        initial_cash=_round_money(
            active_config.initial_cash
        ),
        ending_cash=_round_money(
            state.cash
        ),
        ending_equity=_round_money(
            final_equity
        ),
        total_return_amount=_round_money(
            total_return_amount
        ),
        total_return_percent=round(
            total_return_percent,
            4,
        ),
        maximum_drawdown_amount=_round_money(
            state.maximum_drawdown_amount
        ),
        maximum_drawdown_percent=round(
            state.maximum_drawdown_percent,
            4,
        ),
        completed_trades=int(
            statistics[
                "completed_trades"
            ]
        ),
        winning_trades=int(
            statistics[
                "winning_trades"
            ]
        ),
        losing_trades=int(
            statistics[
                "losing_trades"
            ]
        ),
        win_rate_percent=float(
            statistics[
                "win_rate_percent"
            ]
        ),
        gross_profit=float(
            statistics[
                "gross_profit"
            ]
        ),
        gross_loss=float(
            statistics[
                "gross_loss"
            ]
        ),
        profit_factor=float(
            statistics[
                "profit_factor"
            ]
        ),
        total_fees=float(
            statistics[
                "total_fees"
            ]
        ),
        open_position=_public_position(
            state.position
        ),
        rejected_signals=(
            state.rejected_signals
        ),
        skipped_signals=(
            state.skipped_signals
        ),
        trades=state.trades,
        equity_curve=state.equity_curve,
    )


def print_backtest_result(
    result: BacktestResult,
) -> None:
    """Print a compact backtest summary."""

    profit_factor_text = (
        f"{result.profit_factor:.4f}"
        if isfinite(
            result.profit_factor
        )
        else "INF"
    )

    print()
    print("=" * 92)
    print("BACKTEST RESULT")
    print("=" * 92)

    print(
        f"Ticker:                    "
        f"{result.ticker}"
    )

    print(
        f"Period start:              "
        f"{result.start_date}"
    )

    print(
        f"Period end:                "
        f"{result.end_date}"
    )

    print()

    print(
        f"Initial capital:           "
        f"{result.initial_cash:,.2f}"
    )

    print(
        f"Ending equity:             "
        f"{result.ending_equity:,.2f}"
    )

    print(
        f"Total return:              "
        f"{result.total_return_amount:+,.2f}"
    )

    print(
        f"Total return %:            "
        f"{result.total_return_percent:+.4f}%"
    )

    print()

    print(
        f"Completed trades:          "
        f"{result.completed_trades}"
    )

    print(
        f"Winning trades:            "
        f"{result.winning_trades}"
    )

    print(
        f"Losing trades:             "
        f"{result.losing_trades}"
    )

    print(
        f"Win rate:                  "
        f"{result.win_rate_percent:.4f}%"
    )

    print(
        f"Profit factor:             "
        f"{profit_factor_text}"
    )

    print(
        f"Gross profit:              "
        f"{result.gross_profit:+,.2f}"
    )

    print(
        f"Gross loss:                "
        f"{result.gross_loss:,.2f}"
    )

    print(
        f"Total fees:                "
        f"{result.total_fees:,.2f}"
    )

    print()

    print(
        f"Maximum drawdown:          "
        f"{result.maximum_drawdown_amount:,.2f}"
    )

    print(
        f"Maximum drawdown %:        "
        f"{result.maximum_drawdown_percent:.4f}%"
    )

    print(
        "Open position:             "
        + (
            "YES"
            if result.open_position is not None
            else "NO"
        )
    )

    print(
        f"Rejected signals:          "
        f"{result.rejected_signals}"
    )

    print(
        f"Skipped signals:           "
        f"{result.skipped_signals}"
    )

    print("=" * 92)


def print_backtest_trades(
    result: BacktestResult,
) -> None:
    """Print all completed backtest trades."""

    print()
    print("=" * 132)
    print("BACKTEST TRADES")
    print("=" * 132)

    if not result.trades:
        print("No completed trades.")
        print("=" * 132)
        return

    print(
        f"{'#':>4}"
        f"{'Entry':<27}"
        f"{'Exit':<27}"
        f"{'Qty':>11}"
        f"{'Entry Px':>13}"
        f"{'Exit Px':>13}"
        f"{'Net PnL':>13}"
        f"{'Return %':>11}"
        f"{'Reason':>22}"
    )

    print("-" * 132)

    for index, trade in enumerate(
        result.trades,
        start=1,
    ):
        print(
            f"{index:>4}"
            f"{trade.entry_timestamp:<27.27}"
            f"{trade.exit_timestamp:<27.27}"
            f"{trade.quantity:>11.6f}"
            f"{trade.entry_price:>13.4f}"
            f"{trade.exit_price:>13.4f}"
            f"{trade.net_pnl:>+13.2f}"
            f"{trade.return_percent:>11.2f}"
            f"{trade.exit_reason:>22.22}"
        )

    print("=" * 132)