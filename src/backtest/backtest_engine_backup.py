"""Deterministic long-only backtest engine for AI-Stock-Radar.

Execution model:
- Strategy signals are evaluated after a bar closes.
- BUY signals are executed at the next bar's opening price.
- Stop-loss and take-profit are checked using each bar's High/Low.
- If stop and target are both touched in the same bar, the conservative
  assumption is used: stop-loss executes first.
- Commission and adverse slippage are included.
- Paper Trading V1 supports one open long position per ticker.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Callable
from uuid import uuid4

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
class BacktestState:
    """Mutable internal state of one backtest run."""

    cash: float
    realized_pnl: float

    open_position: BacktestPosition | None

    trades: list[BacktestTrade]
    equity_curve: list[BacktestEquityPoint]

    rejected_signals: int
    skipped_signals: int

    pending_buy_signal: BacktestSignal | None
    position_entry_bar_index: int | None

    peak_equity: float


def _round_money(value: float) -> float:
    """Round monetary values to two decimal places."""

    return round(float(value), 2)


def _round_price(value: float) -> float:
    """Round prices while retaining useful precision."""

    return round(float(value), 6)


def _create_identifier(prefix: str) -> str:
    """Create a compact unique identifier."""

    return f"{prefix}-{uuid4().hex[:12]}"


def _validate_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate, normalize, and sort OHLC market data."""

    if data.empty:
        raise ValueError(
            "Backtest market data cannot be empty."
        )

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Market data is missing required columns: "
            f"{missing_text}"
        )

    normalized = data.copy()

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

    if normalized.empty:
        raise ValueError(
            "No valid OHLC rows remain after cleaning."
        )

    invalid_rows = normalized[
        (
            normalized["Open"] <= 0
        )
        | (
            normalized["High"] <= 0
        )
        | (
            normalized["Low"] <= 0
        )
        | (
            normalized["Close"] <= 0
        )
        | (
            normalized["High"]
            < normalized["Low"]
        )
    ]

    if not invalid_rows.empty:
        raise ValueError(
            "Market data contains invalid OHLC values."
        )

    return normalized


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert a DataFrame index value to a stable string."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return str(value)


def _calculate_fee(
    *,
    transaction_value: float,
    config: BacktestConfig,
) -> float:
    """Calculate simulated broker commission."""

    if transaction_value < 0:
        raise ValueError(
            "transaction_value cannot be negative."
        )

    proportional_fee = (
        transaction_value
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
    requested_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse BUY slippage."""

    multiplier = (
        1
        + config.slippage_bps
        / 10_000
    )

    return _round_price(
        requested_price
        * multiplier
    )


def _apply_sell_slippage(
    *,
    requested_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse SELL slippage."""

    multiplier = (
        1
        - config.slippage_bps
        / 10_000
    )

    return _round_price(
        requested_price
        * multiplier
    )


def _calculate_quantity(
    *,
    cash: float,
    equity: float,
    entry_price: float,
    stop_loss: float,
    config: BacktestConfig,
) -> float:
    """Calculate risk-based and capital-limited quantity."""

    if cash <= 0:
        return 0.0

    if equity <= 0:
        return 0.0

    stop_distance = (
        entry_price
        - stop_loss
    )

    if stop_distance <= 0:
        return 0.0

    maximum_risk_amount = (
        equity
        * config.risk_per_trade_percent
        / 100
    )

    risk_based_quantity = (
        maximum_risk_amount
        / stop_distance
    )

    maximum_position_value = (
        equity
        * config.maximum_position_percent
        / 100
    )

    affordable_position_value = min(
        cash,
        maximum_position_value,
    )

    capital_based_quantity = (
        affordable_position_value
        / entry_price
    )

    raw_quantity = min(
        risk_based_quantity,
        capital_based_quantity,
    )

    if config.allow_fractional:
        quantity = round(
            raw_quantity,
            6,
        )
    else:
        quantity = float(
            floor(raw_quantity)
        )

    return max(
        quantity,
        0.0,
    )


def _current_equity(
    *,
    state: BacktestState,
    current_price: float,
) -> float:
    """Calculate current marked-to-market equity."""

    positions_market_value = 0.0

    if state.open_position is not None:
        positions_market_value = (
            state.open_position.quantity
            * current_price
        )

    return _round_money(
        state.cash
        + positions_market_value
    )


def _open_position(
    *,
    state: BacktestState,
    signal: BacktestSignal,
    ticker: str,
    timestamp: str,
    opening_price: float,
    bar_index: int,
    config: BacktestConfig,
) -> bool:
    """Execute a pending BUY signal at the current bar open."""

    if state.open_position is not None:
        state.skipped_signals += 1
        return False

    requested_entry_price = float(
        opening_price
    )

    entry_price = _apply_buy_slippage(
        requested_price=requested_entry_price,
        config=config,
    )

    stop_loss = _round_price(
        entry_price
        * (
            1
            - config.stop_loss_percent
            / 100
        )
    )

    take_profit = _round_price(
        entry_price
        * (
            1
            + config.take_profit_percent
            / 100
        )
    )

    equity = _current_equity(
        state=state,
        current_price=entry_price,
    )

    quantity = _calculate_quantity(
        cash=state.cash,
        equity=equity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        config=config,
    )

    if quantity <= 0:
        state.rejected_signals += 1
        return False

    entry_value = _round_money(
        quantity
        * entry_price
    )

    entry_fee = _calculate_fee(
        transaction_value=entry_value,
        config=config,
    )

    total_cost = _round_money(
        entry_value
        + entry_fee
    )

    if total_cost > state.cash:
        if config.allow_fractional:
            affordable_quantity = (
                max(
                    state.cash
                    - config.minimum_fee,
                    0.0,
                )
                / entry_price
            )

            quantity = round(
                affordable_quantity,
                6,
            )

        else:
            quantity = float(
                floor(
                    max(
                        state.cash
                        - config.minimum_fee,
                        0.0,
                    )
                    / entry_price
                )
            )

        if quantity <= 0:
            state.rejected_signals += 1
            return False

        entry_value = _round_money(
            quantity
            * entry_price
        )

        entry_fee = _calculate_fee(
            transaction_value=entry_value,
            config=config,
        )

        total_cost = _round_money(
            entry_value
            + entry_fee
        )

    if total_cost > state.cash:
        state.rejected_signals += 1
        return False

    initial_risk_amount = _round_money(
        quantity
        * (
            entry_price
            - stop_loss
        )
    )

    state.cash = _round_money(
        state.cash
        - total_cost
    )

    state.open_position = BacktestPosition(
        position_id=_create_identifier(
            "bt-position"
        ),
        ticker=ticker.upper(),
        quantity=quantity,
        entry_timestamp=timestamp,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_value=entry_value,
        entry_fee=entry_fee,
        initial_risk_amount=(
            initial_risk_amount
        ),
        signal_score=float(
            signal.overall_score
        ),
        signal_confidence=float(
            signal.confidence
        ),
    )

    state.position_entry_bar_index = (
        bar_index
    )

    return True


def _close_position(
    *,
    state: BacktestState,
    timestamp: str,
    requested_exit_price: float,
    current_bar_index: int,
    exit_reason: str,
    config: BacktestConfig,
) -> BacktestTrade:
    """Close the currently open position."""

    position = state.open_position

    if position is None:
        raise RuntimeError(
            "Cannot close a position because none is open."
        )

    exit_price = _apply_sell_slippage(
        requested_price=requested_exit_price,
        config=config,
    )

    exit_value = _round_money(
        position.quantity
        * exit_price
    )

    exit_fee = _calculate_fee(
        transaction_value=exit_value,
        config=config,
    )

    net_proceeds = _round_money(
        exit_value
        - exit_fee
    )

    entry_bar_index = (
        state.position_entry_bar_index
    )

    if entry_bar_index is None:
        holding_period_bars = 0
    else:
        holding_period_bars = max(
            current_bar_index
            - entry_bar_index,
            0,
        )

    trade = BacktestTrade.create(
        trade_id=_create_identifier(
            "bt-trade"
        ),
        position=position,
        exit_timestamp=timestamp,
        exit_price=exit_price,
        exit_fee=exit_fee,
        holding_period_bars=(
            holding_period_bars
        ),
        exit_reason=exit_reason,
    )

    state.cash = _round_money(
        state.cash
        + net_proceeds
    )

    state.realized_pnl = _round_money(
        state.realized_pnl
        + trade.net_pnl
    )

    state.trades.append(
        trade
    )

    state.open_position = None
    state.position_entry_bar_index = None

    return trade


def _process_intrabar_exit(
    *,
    state: BacktestState,
    timestamp: str,
    bar_high: float,
    bar_low: float,
    bar_index: int,
    config: BacktestConfig,
) -> BacktestTrade | None:
    """Process stop-loss and take-profit using current bar range."""

    position = state.open_position

    if position is None:
        return None

    stop_touched = (
        bar_low
        <= position.stop_loss
    )

    target_touched = (
        bar_high
        >= position.take_profit
    )

    if stop_touched:
        return _close_position(
            state=state,
            timestamp=timestamp,
            requested_exit_price=(
                position.stop_loss
            ),
            current_bar_index=bar_index,
            exit_reason=(
                "STOP_LOSS"
                if not target_touched
                else "STOP_AND_TARGET_SAME_BAR"
            ),
            config=config,
        )

    if target_touched:
        return _close_position(
            state=state,
            timestamp=timestamp,
            requested_exit_price=(
                position.take_profit
            ),
            current_bar_index=bar_index,
            exit_reason="TAKE_PROFIT",
            config=config,
        )

    return None


def _record_equity_point(
    *,
    state: BacktestState,
    timestamp: str,
    closing_price: float,
) -> None:
    """Record marked-to-market equity after processing the bar."""

    positions_market_value = 0.0
    unrealized_pnl = 0.0

    if state.open_position is not None:
        positions_market_value = (
            state.open_position.market_value(
                closing_price
            )
        )

        unrealized_pnl = (
            state.open_position.unrealized_pnl(
                closing_price
            )
        )

    point = BacktestEquityPoint.create(
        timestamp=timestamp,
        cash=state.cash,
        positions_market_value=(
            positions_market_value
        ),
        realized_pnl=state.realized_pnl,
        unrealized_pnl=unrealized_pnl,
        open_position_count=(
            1
            if state.open_position is not None
            else 0
        ),
        previous_peak_equity=(
            state.peak_equity
        ),
    )

    state.peak_equity = (
        point.peak_equity
    )

    state.equity_curve.append(
        point
    )


def _validate_signal(
    *,
    signal: BacktestSignal,
    ticker: str,
    timestamp: str,
) -> None:
    """Validate strategy signal consistency."""

    signal.validate()

    if signal.ticker.upper() != ticker.upper():
        raise ValueError(
            "Signal ticker does not match "
            "the backtest ticker."
        )

    if signal.timestamp != timestamp:
        raise ValueError(
            "Signal timestamp does not match "
            "the current bar timestamp."
        )


def run_backtest(
    *,
    ticker: str,
    data: pd.DataFrame,
    signal_provider: SignalProvider,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run one historical long-only backtest.

    The signal provider receives:
        data available up to the current bar,
        current bar integer index,
        ticker.

    The provider must not inspect future rows.
    A BUY signal generated on bar N is executed on bar N+1 Open.
    """

    normalized_ticker = (
        ticker.strip().upper()
    )

    if not normalized_ticker:
        raise ValueError(
            "ticker cannot be empty."
        )

    if config is None:
        config = BacktestConfig()

    config.validate()

    market_data = _validate_market_data(
        data
    )

    started_at = utc_now()

    state = BacktestState(
        cash=_round_money(
            config.initial_cash
        ),
        realized_pnl=0.0,
        open_position=None,
        trades=[],
        equity_curve=[],
        rejected_signals=0,
        skipped_signals=0,
        pending_buy_signal=None,
        position_entry_bar_index=None,
        peak_equity=_round_money(
            config.initial_cash
        ),
    )

    row_count = len(
        market_data
    )

    for bar_index in range(
        row_count
    ):
        row = market_data.iloc[
            bar_index
        ]

        timestamp = _timestamp_to_string(
            market_data.index[
                bar_index
            ]
        )

        opening_price = float(
            row["Open"]
        )

        high_price = float(
            row["High"]
        )

        low_price = float(
            row["Low"]
        )

        closing_price = float(
            row["Close"]
        )

        # -----------------------------------------------------
        # Execute yesterday's pending BUY at today's opening.
        # -----------------------------------------------------
        pending_signal = (
            state.pending_buy_signal
        )

        state.pending_buy_signal = None

        if pending_signal is not None:
            _open_position(
                state=state,
                signal=pending_signal,
                ticker=normalized_ticker,
                timestamp=timestamp,
                opening_price=opening_price,
                bar_index=bar_index,
                config=config,
            )

        # -----------------------------------------------------
        # Process today's stop-loss and take-profit.
        # Conservative assumption: stop executes first.
        # -----------------------------------------------------
        _process_intrabar_exit(
            state=state,
            timestamp=timestamp,
            bar_high=high_price,
            bar_low=low_price,
            bar_index=bar_index,
            config=config,
        )

        # -----------------------------------------------------
        # Build the historical slice visible to the strategy.
        # No future rows are provided.
        # -----------------------------------------------------
        visible_data = market_data.iloc[
            : bar_index + 1
        ].copy()

        signal = signal_provider(
            visible_data,
            bar_index,
            normalized_ticker,
        )

        if signal is not None:
            try:
                _validate_signal(
                    signal=signal,
                    ticker=normalized_ticker,
                    timestamp=timestamp,
                )

            except Exception:
                state.rejected_signals += 1
                signal = None

        if signal is not None:
            action = (
                signal.action
                .strip()
                .upper()
            )

            if action == "BUY":
                if state.open_position is None:
                    if bar_index < row_count - 1:
                        state.pending_buy_signal = (
                            signal
                        )
                    else:
                        state.skipped_signals += 1
                else:
                    state.skipped_signals += 1

            elif action == "EXIT":
                if state.open_position is not None:
                    _close_position(
                        state=state,
                        timestamp=timestamp,
                        requested_exit_price=(
                            closing_price
                        ),
                        current_bar_index=(
                            bar_index
                        ),
                        exit_reason="SIGNAL_EXIT",
                        config=config,
                    )
                else:
                    state.skipped_signals += 1

        _record_equity_point(
            state=state,
            timestamp=timestamp,
            closing_price=closing_price,
        )

    # ---------------------------------------------------------
    # Force close remaining position at final close.
    # ---------------------------------------------------------
    if (
        config.force_close_at_end
        and state.open_position is not None
    ):
        final_bar_index = (
            row_count - 1
        )

        final_timestamp = (
            _timestamp_to_string(
                market_data.index[
                    final_bar_index
                ]
            )
        )

        final_close = float(
            market_data.iloc[
                final_bar_index
            ]["Close"]
        )

        _close_position(
            state=state,
            timestamp=final_timestamp,
            requested_exit_price=final_close,
            current_bar_index=(
                final_bar_index
            ),
            exit_reason="END_OF_DATA",
            config=config,
        )

        _record_equity_point(
            state=state,
            timestamp=final_timestamp,
            closing_price=final_close,
        )

    ending_equity = state.cash

    if state.open_position is not None:
        final_close = float(
            market_data.iloc[-1][
                "Close"
            ]
        )

        ending_equity = _round_money(
            state.cash
            + state.open_position.market_value(
                final_close
            )
        )

    completed_at = utc_now()

    return BacktestResult(
        ticker=normalized_ticker,
        started_at=started_at,
        completed_at=completed_at,
        start_date=_timestamp_to_string(
            market_data.index[0]
        ),
        end_date=_timestamp_to_string(
            market_data.index[-1]
        ),
        initial_cash=_round_money(
            config.initial_cash
        ),
        ending_cash=_round_money(
            state.cash
        ),
        ending_equity=_round_money(
            ending_equity
        ),
        trades=tuple(
            state.trades
        ),
        equity_curve=tuple(
            state.equity_curve
        ),
        rejected_signals=(
            state.rejected_signals
        ),
        skipped_signals=(
            state.skipped_signals
        ),
        config=config,
    )


def print_backtest_result(
    result: BacktestResult,
) -> None:
    """Print a readable backtest summary."""

    winning_trades = [
        trade
        for trade in result.trades
        if trade.net_pnl > 0
    ]

    losing_trades = [
        trade
        for trade in result.trades
        if trade.net_pnl < 0
    ]

    total_trades = (
        result.total_trades
    )

    if total_trades == 0:
        win_rate = 0.0
        average_trade = 0.0
    else:
        win_rate = (
            len(winning_trades)
            / total_trades
            * 100
        )

        average_trade = (
            sum(
                trade.net_pnl
                for trade in result.trades
            )
            / total_trades
        )

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
        profit_factor_text = (
            "∞"
            if gross_profit > 0
            else "0.0000"
        )
    else:
        profit_factor_text = (
            f"{gross_profit / gross_loss:.4f}"
        )

    print()
    print("=" * 92)
    print("BACKTEST RESULT")
    print("=" * 92)

    print(f"Ticker:                   {result.ticker}")
    print(f"Period start:             {result.start_date}")
    print(f"Period end:               {result.end_date}")

    print()
    print(
        f"Initial capital:          "
        f"{result.initial_cash:,.2f}"
    )

    print(
        f"Ending equity:            "
        f"{result.ending_equity:,.2f}"
    )

    print(
        f"Total return:             "
        f"{result.total_return_amount:+,.2f}"
    )

    print(
        f"Total return %:           "
        f"{result.total_return_percent:+.4f}%"
    )

    print()
    print(
        f"Completed trades:         "
        f"{total_trades}"
    )

    print(
        f"Winning trades:           "
        f"{len(winning_trades)}"
    )

    print(
        f"Losing trades:            "
        f"{len(losing_trades)}"
    )

    print(
        f"Win rate:                 "
        f"{win_rate:.4f}%"
    )

    print(
        f"Average trade:            "
        f"{average_trade:+,.2f}"
    )

    print(
        f"Profit factor:            "
        f"{profit_factor_text}"
    )

    print()
    print(
        f"Maximum drawdown:         "
        f"{result.maximum_drawdown_amount:,.2f}"
    )

    print(
        f"Maximum drawdown %:       "
        f"{result.maximum_drawdown_percent:.4f}%"
    )

    print(
        f"Rejected signals:         "
        f"{result.rejected_signals}"
    )

    print(
        f"Skipped signals:          "
        f"{result.skipped_signals}"
    )

    print("=" * 92)


def print_backtest_trades(
    result: BacktestResult,
) -> None:
    """Print all completed backtest trades."""

    print()
    print("=" * 120)
    print("BACKTEST TRADES")
    print("=" * 120)

    if not result.trades:
        print("No completed trades.")
        print("=" * 120)
        return

    for rank, trade in enumerate(
        result.trades,
        start=1,
    ):
        print(
            f"{rank:>3}. "
            f"{trade.ticker:<12} "
            f"Entry: {trade.entry_price:>10.4f} "
            f"Exit: {trade.exit_price:>10.4f} "
            f"Qty: {trade.quantity:<10} "
            f"PnL: {trade.net_pnl:>+10.2f} "
            f"Return: {trade.return_percent:>+8.4f}% "
            f"Bars: {trade.holding_period_bars:<5} "
            f"Reason: {trade.exit_reason}"
        )

    print("=" * 120)
