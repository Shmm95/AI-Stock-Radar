"""Regression tests for the deterministic backtest engine.

These tests lock the most important execution assumptions:

1. BUY signals execute at the next bar's Open.
2. EXIT signals execute at the next bar's Open.
3. Gap stop-loss exits use the actual opening price.
4. Gap take-profit exits use the actual opening price.
5. When stop and target are touched in the same bar,
   the engine chooses stop-loss conservatively.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from src.backtest.backtest_engine import run_backtest
from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestSignal,
)


SignalProvider = Callable[
    [pd.DataFrame, int, str],
    BacktestSignal | None,
]


def _market_data(
    rows: list[dict[str, float]],
) -> pd.DataFrame:
    """Create valid daily OHLC test data."""

    index = pd.date_range(
        start="2026-01-01",
        periods=len(rows),
        freq="D",
    )

    return pd.DataFrame(
        rows,
        index=index,
    )


def _config(
    *,
    stop_loss_percent: float = 5.0,
    take_profit_percent: float = 10.0,
    force_close_at_end: bool = True,
) -> BacktestConfig:
    """Create a zero-cost deterministic test configuration."""

    return BacktestConfig(
        initial_cash=10_000.0,
        risk_per_trade_percent=1.0,
        maximum_position_percent=100.0,
        stop_loss_percent=stop_loss_percent,
        take_profit_percent=take_profit_percent,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        allow_fractional=False,
        maximum_open_positions=1,
        force_close_at_end=force_close_at_end,
    )


def _signal(
    *,
    visible_data: pd.DataFrame,
    ticker: str,
    action: str,
    reason: str,
) -> BacktestSignal:
    """Create one deterministic historical signal."""

    return BacktestSignal(
        timestamp=(
            visible_data.index[-1].isoformat()
        ),
        ticker=ticker,
        action=action,
        reference_price=float(
            visible_data["Close"].iloc[-1]
        ),
        overall_score=80.0,
        technical_score=80.0,
        confidence=80.0,
        reason=reason,
    )


def test_buy_signal_executes_at_next_bar_open() -> None:
    """BUY created after bar zero must enter at bar one's Open."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
            },
            {
                "Open": 105.0,
                "High": 106.0,
                "Low": 104.0,
                "Close": 105.0,
            },
        ]
    )

    def provider(
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        if bar_index == 0:
            return _signal(
                visible_data=visible_data,
                ticker=ticker,
                action="BUY",
                reason="Next-open BUY test.",
            )

        return None

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=_config(
            stop_loss_percent=20.0,
            take_profit_percent=20.0,
            force_close_at_end=False,
        ),
    )

    assert result.completed_trades == 0
    assert result.open_position is not None

    assert (
        result.open_position.entry_timestamp
        == data.index[1].isoformat()
    )

    assert (
        result.open_position.entry_bar_index
        == 1
    )

    assert (
        result.open_position.entry_price
        == pytest.approx(105.0)
    )

    assert result.open_position.quantity > 0


def test_exit_signal_executes_at_next_bar_open() -> None:
    """EXIT created after bar one must exit at bar two's Open."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
            },
            {
                "Open": 105.0,
                "High": 106.0,
                "Low": 104.0,
                "Close": 105.0,
            },
            {
                "Open": 110.0,
                "High": 111.0,
                "Low": 109.0,
                "Close": 110.0,
            },
        ]
    )

    def provider(
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        if bar_index == 0:
            return _signal(
                visible_data=visible_data,
                ticker=ticker,
                action="BUY",
                reason="Next-open BUY before EXIT test.",
            )

        if bar_index == 1:
            return _signal(
                visible_data=visible_data,
                ticker=ticker,
                action="EXIT",
                reason="Next-open EXIT test.",
            )

        return None

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=_config(
            stop_loss_percent=20.0,
            take_profit_percent=50.0,
        ),
    )

    assert result.completed_trades == 1
    assert result.open_position is None

    trade = result.trades[0]

    assert (
        trade.entry_timestamp
        == data.index[1].isoformat()
    )

    assert (
        trade.exit_timestamp
        == data.index[2].isoformat()
    )

    assert trade.entry_bar_index == 1
    assert trade.exit_bar_index == 2

    assert trade.entry_price == pytest.approx(
        105.0
    )

    assert trade.exit_price == pytest.approx(
        110.0
    )

    assert (
        trade.exit_reason
        == "EXIT_SIGNAL_NEXT_OPEN"
    )


def test_gap_stop_loss_uses_actual_open_price() -> None:
    """A gap below stop must exit at Open, not at the stop price."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
            },
            {
                "Open": 90.0,
                "High": 92.0,
                "Low": 88.0,
                "Close": 91.0,
            },
        ]
    )

    def provider(
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        if bar_index == 0:
            return _signal(
                visible_data=visible_data,
                ticker=ticker,
                action="BUY",
                reason="Gap stop-loss test.",
            )

        return None

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=_config(
            stop_loss_percent=5.0,
            take_profit_percent=50.0,
        ),
    )

    assert result.completed_trades == 1

    trade = result.trades[0]

    assert trade.entry_price == pytest.approx(
        100.0
    )

    # Stop level would be 95. The actual gap Open is 90.
    assert trade.exit_price == pytest.approx(
        90.0
    )

    assert (
        trade.exit_reason
        == "GAP_STOP_LOSS"
    )

    assert trade.return_percent < -5.0


def test_gap_take_profit_uses_actual_open_price() -> None:
    """A gap above target must exit at Open, not at target price."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
            },
            {
                "Open": 120.0,
                "High": 122.0,
                "Low": 119.0,
                "Close": 121.0,
            },
        ]
    )

    def provider(
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        if bar_index == 0:
            return _signal(
                visible_data=visible_data,
                ticker=ticker,
                action="BUY",
                reason="Gap take-profit test.",
            )

        return None

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=_config(
            stop_loss_percent=50.0,
            take_profit_percent=10.0,
        ),
    )

    assert result.completed_trades == 1

    trade = result.trades[0]

    assert trade.entry_price == pytest.approx(
        100.0
    )

    # Target would be 110. The actual gap Open is 120.
    assert trade.exit_price == pytest.approx(
        120.0
    )

    assert (
        trade.exit_reason
        == "GAP_TAKE_PROFIT"
    )

    assert trade.return_percent > 10.0


def test_same_bar_stop_and_target_selects_stop() -> None:
    """When both levels are touched, the conservative stop must win."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
            },
            {
                "Open": 100.0,
                "High": 112.0,
                "Low": 94.0,
                "Close": 105.0,
            },
        ]
    )

    def provider(
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        if bar_index == 0:
            return _signal(
                visible_data=visible_data,
                ticker=ticker,
                action="BUY",
                reason=(
                    "Same-bar stop and target test."
                ),
            )

        return None

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=_config(
            stop_loss_percent=5.0,
            take_profit_percent=10.0,
        ),
    )

    assert result.completed_trades == 1
    assert result.open_position is None

    trade = result.trades[0]

    assert trade.entry_price == pytest.approx(
        100.0
    )

    assert trade.exit_price == pytest.approx(
        95.0
    )

    assert (
        trade.exit_reason
        == "STOP_AND_TARGET_SAME_BAR"
    )

    assert trade.holding_period_bars == 0
    assert trade.net_pnl < 0