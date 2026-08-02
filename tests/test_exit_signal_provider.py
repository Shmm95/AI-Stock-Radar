"""Integration tests for the stateful exit signal provider.

These tests verify synchronization between:

- ExitSignalProvider
- BacktestEngine
- Pending BUY orders
- Pending EXIT orders
- Protective stop-loss exits
- Close-based trailing exits

The shared engine remains responsible for actual fills.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.backtest_engine import run_backtest
from src.backtest.backtest_models import BacktestConfig
from src.backtest.run_exit_ablation import (
    DISABLED_TARGET_PERCENT,
    ExitSignalProvider,
    ExitVariant,
)


def _market_data(
    rows: list[dict[str, float]],
) -> pd.DataFrame:
    """Create deterministic OHLC and indicator data."""

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
    force_close_at_end: bool = False,
) -> BacktestConfig:
    """Create a zero-cost deterministic configuration."""

    return BacktestConfig(
        initial_cash=10_000.0,
        risk_per_trade_percent=1.0,
        maximum_position_percent=100.0,
        stop_loss_percent=5.0,
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        allow_fractional=False,
        maximum_open_positions=1,
        force_close_at_end=force_close_at_end,
    )


def _variant(
    trailing_close_percent: float = 10.0,
) -> ExitVariant:
    """Create a Close-based trailing variant."""

    return ExitVariant(
        name=(
            "TEST_TRAILING_CLOSE_"
            f"{trailing_close_percent:g}"
        ),
        description=(
            "Synthetic provider integration test."
        ),
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        use_trend_rsi_exit=False,
        trailing_close_percent=(
            trailing_close_percent
        ),
    )


def test_trailing_exit_fills_at_next_open() -> None:
    """A trailing signal must execute at the following Open."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "EMA20": 100.0,
                "EMA50": 99.0,
                "RSI14": 55.0,
            },
            {
                "Open": 102.0,
                "High": 111.0,
                "Low": 101.0,
                "Close": 110.0,
                "EMA20": 105.0,
                "EMA50": 100.0,
                "RSI14": 60.0,
            },
            {
                "Open": 108.0,
                "High": 109.0,
                "Low": 97.5,
                "Close": 98.0,
                "EMA20": 104.0,
                "EMA50": 100.0,
                "RSI14": 60.0,
            },
            {
                "Open": 97.5,
                "High": 99.0,
                "Low": 97.0,
                "Close": 98.0,
                "EMA20": 103.0,
                "EMA50": 100.0,
                "RSI14": 60.0,
            },
        ]
    )

    config = _config()

    provider = ExitSignalProvider(
        variant=_variant(10.0),
        config=config,
    )

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=config,
    )

    assert result.completed_trades == 1
    assert result.open_position is None

    trade = result.trades[0]

    assert trade.entry_bar_index == 2
    assert trade.exit_bar_index == 4

    assert trade.entry_price == pytest.approx(
        102.0
    )

    assert trade.exit_price == pytest.approx(
        97.5
    )

    assert (
        trade.exit_reason
        == "EXIT_SIGNAL_NEXT_OPEN"
    )


def test_ema_exit_fills_at_next_open() -> None:
    """An EMA trend exit must execute at the following Open."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "EMA20": 100.0,
                "EMA50": 99.0,
                "RSI14": 55.0,
            },
            {
                "Open": 102.0,
                "High": 105.0,
                "Low": 101.0,
                "Close": 104.0,
                "EMA20": 103.0,
                "EMA50": 100.0,
                "RSI14": 60.0,
            },
            {
                "Open": 103.0,
                "High": 104.0,
                "Low": 100.0,
                "Close": 101.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
            },
        ]
    )

    config = _config()

    provider = ExitSignalProvider(
        variant=_variant(50.0),
        config=config,
    )

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=config,
    )

    assert result.completed_trades == 1
    assert result.open_position is None

    trade = result.trades[0]

    assert trade.entry_bar_index == 2
    assert trade.exit_bar_index == 4

    assert trade.entry_price == pytest.approx(
        102.0
    )

    assert trade.exit_price == pytest.approx(
        100.0
    )

    assert (
        trade.exit_reason
        == "EXIT_SIGNAL_NEXT_OPEN"
    )


def test_gap_stop_resynchronizes_and_allows_reentry() -> None:
    """A gap stop must clear mirrored state and permit a later BUY."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "EMA20": 100.0,
                "EMA50": 99.0,
                "RSI14": 55.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "EMA20": 100.0,
                "EMA50": 99.0,
                "RSI14": 55.0,
            },
            {
                "Open": 90.0,
                "High": 92.0,
                "Low": 88.0,
                "Close": 91.0,
                "EMA20": 100.0,
                "EMA50": 101.0,
                "RSI14": 40.0,
            },
            {
                "Open": 92.0,
                "High": 93.0,
                "Low": 90.0,
                "Close": 91.0,
                "EMA20": 99.0,
                "EMA50": 101.0,
                "RSI14": 40.0,
            },
            {
                "Open": 93.0,
                "High": 95.0,
                "Low": 92.0,
                "Close": 94.0,
                "EMA20": 93.0,
                "EMA50": 92.0,
                "RSI14": 55.0,
            },
            {
                "Open": 95.0,
                "High": 98.0,
                "Low": 94.0,
                "Close": 97.0,
                "EMA20": 95.0,
                "EMA50": 93.0,
                "RSI14": 58.0,
            },
        ]
    )

    config = _config()

    provider = ExitSignalProvider(
        variant=_variant(10.0),
        config=config,
    )

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=config,
    )

    assert result.completed_trades == 1

    stopped_trade = result.trades[0]

    assert (
        stopped_trade.exit_reason
        == "GAP_STOP_LOSS"
    )

    assert stopped_trade.exit_bar_index == 3
    assert stopped_trade.exit_price == pytest.approx(
        90.0
    )

    assert result.open_position is not None

    assert (
        result.open_position.entry_bar_index
        == 6
    )

    assert (
        result.open_position.entry_price
        == pytest.approx(95.0)
    )


def test_same_bar_stop_resynchronizes_and_allows_reentry() -> None:
    """A same-bar stop after entry must not leave ghost state."""

    data = _market_data(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "EMA20": 100.0,
                "EMA50": 99.0,
                "RSI14": 55.0,
            },
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 94.0,
                "Close": 96.0,
                "EMA20": 100.0,
                "EMA50": 99.0,
                "RSI14": 40.0,
            },
            {
                "Open": 97.0,
                "High": 98.0,
                "Low": 95.0,
                "Close": 96.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 40.0,
            },
            {
                "Open": 97.0,
                "High": 100.0,
                "Low": 96.0,
                "Close": 99.0,
                "EMA20": 98.0,
                "EMA50": 97.0,
                "RSI14": 55.0,
            },
            {
                "Open": 100.0,
                "High": 103.0,
                "Low": 99.0,
                "Close": 102.0,
                "EMA20": 100.0,
                "EMA50": 98.0,
                "RSI14": 58.0,
            },
        ]
    )

    config = _config()

    provider = ExitSignalProvider(
        variant=_variant(10.0),
        config=config,
    )

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=config,
    )

    assert result.completed_trades == 1

    stopped_trade = result.trades[0]

    assert stopped_trade.entry_bar_index == 2
    assert stopped_trade.exit_bar_index == 2
    assert stopped_trade.holding_period_bars == 0

    assert (
        stopped_trade.exit_reason
        == "STOP_LOSS"
    )

    assert stopped_trade.exit_price == pytest.approx(
        95.0
    )

    assert result.open_position is not None

    assert (
        result.open_position.entry_bar_index
        == 5
    )

    assert (
        result.open_position.entry_price
        == pytest.approx(100.0)
    )