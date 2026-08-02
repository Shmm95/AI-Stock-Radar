"""Tests for the allocation-matched buy-and-hold benchmark."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.backtest_models import BacktestConfig
from src.backtest.benchmark import (
    calculate_effective_allocation_percent,
    run_buy_and_hold_benchmark,
)


def _data() -> pd.DataFrame:
    """Create simple rising benchmark data."""

    return pd.DataFrame(
        {
            "Open": [
                100.0,
                105.0,
                110.0,
            ],
            "Close": [
                100.0,
                105.0,
                110.0,
            ],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="D",
        ),
    )


def _config(
    *,
    risk: float = 1.0,
    maximum_position: float = 25.0,
    stop: float = 5.0,
) -> BacktestConfig:
    """Create a zero-cost benchmark configuration."""

    return BacktestConfig(
        initial_cash=10_000.0,
        risk_per_trade_percent=risk,
        maximum_position_percent=maximum_position,
        stop_loss_percent=stop,
        take_profit_percent=10.0,
        commission_rate=0.0,
        minimum_fee=0.0,
        slippage_bps=0.0,
        allow_fractional=False,
        maximum_open_positions=1,
        force_close_at_end=True,
    )


def test_effective_allocation_uses_risk_limit() -> None:
    """1% risk and 5% stop should create 20% exposure."""

    config = _config(
        risk=1.0,
        maximum_position=25.0,
        stop=5.0,
    )

    allocation = (
        calculate_effective_allocation_percent(
            config
        )
    )

    assert allocation == pytest.approx(
        20.0
    )


def test_effective_allocation_respects_position_cap() -> None:
    """Maximum position must cap risk-based exposure."""

    config = _config(
        risk=2.0,
        maximum_position=10.0,
        stop=5.0,
    )

    allocation = (
        calculate_effective_allocation_percent(
            config
        )
    )

    assert allocation == pytest.approx(
        10.0
    )


def test_matched_benchmark_uses_only_twenty_percent() -> None:
    """Default test settings should buy 20 shares at 100."""

    result = run_buy_and_hold_benchmark(
        ticker="TEST",
        data=_data(),
        config=_config(),
    )

    assert (
        result.target_allocation_percent
        == pytest.approx(20.0)
    )

    assert (
        result.actual_allocation_percent
        == pytest.approx(20.0)
    )

    assert result.quantity == pytest.approx(
        20.0
    )

    assert result.allocated_capital == pytest.approx(
        2_000.0
    )

    assert (
        result.remaining_cash_after_entry
        == pytest.approx(8_000.0)
    )

    # Twenty shares gain 10 each: total account gain = 200.
    assert result.total_return_amount == pytest.approx(
        200.0
    )

    assert result.total_return_percent == pytest.approx(
        2.0
    )

    assert (
        result.invested_capital_return_percent
        == pytest.approx(10.0)
    )