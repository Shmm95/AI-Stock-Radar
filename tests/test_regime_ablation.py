"""Tests for market-regime ablation."""

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
from src.backtest.run_regime_ablation import (
    REGIME_VARIANTS,
    RegimeFilteredProvider,
    build_regime_allowed,
)


def _variant(name: str):
    """Find one regime variant by name."""

    return next(
        variant
        for variant in REGIME_VARIANTS
        if variant.name == name
    )


def test_regime_alignment_does_not_backfill_future_data() -> None:
    """Dates before the first regime observation must remain disabled."""

    asset_index = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ]
    )

    regime_data = pd.DataFrame(
        {
            "MarketClose": [
                110.0,
                112.0,
            ],
            "MarketEMA50": [
                105.0,
                106.0,
            ],
            "MarketEMA200": [
                100.0,
                101.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-04",
            ]
        ),
    )

    allowed = build_regime_allowed(
        asset_index=asset_index,
        regime_data=regime_data,
        variant=_variant(
            "MARKET_ABOVE_EMA200"
        ),
    )

    assert allowed.iloc[0] == False
    assert allowed.iloc[1] == True
    assert allowed.iloc[2] == True
    assert allowed.iloc[3] == True


def test_bull_trend_requires_both_conditions() -> None:
    """Bull trend requires price and EMA50 above EMA200."""

    asset_index = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    regime_data = pd.DataFrame(
        {
            "MarketClose": [
                110.0,
                110.0,
                90.0,
            ],
            "MarketEMA50": [
                95.0,
                105.0,
                105.0,
            ],
            "MarketEMA200": [
                100.0,
                100.0,
                100.0,
            ],
        },
        index=asset_index,
    )

    allowed = build_regime_allowed(
        asset_index=asset_index,
        regime_data=regime_data,
        variant=_variant(
            "MARKET_BULL_TREND"
        ),
    )

    assert allowed.tolist() == [
        False,
        True,
        False,
    ]


def test_regime_gate_blocks_entry_until_allowed() -> None:
    """An existing stock setup becomes tradable when the gate opens."""

    index = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="D",
    )

    data = pd.DataFrame(
        [
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "EMA20": 99.0,
                "EMA50": 100.0,
                "RSI14": 50.0,
                "RegimeAllowed": False,
            },
            {
                "Open": 100.0,
                "High": 103.0,
                "Low": 99.0,
                "Close": 102.0,
                "EMA20": 101.0,
                "EMA50": 100.0,
                "RSI14": 55.0,
                "RegimeAllowed": False,
            },
            {
                "Open": 102.0,
                "High": 104.0,
                "Low": 101.0,
                "Close": 103.0,
                "EMA20": 102.0,
                "EMA50": 100.0,
                "RSI14": 55.0,
                "RegimeAllowed": True,
            },
            {
                "Open": 104.0,
                "High": 106.0,
                "Low": 103.0,
                "Close": 105.0,
                "EMA20": 103.0,
                "EMA50": 100.0,
                "RSI14": 58.0,
                "RegimeAllowed": True,
            },
        ],
        index=index,
    )

    config = BacktestConfig(
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
        force_close_at_end=False,
    )

    trailing_variant = ExitVariant(
        name="TEST_TRAILING",
        description="Synthetic test.",
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        use_trend_rsi_exit=False,
        trailing_close_percent=7.5,
    )

    base_provider = ExitSignalProvider(
        variant=trailing_variant,
        config=config,
    )

    provider = RegimeFilteredProvider(
        base_provider=base_provider,
        enabled=True,
    )

    result = run_backtest(
        ticker="TEST",
        data=data,
        signal_provider=provider,
        config=config,
    )

    assert result.completed_trades == 0
    assert result.open_position is not None

    assert (
        result.open_position.entry_bar_index
        == 3
    )

    assert (
        result.open_position.entry_price
        == pytest.approx(104.0)
    )